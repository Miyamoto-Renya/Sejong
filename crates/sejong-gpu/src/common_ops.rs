use crate::{Engine, Tensor, n, p};
use sejong_core::{DType, Result, ensure, json};
impl Engine {
    pub fn layer_norm(&mut self, x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2 && w.len() == x.shape[1] && b.len() == w.len(),
            "layer norm shape mismatch",
        )?;
        let y = self.empty(&x.shape, x.dtype)?;
        let d = x.shape[1];
        self.launch(
            "layernorm",
            &[("X", p(x)), ("W", p(w)), ("B", p(b)), ("Y", p(&y))],
            json::json!({"D":d,"EPS":eps,"BLOCK":d.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok(y)
    }
    pub fn layer_norm_backward(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        dy: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dx = self.empty(&x.shape, DType::F32)?;
        let dw = self.full(&w.shape, DType::F32, 0.)?;
        let db = self.full(&b.shape, DType::F32, 0.)?;
        let d = x.shape[1];
        self.launch(
            "layernorm_backward",
            &[
                ("X", p(x)),
                ("W", p(w)),
                ("DY", p(dy)),
                ("DX", p(&dx)),
                ("DW", p(&dw)),
                ("DB", p(&db)),
            ],
            json::json!({"D":d,"EPS":eps,"BLOCK":d.next_power_of_two()}),
            [x.shape[0], 1, 1],
        )?;
        Ok((dx, dw, db))
    }
    pub fn bias_add(&mut self, x: &Tensor, b: &Tensor) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2 && b.len() == x.shape[1],
            "affine bias shape mismatch",
        )?;
        let y = self.empty(&x.shape, x.dtype)?;
        self.launch(
            "bias_add",
            &[
                ("X", p(x)),
                ("B", p(b)),
                ("Y", p(&y)),
                ("N", n(x.len())?),
                ("D", n(x.shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn bias_reduce(&mut self, x: &Tensor, shape: &[usize]) -> Result<Tensor> {
        let y = self.empty(shape, DType::F32)?;
        let d = x.shape[1];
        ensure(y.len() == d, "bias gradient shape mismatch")?;
        self.launch(
            "bias_reduce",
            &[("X", p(x)), ("Y", p(&y))],
            json::json!({"R":x.shape[0],"D":d,"BLOCK":128}),
            [d.div_ceil(128), 1, 1],
        )?;
        Ok(y)
    }
    fn pointwise(
        &mut self,
        x: &Tensor,
        z: Option<&Tensor>,
        alpha: f32,
        mode: usize,
    ) -> Result<Tensor> {
        if let Some(z) = z {
            ensure(x.shape == z.shape, "elementwise shape mismatch")?;
        }
        let y = self.empty(&x.shape, x.dtype)?;
        self.launch(
            "elementwise",
            &[
                ("X", p(x)),
                ("Z", p(z.unwrap_or(x))),
                ("Y", p(&y)),
                ("N", n(x.len())?),
            ],
            json::json!({"ALPHA":alpha,"MODE":mode,"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }
    pub fn scale(&mut self, x: &Tensor, alpha: f32) -> Result<Tensor> {
        self.pointwise(x, None, alpha, 0)
    }
    pub fn relu(&mut self, x: &Tensor) -> Result<Tensor> {
        self.pointwise(x, None, 1., 1)
    }
    pub fn relu_backward(&mut self, x: &Tensor, dy: &Tensor) -> Result<Tensor> {
        let x = self.cast(x, DType::F32)?;
        self.pointwise(&x, Some(dy), 1., 2)
    }
    pub fn mul(&mut self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        self.pointwise(x, Some(z), 1., 3)
    }
    pub fn slice_columns(&mut self, x: &Tensor, start: usize, width: usize) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2 && start + width <= x.shape[1] && width > 0,
            "column slice bounds",
        )?;
        let y = self.empty(&[x.shape[0], width], x.dtype)?;
        self.launch("columns",&[("X",p(x)),("Y",p(&y)),("R",n(x.shape[0])?)],json::json!({"SRC_D":x.shape[1],"DST_D":width,"START":start,"WIDTH":width,"INSERT":false,"BLOCK":256}),[y.len().div_ceil(256),1,1])?;
        Ok(y)
    }
    pub fn insert_columns(&mut self, x: &Tensor, y: &Tensor, start: usize) -> Result<()> {
        ensure(
            x.shape.len() == 2
                && y.shape.len() == 2
                && x.shape[0] == y.shape[0]
                && start + x.shape[1] <= y.shape[1],
            "column insert bounds",
        )?;
        self.launch("columns",&[("X",p(x)),("Y",p(y)),("R",n(x.shape[0])?)],json::json!({"SRC_D":x.shape[1],"DST_D":y.shape[1],"START":start,"WIDTH":x.shape[1],"INSERT":true,"BLOCK":256}),[x.len().div_ceil(256),1,1])
    }
    pub fn concat_columns(&mut self, xs: &[Tensor]) -> Result<Tensor> {
        ensure(
            !xs.is_empty()
                && xs.iter().all(|x| {
                    x.shape.len() == 2 && x.shape[0] == xs[0].shape[0] && x.dtype == xs[0].dtype
                }),
            "column concat shape mismatch",
        )?;
        let y = self.empty(
            &[xs[0].shape[0], xs.iter().map(|x| x.shape[1]).sum()],
            xs[0].dtype,
        )?;
        let mut start = 0;
        for x in xs {
            self.insert_columns(x, &y, start)?;
            start += x.shape[1];
        }
        Ok(y)
    }
    pub fn concat_rows(&mut self, xs: &[Tensor]) -> Result<Tensor> {
        ensure(
            !xs.is_empty()
                && xs.iter().all(|x| {
                    x.shape.len() == 2 && x.shape[1] == xs[0].shape[1] && x.dtype == xs[0].dtype
                }),
            "row concat shape mismatch",
        )?;
        let y = self.empty(
            &[xs.iter().map(|x| x.shape[0]).sum(), xs[0].shape[1]],
            xs[0].dtype,
        )?;
        let mut start = 0;
        for x in xs {
            let row = y.rows(start, x.shape[0])?;
            self.copy_into(x, &row)?;
            start += x.shape[0];
        }
        Ok(y)
    }
}
