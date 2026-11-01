use crate::parameter_vjp::{BK, BM, BN};
use crate::{Engine, Tensor, n, p};
use sejong_core::{DType, Result, json};

impl Engine {
    pub fn project_qkv(
        &mut self,
        x: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if x.dtype == DType::F64 {
            return Ok((
                self.matmul(x, wq, false, false, DType::F64)?,
                self.matmul(x, wk, false, false, DType::F64)?,
                self.matmul(x, wv, false, false, DType::F64)?,
            ));
        }
        let (rows, input, query, kv) = (x.shape[0], x.shape[1], wq.shape[1], wk.shape[1]);
        let q = self.empty(&[rows, query], x.dtype)?;
        let k = self.empty(&[rows, kv], x.dtype)?;
        let v = self.empty(&[rows, kv], x.dtype)?;
        self.launch(
            "projection_qkv",
            &[
                ("X", p(x)),
                ("WQ", p(wq)),
                ("WK", p(wk)),
                ("WV", p(wv)),
                ("Q", p(&q)),
                ("KOUT", p(&k)),
                ("VOUT", p(&v)),
                ("M", n(rows)?),
                ("N", n(query)?),
                ("K", n(input)?),
                ("KV", n(kv)?),
            ],
            json::json!({"BM":BM,"BN":BN,"BK":BK}),
            [rows.div_ceil(BM), query.div_ceil(BN), 1],
        )?;
        Ok((q, k, v))
    }

    pub fn gate_up(
        &mut self,
        x: &Tensor,
        wg: &Tensor,
        wu: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if x.dtype == DType::F64 {
            let g = self.matmul(x, wg, false, false, DType::F64)?;
            let u = self.matmul(x, wu, false, false, DType::F64)?;
            let h = self.swiglu(&g, &u)?;
            return Ok((g, u, h));
        }
        let (rows, input, width) = (x.shape[0], x.shape[1], wg.shape[1]);
        let g = self.empty(&[rows, width], x.dtype)?;
        let u = self.empty(&[rows, width], x.dtype)?;
        let h = self.empty(&[rows, width], x.dtype)?;
        self.launch(
            "projection_gate_up",
            &[
                ("X", p(x)),
                ("WG", p(wg)),
                ("WU", p(wu)),
                ("G", p(&g)),
                ("U", p(&u)),
                ("H", p(&h)),
                ("M", n(rows)?),
                ("N", n(width)?),
                ("K", n(input)?),
            ],
            json::json!({"BM":BM,"BN":BN,"BK":BK}),
            [rows.div_ceil(BM), width.div_ceil(BN), 1],
        )?;
        Ok((g, u, h))
    }
}
