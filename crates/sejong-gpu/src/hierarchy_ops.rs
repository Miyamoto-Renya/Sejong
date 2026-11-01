use crate::{Engine, Tensor, n, p};
use sejong_core::{DType, Error, Result, ensure, json};

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Message("tensor dimension product overflow".into()))
}

impl Engine {
    pub fn attention_grouped(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        tq: usize,
        tk: usize,
        offset: usize,
        query_heads: usize,
        kv_heads: usize,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        ensure(
            q.shape.len() == 2 && k.shape.len() == 2 && k.shape == v.shape,
            "grouped attention requires Q/K/V matrices",
        )?;
        ensure(
            tq > 0
                && tk > 0
                && query_heads > 0
                && kv_heads > 0
                && query_heads % kv_heads == 0
                && q.shape[1] % query_heads == 0
                && q.shape[0] % tq == 0
                && k.shape[0] % tk == 0
                && q.shape[0] / tq == k.shape[0] / tk
                && window.unwrap_or(0) <= i32::MAX as usize
                && offset.checked_add(tq).is_some_and(|end| end <= tk),
            "grouped attention sequence/head dimensions",
        )?;
        let d = q.shape[1] / query_heads;
        ensure(
            d > 0 && k.shape[1] == kv_heads * d && q.dtype == k.dtype && q.dtype == v.dtype,
            "grouped attention head width/dtype mismatch",
        )?;
        let y = self.empty(&q.shape, q.dtype)?;
        let lse = self.empty(
            &[q.shape[0] * query_heads],
            if q.dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            },
        )?;
        let bk = tk.min(32).next_power_of_two();
        self.launch(
            "hierarchy_attention",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("O", p(&y)),
                ("LSE", p(&lse)),
                ("TQ", n(tq)?),
                ("TK", n(tk)?),
                ("OFFSET", n(offset)?),
            ],
            json::json!({"HQ":query_heads,"HK":kv_heads,"D":d,"BD":d.next_power_of_two(),
                         "BK":bk,"WINDOW":window.unwrap_or(0),"FP64":q.dtype==DType::F64}),
            [q.shape[0] * query_heads, 1, 1],
        )?;
        Ok((y, lse))
    }

    pub fn attention_grouped_backward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        dy: &Tensor,
        t: usize,
        query_heads: usize,
        kv_heads: usize,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        ensure(
            q.shape.len() == 2
                && k.shape.len() == 2
                && q.shape == o.shape
                && q.shape == dy.shape
                && k.shape == v.shape
                && t > 0
                && query_heads > 0
                && kv_heads > 0
                && query_heads % kv_heads == 0
                && q.shape[0] == k.shape[0]
                && q.shape[0] % t == 0
                && q.shape[1] % query_heads == 0
                && window.unwrap_or(0) <= i32::MAX as usize,
            "grouped attention backward dimensions",
        )?;
        let d = q.shape[1] / query_heads;
        ensure(
            d > 0 && k.shape[1] == kv_heads * d && lse.len() == q.shape[0] * query_heads,
            "grouped attention backward head/LSE dimensions",
        )?;
        let dq = self.empty(&q.shape, DType::F32)?;
        let dk = self.empty(&k.shape, DType::F32)?;
        let dv = self.empty(&v.shape, DType::F32)?;
        let delta = self.empty(&lse.shape, DType::F32)?;
        let constants = json::json!({"HQ":query_heads,"HK":kv_heads,"D":d,
            "BD":d.next_power_of_two(),"BK":t.min(32).next_power_of_two(),"WINDOW":window.unwrap_or(0)});
        self.launch(
            "hierarchy_attention_dq",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("O", p(o)),
                ("LSE", p(lse)),
                ("DO", p(dy)),
                ("DQ", p(&dq)),
                ("Delta", p(&delta)),
                ("T", n(t)?),
            ],
            constants.clone(),
            [q.shape[0] * query_heads, 1, 1],
        )?;
        self.launch(
            "hierarchy_attention_dkv",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("LSE", p(lse)),
                ("DO", p(dy)),
                ("Delta", p(&delta)),
                ("DK", p(&dk)),
                ("DV", p(&dv)),
                ("T", n(t)?),
            ],
            constants,
            [k.shape[0] * kv_heads, 1, 1],
        )?;
        Ok((dq, dk, dv))
    }

    pub fn patch_pool(&mut self, x: &Tensor, sequence: usize, patch: usize) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2
                && sequence > 0
                && patch > 0
                && patch <= i32::MAX as usize
                && x.shape[0] % sequence == 0,
            "patch pool sequence dimensions",
        )?;
        let rows = x.shape[0] / sequence * sequence.div_ceil(patch);
        let d = x.shape[1];
        let y = self.empty(&[rows, d], x.dtype)?;
        let block = d.min(256).next_power_of_two();
        self.launch(
            "hierarchy_patch_pool",
            &[("X", p(x)), ("Y", p(&y)), ("T", n(sequence)?), ("D", n(d)?)],
            json::json!({"PATCH":patch,"BLOCK":block}),
            [rows, d.div_ceil(block), 1],
        )?;
        Ok(y)
    }

    pub fn patch_pool_backward(
        &mut self,
        dy: &Tensor,
        shape: &[usize],
        sequence: usize,
        patch: usize,
    ) -> Result<Tensor> {
        ensure(
            shape.len() == 2
                && sequence > 0
                && patch > 0
                && patch <= i32::MAX as usize
                && shape[0] % sequence == 0,
            "patch pool backward sequence dimensions",
        )?;
        let rows = product(shape[0] / sequence, sequence.div_ceil(patch))?;
        ensure(
            dy.shape == [rows, shape[1]],
            "patch pool backward gradient dimensions",
        )?;
        let dx = self.empty(shape, DType::F32)?;
        self.launch(
            "hierarchy_patch_pool_backward",
            &[
                ("DY", p(dy)),
                ("DX", p(&dx)),
                ("N", n(dx.len())?),
                ("T", n(sequence)?),
                ("D", n(shape[1])?),
            ],
            json::json!({"PATCH":patch,"BLOCK":256}),
            [dx.len().div_ceil(256), 1, 1],
        )?;
        Ok(dx)
    }

    pub fn patch_broadcast(
        &mut self,
        x: &Tensor,
        bos: &Tensor,
        sequence: usize,
        patch: usize,
    ) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2
                && bos.shape == [1, x.shape[1]]
                && sequence > 0
                && patch > 0
                && patch <= i32::MAX as usize
                && x.shape[0] % sequence.div_ceil(patch) == 0
                && x.dtype == bos.dtype,
            "patch broadcast sequence/BOS dimensions",
        )?;
        let rows = product(x.shape[0] / sequence.div_ceil(patch), sequence)?;
        let y = self.empty(&[rows, x.shape[1]], x.dtype)?;
        self.launch(
            "hierarchy_patch_broadcast",
            &[
                ("X", p(x)),
                ("BOS", p(bos)),
                ("Y", p(&y)),
                ("N", n(y.len())?),
                ("T", n(sequence)?),
                ("D", n(x.shape[1])?),
            ],
            json::json!({"PATCH":patch,"BLOCK":256}),
            [y.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }

    pub fn patch_broadcast_backward(
        &mut self,
        dy: &Tensor,
        shape: &[usize],
        sequence: usize,
        patch: usize,
    ) -> Result<(Tensor, Tensor)> {
        ensure(
            shape.len() == 2
                && sequence > 0
                && patch > 0
                && patch <= i32::MAX as usize
                && shape[0] % sequence.div_ceil(patch) == 0,
            "patch broadcast backward sequence dimensions",
        )?;
        let rows = product(shape[0] / sequence.div_ceil(patch), sequence)?;
        ensure(
            dy.shape == [rows, shape[1]],
            "patch broadcast backward gradient dimensions",
        )?;
        let dx = self.empty(shape, DType::F32)?;
        let dbos = self.empty(&[1, shape[1]], DType::F32)?;
        self.launch(
            "hierarchy_patch_broadcast_backward",
            &[
                ("DY", p(dy)),
                ("DX", p(&dx)),
                ("DBOS", p(&dbos)),
                ("N", n(dy.len())?),
                ("T", n(sequence)?),
                ("D", n(shape[1])?),
            ],
            json::json!({"PATCH":patch,"BLOCK":256}),
            [dx.len().div_ceil(256) + shape[1].div_ceil(256), 1, 1],
        )?;
        Ok((dx, dbos))
    }

    pub fn route_topk(&mut self, logits: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
        ensure(
            logits.shape.len() == 2 && topk > 0 && topk <= logits.shape[1],
            "router top-k dimensions",
        )?;
        let shape = [logits.shape[0], topk];
        let weights = self.empty(&shape, DType::F32)?;
        let ids = self.empty(&shape, DType::I32)?;
        self.launch(
            "hierarchy_route_topk",
            &[("X", p(logits)), ("IDS", p(&ids)), ("WEIGHTS", p(&weights))],
            json::json!({"E":logits.shape[1],"K":topk,"BLOCK":logits.shape[1].next_power_of_two()}),
            [logits.shape[0], 1, 1],
        )?;
        Ok((weights, ids))
    }

    pub fn route_topk_backward(
        &mut self,
        logits: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        dy: &Tensor,
        experts: usize,
    ) -> Result<Tensor> {
        ensure(
            ids.shape.len() == 2
                && ids.dtype == DType::I32
                && experts > 0
                && ids.shape[1] <= experts
                && ids.shape == weights.shape
                && weights.shape == dy.shape
                && logits.shape == [ids.shape[0], experts],
            "router backward dimensions",
        )?;
        let dx = self.full(&[ids.shape[0], experts], DType::F32, 0.)?;
        self.launch(
            "hierarchy_route_backward",
            &[
                ("X", p(logits)),
                ("IDS", p(ids)),
                ("WEIGHTS", p(weights)),
                ("DY", p(dy)),
                ("DX", p(&dx)),
            ],
            json::json!({"E":experts,"K":ids.shape[1],"BLOCK":experts.next_power_of_two()}),
            [ids.shape[0], 1, 1],
        )?;
        Ok(dx)
    }

    pub fn expert_routes(
        &mut self,
        ids: &Tensor,
        experts: usize,
    ) -> Result<Vec<Option<(Tensor, Tensor)>>> {
        ensure(
            ids.shape.len() == 2 && ids.dtype == DType::I32 && experts > 0,
            "expert route dimensions",
        )?;
        let capacity = ids.len();
        let token_ids = self.empty(&[experts, capacity], DType::I32)?;
        let slot_ids = self.empty(&[experts, capacity], DType::I32)?;
        let counts = self.empty(&[experts], DType::I32)?;
        self.launch(
            "hierarchy_route_pack",
            &[
                ("IDS", p(ids)),
                ("TOKENS", p(&token_ids)),
                ("SLOTS", p(&slot_ids)),
                ("COUNTS", p(&counts)),
                ("R", n(capacity)?),
            ],
            json::json!({"K":ids.shape[1],"BLOCK":256}),
            [experts, 1, 1],
        )?;
        let counts = self.to_i32(&counts)?;
        let mut result = Vec::with_capacity(experts);
        for (expert, count) in counts.into_iter().enumerate() {
            result.push(if count == 0 {
                None
            } else {
                let count = count as usize;
                Some((
                    token_ids
                        .rows(expert, 1)?
                        .reshape(&[capacity, 1])?
                        .rows(0, count)?
                        .reshape(&[count])?,
                    slot_ids
                        .rows(expert, 1)?
                        .reshape(&[capacity, 1])?
                        .rows(0, count)?
                        .reshape(&[count])?,
                ))
            });
        }
        Ok(result)
    }

    pub fn expert_combine(
        &mut self,
        outputs: &[(Tensor, Tensor)],
        weights: &Tensor,
    ) -> Result<Tensor> {
        ensure(
            !outputs.is_empty() && weights.shape.len() == 2 && outputs[0].0.shape.len() == 2,
            "expert combine inputs",
        )?;
        let d = outputs[0].0.shape[1];
        let y = self.full(&[weights.shape[0], d], DType::F32, 0.)?;
        for (x, slots) in outputs {
            ensure(
                x.shape.len() == 2
                    && x.shape[1] == d
                    && x.shape[0] == slots.len()
                    && slots.dtype == DType::I32,
                "expert combine packed dimensions",
            )?;
            self.launch(
                "hierarchy_expert_combine",
                &[
                    ("X", p(x)),
                    ("SLOTS", p(slots)),
                    ("WEIGHTS", p(weights)),
                    ("Y", p(&y)),
                    ("N", n(x.shape[0])?),
                    ("D", n(d)?),
                ],
                json::json!({"K":weights.shape[1],"BLOCK":256}),
                [x.len().div_ceil(256), 1, 1],
            )?;
        }
        self.cast(&y, outputs[0].0.dtype)
    }

    pub fn expert_combine_backward(
        &mut self,
        outputs: &[(Tensor, Tensor)],
        weights: &Tensor,
        dy: &Tensor,
    ) -> Result<(Vec<Tensor>, Tensor)> {
        ensure(
            weights.shape.len() == 2 && dy.shape.len() == 2 && dy.shape[0] == weights.shape[0],
            "expert combine backward dimensions",
        )?;
        let dw = self.full(&weights.shape, DType::F32, 0.)?;
        let mut gradients = Vec::with_capacity(outputs.len());
        for (x, slots) in outputs {
            ensure(
                x.shape.len() == 2
                    && x.shape[1] == dy.shape[1]
                    && x.shape[0] == slots.len()
                    && slots.dtype == DType::I32,
                "expert combine backward packed dimensions",
            )?;
            let dx = self.empty(&x.shape, DType::F32)?;
            self.launch(
                "hierarchy_expert_backward",
                &[
                    ("X", p(x)),
                    ("SLOTS", p(slots)),
                    ("WEIGHTS", p(weights)),
                    ("DY", p(dy)),
                    ("DX", p(&dx)),
                    ("DW", p(&dw)),
                    ("D", n(x.shape[1])?),
                ],
                json::json!({"K":weights.shape[1],"BLOCK":x.shape[1].min(256).next_power_of_two()}),
                [x.shape[0], 1, 1],
            )?;
            gradients.push(dx);
        }
        Ok((gradients, dw))
    }

    pub fn row_mean(&mut self, x: &Tensor) -> Result<Tensor> {
        ensure(x.shape.len() == 2, "row mean requires matrix")?;
        let y = self.empty(&[1, x.shape[1]], x.dtype)?;
        let block = x.shape[1].min(256).next_power_of_two();
        self.launch(
            "hierarchy_row_mean",
            &[
                ("X", p(x)),
                ("Y", p(&y)),
                ("ROWS", n(x.shape[0])?),
                ("D", n(x.shape[1])?),
            ],
            json::json!({"BLOCK":block}),
            [x.shape[1].div_ceil(block), 1, 1],
        )?;
        Ok(y)
    }

    pub fn row_mean_backward(&mut self, dy: &Tensor, shape: &[usize]) -> Result<Tensor> {
        ensure(
            shape.len() == 2 && dy.shape == [1, shape[1]],
            "row mean backward dimensions",
        )?;
        let dx = self.empty(shape, DType::F32)?;
        self.launch(
            "hierarchy_row_mean_backward",
            &[
                ("DY", p(dy)),
                ("DX", p(&dx)),
                ("N", n(dx.len())?),
                ("ROWS", n(shape[0])?),
                ("D", n(shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [dx.len().div_ceil(256), 1, 1],
        )?;
        Ok(dx)
    }

    pub fn transpose2d(&mut self, x: &Tensor) -> Result<Tensor> {
        ensure(x.shape.len() == 2, "transpose requires matrix")?;
        let y = self.empty(&[x.shape[1], x.shape[0]], x.dtype)?;
        self.launch(
            "hierarchy_transpose",
            &[
                ("X", p(x)),
                ("Y", p(&y)),
                ("ROWS", n(x.shape[0])?),
                ("COLS", n(x.shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [x.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }

    pub fn transport_coefficients(
        &mut self,
        h: &Tensor,
        left: &Tensor,
        right: &Tensor,
    ) -> Result<Tensor> {
        ensure(
            h.shape.len() == 2
                && left.shape.len() == 2
                && right.shape == left.shape
                && left.shape[0] == left.shape[1]
                && left.shape[0].checked_mul(left.shape[0]) == Some(h.shape[1]),
            "coefficient transport dimensions",
        )?;
        let rank = left.shape[0];
        let intermediate = self.empty(&h.shape, DType::F32)?;
        let output = self.empty(&h.shape, DType::F32)?;
        let bm = 16;
        let bn = rank.min(32).next_power_of_two().max(16);
        let bk = rank.min(32).next_power_of_two().max(16);
        for (input, transform, destination, is_left) in [
            (h, left, &intermediate, true),
            (&intermediate, right, &output, false),
        ] {
            self.launch(
                "hierarchy_coefficient_transport",
                &[
                    ("H", p(input)),
                    ("TRANSFORM", p(transform)),
                    ("Y", p(destination)),
                    ("R", n(rank)?),
                ],
                json::json!({"LEFT":is_left,"BM":bm,"BN":bn,"BK":bk}),
                [rank.div_ceil(bm), rank.div_ceil(bn), h.shape[0]],
            )?;
        }
        Ok(output)
    }

    pub fn kronecker(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ensure(
            a.shape.len() == 2 && b.shape.len() == 2,
            "Kronecker product requires matrices",
        )?;
        let y = self.empty(
            &[
                product(a.shape[0], b.shape[0])?,
                product(a.shape[1], b.shape[1])?,
            ],
            DType::F32,
        )?;
        self.launch(
            "hierarchy_kronecker",
            &[
                ("A", p(a)),
                ("B", p(b)),
                ("Y", p(&y)),
                ("AR", n(a.shape[0])?),
                ("AC", n(a.shape[1])?),
                ("BR", n(b.shape[0])?),
                ("BC", n(b.shape[1])?),
            ],
            json::json!({"BLOCK":256}),
            [y.len().div_ceil(256), 1, 1],
        )?;
        Ok(y)
    }

    pub fn dot(&mut self, a: &Tensor, b: &Tensor) -> Result<f64> {
        ensure(a.shape == b.shape, "dot product shape mismatch")?;
        let y = self.empty(&[a.len().div_ceil(256)], DType::F32)?;
        self.launch(
            "hierarchy_dot",
            &[("A", p(a)), ("B", p(b)), ("Y", p(&y)), ("N", n(a.len())?)],
            json::json!({"BLOCK":256}),
            [y.len(), 1, 1],
        )?;
        Ok(self.to_f32(&y)?.into_iter().map(f64::from).sum())
    }

    pub fn orthonormalize_columns(&mut self, x: &Tensor) -> Result<Tensor> {
        ensure(
            x.shape.len() == 2 && x.shape[1] <= x.shape[0],
            "orthogonal basis requires rows >= columns",
        )?;
        let q = self.empty(&x.shape, DType::F32)?;
        let block = x.shape[0].min(128).next_power_of_two();
        for column in 0..x.shape[1] {
            self.launch(
                "hierarchy_orthogonal_column",
                &[
                    ("X", p(x)),
                    ("Q", p(&q)),
                    ("ROWS", n(x.shape[0])?),
                    ("COLS", n(x.shape[1])?),
                ],
                json::json!({"COLUMN":column,"BLOCK":block}),
                [1, 1, 1],
            )?;
        }
        Ok(q)
    }
}
