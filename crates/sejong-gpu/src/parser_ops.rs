//! Relative and shared-identity attention for trainable query encoders.
use crate::{Engine, Tensor, n, p};
use sejong_core::{DType, Result, ensure, json};
impl Engine {
    pub fn biased_attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        bias: &Tensor,
        t: usize,
        heads: usize,
    ) -> Result<(Tensor, Tensor)> {
        ensure(
            q.shape == k.shape
                && q.shape == v.shape
                && q.shape[0].is_multiple_of(t)
                && q.shape[1].is_multiple_of(heads)
                && bias.shape == [q.shape[0] * t, heads],
            "relative attention shape mismatch",
        )?;
        let d = q.shape[1] / heads;
        let y = self.empty(&q.shape, DType::F32)?;
        let lse = self.empty(&[q.shape[0] * heads], DType::F32)?;
        self.launch(
            "parser_attention",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("Bias", p(bias)),
                ("O", p(&y)),
                ("LSE", p(&lse)),
                ("T", n(t)?),
            ],
            json::json!({"H":heads,"D":d,"BD":d.next_power_of_two(),"BK":32}),
            [q.shape[0] * heads, 1, 1],
        )?;
        Ok((y, lse))
    }
    pub fn biased_attention_backward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        bias: &Tensor,
        o: &Tensor,
        lse: &Tensor,
        dy: &Tensor,
        t: usize,
        heads: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let d = q.shape[1] / heads;
        let dq = self.empty(&q.shape, DType::F32)?;
        let dk = self.empty(&q.shape, DType::F32)?;
        let dv = self.empty(&q.shape, DType::F32)?;
        let db = self.empty(&bias.shape, DType::F32)?;
        let args = [
            ("Q", p(q)),
            ("K", p(k)),
            ("V", p(v)),
            ("Bias", p(bias)),
            ("O", p(o)),
            ("LSE", p(lse)),
            ("DY", p(dy)),
            ("DQ", p(&dq)),
            ("DB", p(&db)),
            ("T", n(t)?),
        ];
        let constants = json::json!({"H":heads,"D":d,"BD":d.next_power_of_two(),"BK":32});
        self.launch(
            "parser_attention_dq",
            &args,
            constants.clone(),
            [q.shape[0] * heads, 1, 1],
        )?;
        self.launch(
            "parser_attention_dkv",
            &[
                ("Q", p(q)),
                ("K", p(k)),
                ("V", p(v)),
                ("Bias", p(bias)),
                ("O", p(o)),
                ("LSE", p(lse)),
                ("DY", p(dy)),
                ("DK", p(&dk)),
                ("DV", p(&dv)),
                ("T", n(t)?),
            ],
            constants,
            [q.shape[0] * heads, 1, 1],
        )?;
        Ok((dq, dk, dv, db))
    }
}
