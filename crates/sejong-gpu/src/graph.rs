//! Reverse-mode execution tape with FP32 gradient accumulation.
use crate::{Engine, Tensor};
use std::rc::Rc;
type Recompute = Rc<dyn for<'b> Fn(&mut Graph<'b>, &[Var]) -> Result<Var>>;
use sejong_core::{DType, Result, ensure};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var(pub usize);
#[derive(Clone)]
enum Op {
    Checkpoint(Vec<Var>, Recompute),
    Select(Var, Var, Tensor),
    Gelu(Var),
    Gather(Var, Tensor),
    LayerNorm(Var, Var, Var, f32),
    Bias(Var, Var),
    Scale(Var, f32),
    Relu(Var),
    Mul(Var, Var),
    Reshape(Var),
    Concat(Vec<Var>, bool),
    Slice(Var, usize),
    BiasedAttention(Var, Var, Var, Var, Tensor, usize, usize),
    Leaf,
    Cast(Var),
    Add(Var, Var),
    Matmul(Var, Var),
    Embedding(Var, Tensor),
    Norm(Var, Var, f32),
    SwiGlu(Var, Var),
    Rope(Var, usize, usize, usize, f32),
    Attention(Var, Var, Var, Tensor, usize, usize, bool),
    CrossEntropy(Var, Tensor),
}
struct Node {
    value: Tensor,
    gradient: Option<Tensor>,
    op: Op,
    requires_grad: bool,
}
pub struct Graph<'a> {
    pub engine: &'a mut Engine,
    nodes: Vec<Node>,
    training: bool,
    pub checkpointing: bool,
}
impl<'a> Graph<'a> {
    pub fn new(engine: &'a mut Engine, training: bool) -> Self {
        Self {
            engine,
            nodes: vec![],
            training,
            checkpointing: training,
        }
    }
    fn insert(&mut self, value: Tensor, op: Op, requires_grad: bool) -> Var {
        let i = self.nodes.len();
        self.nodes.push(Node {
            value,
            gradient: None,
            op: if self.training && requires_grad {
                op
            } else {
                Op::Leaf
            },
            requires_grad: self.training && requires_grad,
        });
        Var(i)
    }
    pub fn checkpoint(
        &mut self,
        inputs: &[Var],
        forward: impl for<'b> Fn(&mut Graph<'b>, &[Var]) -> Result<Var> + 'static,
    ) -> Result<Var> {
        if !self.checkpointing {
            return forward(self, inputs);
        }
        let required = self.needs(inputs);
        let tensors: Vec<_> = inputs.iter().map(|v| self.tensor(*v).clone()).collect();
        let mut temporary = Graph::new(self.engine, false);
        let variables: Vec<_> = tensors
            .into_iter()
            .map(|t| temporary.leaf(t, false))
            .collect();
        let result = forward(&mut temporary, &variables)?;
        let value = temporary.tensor(result).clone();
        drop(temporary);
        Ok(self.insert(
            value,
            Op::Checkpoint(inputs.to_vec(), Rc::new(forward)),
            required,
        ))
    }
    pub fn tensor(&self, v: Var) -> &Tensor {
        &self.nodes[v.0].value
    }
    pub fn value(&mut self, v: Var) -> Result<Vec<f32>> {
        self.engine.to_f32(&self.nodes[v.0].value)
    }
    pub fn gradient(&self, v: Var) -> Option<&Tensor> {
        self.nodes[v.0].gradient.as_ref()
    }
    pub fn leaf(&mut self, value: Tensor, trainable: bool) -> Var {
        self.insert(value, Op::Leaf, trainable)
    }
    fn needs(&self, vars: &[Var]) -> bool {
        vars.iter().any(|v| self.nodes[v.0].requires_grad)
    }
    pub fn cast(&mut self, x: Var, dtype: DType) -> Result<Var> {
        if self.tensor(x).dtype == dtype {
            return Ok(x);
        }
        let y = self.engine.cast(&self.nodes[x.0].value, dtype)?;
        Ok(self.insert(y, Op::Cast(x), self.needs(&[x])))
    }
    pub fn add(&mut self, x: Var, z: Var) -> Result<Var> {
        let y = self.engine.add(
            &self.nodes[x.0].value,
            &self.nodes[z.0].value,
            1.,
            self.tensor(x).dtype,
        )?;
        Ok(self.insert(y, Op::Add(x, z), self.needs(&[x, z])))
    }
    pub fn matmul(&mut self, a: Var, b: Var) -> Result<Var> {
        let y = self.engine.matmul(
            &self.nodes[a.0].value,
            &self.nodes[b.0].value,
            false,
            false,
            self.tensor(a).dtype,
        )?;
        Ok(self.insert(y, Op::Matmul(a, b), self.needs(&[a, b])))
    }
    pub fn embedding(&mut self, w: Var, ids: &Tensor, dtype: DType) -> Result<Var> {
        let y = self.engine.embedding(&self.nodes[w.0].value, ids, dtype)?;
        Ok(self.insert(y, Op::Embedding(w, ids.clone()), self.needs(&[w])))
    }
    pub fn norm(&mut self, x: Var, w: Var, eps: f32) -> Result<Var> {
        let y = self
            .engine
            .norm(&self.nodes[x.0].value, &self.nodes[w.0].value, eps)?;
        Ok(self.insert(y, Op::Norm(x, w, eps), self.needs(&[x, w])))
    }
    pub fn swiglu(&mut self, g: Var, u: Var) -> Result<Var> {
        let y = self
            .engine
            .swiglu(&self.nodes[g.0].value, &self.nodes[u.0].value)?;
        Ok(self.insert(y, Op::SwiGlu(g, u), self.needs(&[g, u])))
    }
    pub fn rope(
        &mut self,
        x: Var,
        t: usize,
        offset: usize,
        heads: usize,
        theta: f32,
    ) -> Result<Var> {
        let y = self
            .engine
            .rope(&self.nodes[x.0].value, t, offset, heads, theta, false)?;
        Ok(self.insert(y, Op::Rope(x, t, offset, heads, theta), self.needs(&[x])))
    }
    pub fn attention(&mut self, q: Var, k: Var, v: Var, t: usize, heads: usize) -> Result<Var> {
        self.attention_mode(q, k, v, t, heads, true)
    }
    pub fn attention_mode(
        &mut self,
        q: Var,
        k: Var,
        v: Var,
        t: usize,
        heads: usize,
        causal: bool,
    ) -> Result<Var> {
        let (y, lse) = self.engine.attention_mode(
            &self.nodes[q.0].value,
            &self.nodes[k.0].value,
            &self.nodes[v.0].value,
            t,
            t,
            0,
            heads,
            causal,
        )?;
        Ok(self.insert(
            y,
            Op::Attention(q, k, v, lse, t, heads, causal),
            self.needs(&[q, k, v]),
        ))
    }
    pub fn gelu(&mut self, x: Var) -> Result<Var> {
        let y = self.engine.gelu(&self.nodes[x.0].value)?;
        Ok(self.insert(y, Op::Gelu(x), self.needs(&[x])))
    }
    pub fn gather(&mut self, x: Var, ids: &Tensor) -> Result<Var> {
        let y = self.engine.gather(&self.nodes[x.0].value, ids)?;
        Ok(self.insert(y, Op::Gather(x, ids.clone()), self.needs(&[x])))
    }
    pub fn layer_norm(&mut self, x: Var, w: Var, b: Var, eps: f32) -> Result<Var> {
        let y = self.engine.layer_norm(
            &self.nodes[x.0].value,
            &self.nodes[w.0].value,
            &self.nodes[b.0].value,
            eps,
        )?;
        Ok(self.insert(y, Op::LayerNorm(x, w, b, eps), self.needs(&[x, w, b])))
    }
    pub fn bias(&mut self, x: Var, b: Var) -> Result<Var> {
        let y = self
            .engine
            .bias_add(&self.nodes[x.0].value, &self.nodes[b.0].value)?;
        Ok(self.insert(y, Op::Bias(x, b), self.needs(&[x, b])))
    }
    pub fn affine(&mut self, x: Var, w: Var, b: Var) -> Result<Var> {
        let x = self.matmul(x, w)?;
        self.bias(x, b)
    }
    pub fn scale(&mut self, x: Var, alpha: f32) -> Result<Var> {
        let y = self.engine.scale(&self.nodes[x.0].value, alpha)?;
        Ok(self.insert(y, Op::Scale(x, alpha), self.needs(&[x])))
    }
    pub fn relu(&mut self, x: Var) -> Result<Var> {
        let y = self.engine.relu(&self.nodes[x.0].value)?;
        Ok(self.insert(y, Op::Relu(x), self.needs(&[x])))
    }
    pub fn mul(&mut self, x: Var, z: Var) -> Result<Var> {
        let y = self
            .engine
            .mul(&self.nodes[x.0].value, &self.nodes[z.0].value)?;
        Ok(self.insert(y, Op::Mul(x, z), self.needs(&[x, z])))
    }
    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Result<Var> {
        let y = self.tensor(x).reshape(shape)?;
        Ok(self.insert(y, Op::Reshape(x), self.needs(&[x])))
    }
    pub fn slice_columns(&mut self, x: Var, start: usize, width: usize) -> Result<Var> {
        let y = self
            .engine
            .slice_columns(&self.nodes[x.0].value, start, width)?;
        Ok(self.insert(y, Op::Slice(x, start), self.needs(&[x])))
    }
    fn concat(&mut self, xs: &[Var], rows: bool) -> Result<Var> {
        let tensors = xs
            .iter()
            .map(|v| self.tensor(*v).clone())
            .collect::<Vec<_>>();
        let y = if rows {
            self.engine.concat_rows(&tensors)?
        } else {
            self.engine.concat_columns(&tensors)?
        };
        Ok(self.insert(y, Op::Concat(xs.to_vec(), rows), self.needs(xs)))
    }
    pub fn concat_columns(&mut self, xs: &[Var]) -> Result<Var> {
        self.concat(xs, false)
    }
    pub fn concat_rows(&mut self, xs: &[Var]) -> Result<Var> {
        self.concat(xs, true)
    }
    pub fn biased_attention(
        &mut self,
        q: Var,
        k: Var,
        v: Var,
        bias: Var,
        t: usize,
        heads: usize,
    ) -> Result<Var> {
        let (y, lse) = self.engine.biased_attention(
            &self.nodes[q.0].value,
            &self.nodes[k.0].value,
            &self.nodes[v.0].value,
            &self.nodes[bias.0].value,
            t,
            heads,
        )?;
        Ok(self.insert(
            y,
            Op::BiasedAttention(q, k, v, bias, lse, t, heads),
            self.needs(&[q, k, v, bias]),
        ))
    }
    pub fn cross_entropy(&mut self, x: Var, targets: &Tensor) -> Result<Var> {
        let y = self.engine.cross_entropy(&self.nodes[x.0].value, targets)?;
        Ok(self.insert(y, Op::CrossEntropy(x, targets.clone()), self.needs(&[x])))
    }
    fn accumulate(&mut self, v: Var, g: Tensor) -> Result<()> {
        if !self.nodes[v.0].requires_grad {
            return Ok(());
        }
        ensure(
            g.shape == self.nodes[v.0].value.shape,
            "gradient shape mismatch",
        )?;
        if let Some(current) = &self.nodes[v.0].gradient {
            self.nodes[v.0].gradient = Some(self.engine.add(current, &g, 1., DType::F32)?);
        } else {
            self.nodes[v.0].gradient = Some(g);
        }
        Ok(())
    }
    /// Each supplied root contributes its mean multiplied by the supplied scale.
    pub fn select_rows(&mut self, a: Var, b: Var, mask: &Tensor) -> Result<Var> {
        let y = self
            .engine
            .select_rows(&self.nodes[a.0].value, &self.nodes[b.0].value, mask)?;
        Ok(self.insert(y, Op::Select(a, b, mask.clone()), self.needs(&[a, b])))
    }
    pub fn backward_seeded(&mut self, roots: &[(Var, Tensor)]) -> Result<()> {
        for (v, seed) in roots {
            self.accumulate(*v, seed.clone())?;
        }
        self.propagate()
    }
    pub fn backward(&mut self, roots: &[(Var, f32)]) -> Result<()> {
        for (root, scale) in roots {
            let t = self.tensor(*root);
            let (shape, count) = (t.shape.clone(), t.len());
            let seed = self
                .engine
                .full(&shape, DType::F32, *scale / count as f32)?;
            self.accumulate(*root, seed)?;
        }
        self.propagate()
    }
    fn propagate(&mut self) -> Result<()> {
        for i in (0..self.nodes.len()).rev() {
            let Some(dy) = self.nodes[i].gradient.clone() else {
                continue;
            };
            let op = self.nodes[i].op.clone();
            match op {
                Op::Leaf => {}
                Op::LayerNorm(x, w, b, eps) => {
                    let (dx, dw, db) = self.engine.layer_norm_backward(
                        &self.nodes[x.0].value,
                        &self.nodes[w.0].value,
                        &self.nodes[b.0].value,
                        &dy,
                        eps,
                    )?;
                    self.accumulate(x, dx)?;
                    self.accumulate(w, dw)?;
                    self.accumulate(b, db)?;
                }
                Op::Bias(x, b) => {
                    let db = self.engine.bias_reduce(&dy, &self.nodes[b.0].value.shape)?;
                    self.accumulate(x, dy)?;
                    self.accumulate(b, db)?;
                }
                Op::Scale(x, a) => {
                    let dx = self.engine.scale(&dy, a)?;
                    self.accumulate(x, dx)?;
                }
                Op::Relu(x) => {
                    let dx = self.engine.relu_backward(&self.nodes[x.0].value, &dy)?;
                    self.accumulate(x, dx)?;
                }
                Op::Mul(x, z) => {
                    let xval = self.engine.cast(&self.nodes[x.0].value, DType::F32)?;
                    let zval = self.engine.cast(&self.nodes[z.0].value, DType::F32)?;
                    let dx = self.engine.mul(&dy, &zval)?;
                    let dz = self.engine.mul(&dy, &xval)?;
                    self.accumulate(x, dx)?;
                    self.accumulate(z, dz)?;
                }
                Op::Reshape(x) => {
                    let dx = dy.reshape(&self.nodes[x.0].value.shape)?;
                    self.accumulate(x, dx)?;
                }
                Op::Slice(x, start) => {
                    let dx = self
                        .engine
                        .full(&self.nodes[x.0].value.shape, DType::F32, 0.)?;
                    self.engine.insert_columns(&dy, &dx, start)?;
                    self.accumulate(x, dx)?;
                }
                Op::Concat(xs, rows) => {
                    let mut offset = 0;
                    for x in xs {
                        let shape = &self.nodes[x.0].value.shape;
                        let dx = if rows {
                            dy.rows(offset, shape[0])?
                        } else {
                            self.engine.slice_columns(&dy, offset, shape[1])?
                        };
                        offset += shape[if rows { 0 } else { 1 }];
                        self.accumulate(x, dx)?;
                    }
                }
                Op::BiasedAttention(q, k, v, bias, lse, t, h) => {
                    let (dq, dk, dv, db) = self.engine.biased_attention_backward(
                        &self.nodes[q.0].value,
                        &self.nodes[k.0].value,
                        &self.nodes[v.0].value,
                        &self.nodes[bias.0].value,
                        &self.nodes[i].value,
                        &lse,
                        &dy,
                        t,
                        h,
                    )?;
                    self.accumulate(q, dq)?;
                    self.accumulate(k, dk)?;
                    self.accumulate(v, dv)?;
                    self.accumulate(bias, db)?;
                }

                Op::Gelu(x) => {
                    let dx = self.engine.gelu_backward(&self.nodes[x.0].value, &dy)?;
                    self.accumulate(x, dx)?;
                }
                Op::Gather(x, ids) => {
                    let dx =
                        self.engine
                            .gather_backward(&dy, &ids, &self.nodes[x.0].value.shape)?;
                    self.accumulate(x, dx)?;
                }

                Op::Checkpoint(inputs, forward) => {
                    let tensors: Vec<_> = inputs
                        .iter()
                        .map(|v| (self.tensor(*v).clone(), self.nodes[v.0].requires_grad))
                        .collect();
                    let mut nested = Graph::new(self.engine, true);
                    nested.checkpointing = false;
                    let variables: Vec<_> = tensors
                        .into_iter()
                        .map(|(t, required)| nested.leaf(t, required))
                        .collect();
                    let output = forward(&mut nested, &variables)?;
                    nested.backward_seeded(&[(output, dy)])?;
                    let gradients: Vec<_> = variables
                        .iter()
                        .map(|v| nested.gradient(*v).cloned())
                        .collect();
                    drop(nested);
                    for (input, gradient) in inputs.into_iter().zip(gradients) {
                        if let Some(gradient) = gradient {
                            self.accumulate(input, gradient)?;
                        }
                    }
                }
                Op::Select(a, b, mask) => {
                    let zero = self.engine.full(&dy.shape, DType::F32, 0.)?;
                    let da = self.engine.select_rows(&dy, &zero, &mask)?;
                    let db = self.engine.select_rows(&zero, &dy, &mask)?;
                    self.accumulate(a, da)?;
                    self.accumulate(b, db)?;
                }
                Op::Cast(x) => self.accumulate(x, dy)?,
                Op::Add(x, z) => {
                    self.accumulate(x, dy.clone())?;
                    self.accumulate(z, dy)?;
                }
                Op::Matmul(a, b) => {
                    if self.needs(&[a]) {
                        let da = self.engine.matmul(
                            &dy,
                            &self.nodes[b.0].value,
                            false,
                            true,
                            DType::F32,
                        )?;
                        self.accumulate(a, da)?;
                    }
                    if self.needs(&[b]) {
                        let db = self.engine.matmul(
                            &self.nodes[a.0].value,
                            &dy,
                            true,
                            false,
                            DType::F32,
                        )?;
                        self.accumulate(b, db)?;
                    }
                }
                Op::Embedding(w, ids) => {
                    let dw =
                        self.engine
                            .embedding_backward(&dy, &ids, &self.nodes[w.0].value.shape)?;
                    self.accumulate(w, dw)?;
                }
                Op::Norm(x, w, eps) => {
                    let (dx, dw) = self.engine.norm_backward(
                        &self.nodes[x.0].value,
                        &self.nodes[w.0].value,
                        &dy,
                        eps,
                    )?;
                    self.accumulate(x, dx)?;
                    self.accumulate(w, dw)?;
                }
                Op::SwiGlu(g, u) => {
                    let (dg, du) = self.engine.swiglu_backward(
                        &self.nodes[g.0].value,
                        &self.nodes[u.0].value,
                        &dy,
                    )?;
                    self.accumulate(g, dg)?;
                    self.accumulate(u, du)?;
                }
                Op::Rope(x, t, offset, h, theta) => {
                    let dx = self.engine.rope(&dy, t, offset, h, theta, true)?;
                    self.accumulate(x, dx)?;
                }
                Op::Attention(q, k, v, lse, t, h, causal) => {
                    let (dq, dk, dv) = self.engine.attention_backward_mode(
                        &self.nodes[q.0].value,
                        &self.nodes[k.0].value,
                        &self.nodes[v.0].value,
                        &self.nodes[i].value,
                        &lse,
                        &dy,
                        t,
                        h,
                        causal,
                    )?;
                    self.accumulate(q, dq)?;
                    self.accumulate(k, dk)?;
                    self.accumulate(v, dv)?;
                }
                Op::CrossEntropy(x, targets) => {
                    let dx = self.engine.cross_entropy_backward(
                        &self.nodes[x.0].value,
                        &targets,
                        &dy,
                    )?;
                    self.accumulate(x, dx)?;
                }
            }
            if !matches!(self.nodes[i].op, Op::Leaf) {
                self.nodes[i].gradient = None;
            }
        }
        Ok(())
    }
}
