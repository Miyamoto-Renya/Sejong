//! Reverse-mode execution tape with FP32 gradient accumulation.
use crate::{Engine, Tensor};
use std::rc::Rc;
type Recompute = Rc<dyn for<'b> Fn(&mut Graph<'b>, &[Var]) -> Result<Var>>;
use crate::parameter_vjp::{
    FactoredRows, FeatureSelection, ParameterGradient, PreparedVjp, TokenBasis, VjpStatistics,
    budget, selects_compression,
};
use sejong_core::{DType, Result, ensure};
use sejong_core::{ParameterVjpConfig, Rng};
use std::collections::{HashMap, HashSet};
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
    LowRankMatmul(Var, Var, Tensor, Tensor, Tensor, Tensor),
    GroupedAttention(Var, Var, Var, Tensor, usize, usize, usize, Option<usize>),
    PatchPool(Var, usize, usize),
    PatchBroadcast(Var, Var, usize, usize),
    RouteTopK(Var, Tensor),
    ExpertCombine(Vec<(Var, Tensor)>, Var),
    RowMean(Var),
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
    projected_gradient: Option<Tensor>,
    op: Op,
    requires_grad: bool,
    parameter: bool,
    factored: Vec<FactoredRows>,
}
/// 실제 역전파의 선형 연산 입력·출력·cotangent를 보존하는 진단 표본.
#[derive(Clone)]
pub struct LinearAdjoint {
    pub input: Var,
    pub weight: Var,
    pub output: Var,
    pub x: Tensor,
    pub w: Tensor,
    pub y: Tensor,
    pub dy: Tensor,
}
pub struct Graph<'a> {
    pub engine: &'a mut Engine,
    nodes: Vec<Node>,
    training: bool,
    pub checkpointing: bool,
    linear_adjoints: Option<Vec<LinearAdjoint>>,
    parameter_policy: Option<(ParameterVjpConfig, Rng)>,
    parameter_groups: HashMap<usize, Option<(PreparedVjp, HashSet<usize>)>>,
    parameter_members: HashMap<usize, Vec<usize>>,
    token_bases: HashMap<(usize, usize), TokenBasis>,
    feature_selections: HashMap<usize, FeatureSelection>,
    parameter_statistics: VjpStatistics,
}
impl<'a> Graph<'a> {
    pub fn new(engine: &'a mut Engine, training: bool) -> Self {
        Self {
            engine,
            nodes: vec![],
            training,
            checkpointing: training,
            linear_adjoints: None,
            parameter_policy: None,
            parameter_groups: HashMap::new(),
            parameter_members: HashMap::new(),
            token_bases: HashMap::new(),
            feature_selections: HashMap::new(),
            parameter_statistics: VjpStatistics::default(),
        }
    }
    pub fn configure_parameter_vjp(&mut self, config: ParameterVjpConfig, rng: Rng) -> Result<()> {
        ensure(
            self.training && self.nodes.is_empty(),
            "parameter VJP requires a fresh training graph",
        )?;
        self.parameter_policy = Some((config, rng));
        Ok(())
    }
    pub fn parameter_vjp_rng(&self) -> Option<Rng> {
        self.parameter_policy.as_ref().map(|(_, rng)| rng.clone())
    }
    pub fn parameter_vjp_statistics(&self) -> &VjpStatistics {
        &self.parameter_statistics
    }
    pub fn parameter_gradient(&self, v: Var) -> ParameterGradient {
        ParameterGradient {
            dense: self.nodes[v.0].gradient.clone(),
            terms: self.nodes[v.0].factored.clone(),
        }
    }
    pub fn parameter(&mut self, value: Tensor, trainable: bool) -> Var {
        let v = self.leaf(value, trainable);
        self.nodes[v.0].parameter = true;
        v
    }
    /// 전체 tape를 구성한 진단 실행에서 선형 역전파 operand를 수집한다.
    /// Forward 전에 checkpointing을 false로 설정하면 원래 Var 인덱스를 보존한다.
    pub fn enable_linear_adjoint_trace(&mut self) -> Result<()> {
        ensure(
            self.training && !self.checkpointing && self.nodes.is_empty(),
            "linear adjoint trace requires a fresh training graph with checkpointing=false",
        )?;
        self.linear_adjoints = Some(Vec::new());
        Ok(())
    }
    pub fn linear_adjoint_trace(&self) -> &[LinearAdjoint] {
        self.linear_adjoints.as_deref().unwrap_or(&[])
    }
    fn insert(&mut self, value: Tensor, op: Op, requires_grad: bool) -> Var {
        let i = self.nodes.len();
        self.nodes.push(Node {
            value,
            gradient: None,
            projected_gradient: None,
            op: if self.training && requires_grad {
                op
            } else {
                Op::Leaf
            },
            requires_grad: self.training && requires_grad,
            parameter: false,
            factored: Vec::new(),
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
    /// Dense component of the leaf adjoint.
    pub fn gradient(&self, v: Var) -> Option<&Tensor> {
        self.nodes[v.0].gradient.as_ref()
    }
    pub fn projected_gradient(&self, v: Var) -> Option<&Tensor> {
        self.nodes[v.0].projected_gradient.as_ref()
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
        let v = self.insert(y, Op::Cast(x), self.needs(&[x]));
        self.nodes[v.0].parameter = self.nodes[x.0].parameter;
        Ok(v)
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
    pub fn project_qkv(&mut self, x: Var, wq: Var, wk: Var, wv: Var) -> Result<(Var, Var, Var)> {
        let (q, k, v) = self.engine.project_qkv(
            &self.nodes[x.0].value,
            &self.nodes[wq.0].value,
            &self.nodes[wk.0].value,
            &self.nodes[wv.0].value,
        )?;
        let q = self.insert(q, Op::Matmul(x, wq), self.needs(&[x, wq]));
        let k = self.insert(k, Op::Matmul(x, wk), self.needs(&[x, wk]));
        let v = self.insert(v, Op::Matmul(x, wv), self.needs(&[x, wv]));
        Ok((q, k, v))
    }
    pub fn gate_up(&mut self, x: Var, wg: Var, wu: Var) -> Result<Var> {
        let (g, u, h) = self.engine.gate_up(
            &self.nodes[x.0].value,
            &self.nodes[wg.0].value,
            &self.nodes[wu.0].value,
        )?;
        let g = self.insert(g, Op::Matmul(x, wg), self.needs(&[x, wg]));
        let u = self.insert(u, Op::Matmul(x, wu), self.needs(&[x, wu]));
        Ok(self.insert(h, Op::SwiGlu(g, u), self.needs(&[g, u])))
    }
    pub fn low_rank_matmul(
        &mut self,
        x: Var,
        base: Var,
        u: &Tensor,
        a: &Tensor,
        v: &Tensor,
    ) -> Result<Var> {
        let xt = self.tensor(x).clone();
        let wt = self.tensor(base).clone();
        ensure(
            u.shape.len() == 2
                && a.shape.len() == 2
                && v.shape.len() == 2
                && wt.shape == [u.shape[0], v.shape[0]]
                && a.shape == [u.shape[1], v.shape[1]],
            "low-rank matrix dimensions",
        )?;
        let dense = self.engine.matmul(&xt, &wt, false, false, xt.dtype)?;
        let xu = self.engine.matmul(&xt, u, false, false, DType::F32)?;
        let xa = self.engine.matmul(&xu, a, false, false, DType::F32)?;
        let correction = self.engine.matmul(&xa, v, false, true, xt.dtype)?;
        let y = self.engine.add(&dense, &correction, 1., xt.dtype)?;
        Ok(self.insert(
            y,
            Op::LowRankMatmul(x, base, u.clone(), a.clone(), v.clone(), xu),
            self.needs(&[x, base]),
        ))
    }
    pub fn row_mean(&mut self, x: Var) -> Result<Var> {
        let y = self.engine.row_mean(&self.nodes[x.0].value)?;
        Ok(self.insert(y, Op::RowMean(x), self.needs(&[x])))
    }
    pub fn patch_pool(&mut self, x: Var, sequence: usize, patch: usize) -> Result<Var> {
        let y = self
            .engine
            .patch_pool(&self.nodes[x.0].value, sequence, patch)?;
        Ok(self.insert(y, Op::PatchPool(x, sequence, patch), self.needs(&[x])))
    }
    pub fn patch_broadcast(
        &mut self,
        x: Var,
        bos: Var,
        sequence: usize,
        patch: usize,
    ) -> Result<Var> {
        let y = self.engine.patch_broadcast(
            &self.nodes[x.0].value,
            &self.nodes[bos.0].value,
            sequence,
            patch,
        )?;
        Ok(self.insert(
            y,
            Op::PatchBroadcast(x, bos, sequence, patch),
            self.needs(&[x, bos]),
        ))
    }
    pub fn route_topk(&mut self, x: Var, topk: usize) -> Result<(Var, Tensor)> {
        let (weights, ids) = self.engine.route_topk(&self.nodes[x.0].value, topk)?;
        let y = self.insert(weights, Op::RouteTopK(x, ids.clone()), self.needs(&[x]));
        Ok((y, ids))
    }
    pub fn expert_combine(&mut self, outputs: &[(Var, Tensor)], weights: Var) -> Result<Var> {
        let tensors = outputs
            .iter()
            .map(|(v, ids)| (self.tensor(*v).clone(), ids.clone()))
            .collect::<Vec<_>>();
        let y = self
            .engine
            .expert_combine(&tensors, &self.nodes[weights.0].value)?;
        let mut inputs = outputs.iter().map(|(v, _)| *v).collect::<Vec<_>>();
        inputs.push(weights);
        Ok(self.insert(
            y,
            Op::ExpertCombine(outputs.to_vec(), weights),
            self.needs(&inputs),
        ))
    }
    pub fn attention_grouped(
        &mut self,
        q: Var,
        k: Var,
        v: Var,
        t: usize,
        query_heads: usize,
        kv_heads: usize,
        window: Option<usize>,
    ) -> Result<Var> {
        let (y, lse) = self.engine.attention_grouped(
            &self.nodes[q.0].value,
            &self.nodes[k.0].value,
            &self.nodes[v.0].value,
            t,
            t,
            0,
            query_heads,
            kv_heads,
            window,
        )?;
        Ok(self.insert(
            y,
            Op::GroupedAttention(q, k, v, lse, t, query_heads, kv_heads, window),
            self.needs(&[q, k, v]),
        ))
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
        if let Some(current) = &self.nodes[v.0].gradient {
            self.nodes[v.0].gradient = Some(self.engine.add(current, &g, 1., DType::F32)?);
        } else {
            self.nodes[v.0].gradient = Some(g);
        }
        Ok(())
    }
    fn accumulate_projected(&mut self, v: Var, g: Tensor) -> Result<()> {
        if !self.nodes[v.0].requires_grad {
            return Ok(());
        }
        self.nodes[v.0].projected_gradient =
            Some(if let Some(current) = &self.nodes[v.0].projected_gradient {
                self.engine.add(current, &g, 1., DType::F32)?
            } else {
                g
            });
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
        if self.parameter_policy.is_some() {
            self.parameter_members.clear();
            for (index, node) in self.nodes.iter().enumerate() {
                if let Op::Matmul(input, weight) = node.op {
                    if self.nodes[weight.0].parameter
                        && self.nodes[weight.0].requires_grad
                        && self.nodes[input.0].value.shape[0] >= self.nodes[input.0].value.shape[1]
                    {
                        self.parameter_members
                            .entry(input.0)
                            .or_default()
                            .push(index);
                    }
                }
            }
        }
        for i in (0..self.nodes.len()).rev() {
            if let Op::Cast(x) = self.nodes[i].op {
                let terms = std::mem::take(&mut self.nodes[i].factored);
                self.nodes[x.0].factored.extend(terms);
            }
            let Some(dy) = self.nodes[i].gradient.clone() else {
                continue;
            };
            let op = self.nodes[i].op.clone();
            match op {
                Op::Leaf => {}
                Op::RowMean(x) => {
                    let dx = self
                        .engine
                        .row_mean_backward(&dy, &self.nodes[x.0].value.shape)?;
                    self.accumulate(x, dx)?;
                }
                Op::PatchPool(x, t, p) => {
                    let dx =
                        self.engine
                            .patch_pool_backward(&dy, &self.nodes[x.0].value.shape, t, p)?;
                    self.accumulate(x, dx)?;
                }
                Op::PatchBroadcast(x, bos, t, p) => {
                    let (dx, db) = self.engine.patch_broadcast_backward(
                        &dy,
                        &self.nodes[x.0].value.shape,
                        t,
                        p,
                    )?;
                    self.accumulate(x, dx)?;
                    self.accumulate(bos, db)?;
                }
                Op::RouteTopK(x, ids) => {
                    let dx = self.engine.route_topk_backward(
                        &self.nodes[x.0].value,
                        &ids,
                        &self.nodes[i].value,
                        &dy,
                        self.tensor(x).shape[1],
                    )?;
                    self.accumulate(x, dx)?;
                }
                Op::ExpertCombine(outputs, weights) => {
                    let tensors = outputs
                        .iter()
                        .map(|(v, ids)| (self.tensor(*v).clone(), ids.clone()))
                        .collect::<Vec<_>>();
                    let (gradients, dw) = self.engine.expert_combine_backward(
                        &tensors,
                        &self.nodes[weights.0].value,
                        &dy,
                    )?;
                    for ((v, _), gradient) in outputs.into_iter().zip(gradients) {
                        self.accumulate(v, gradient)?;
                    }
                    self.accumulate(weights, dw)?;
                }
                Op::GroupedAttention(q, k, v, lse, t, h, kh, w) => {
                    let (dq, dk, dv) = self.engine.attention_grouped_backward(
                        &self.nodes[q.0].value,
                        &self.nodes[k.0].value,
                        &self.nodes[v.0].value,
                        &self.nodes[i].value,
                        &lse,
                        &dy,
                        t,
                        h,
                        kh,
                        w,
                    )?;
                    self.accumulate(q, dq)?;
                    self.accumulate(k, dk)?;
                    self.accumulate(v, dv)?;
                }
                Op::LowRankMatmul(x, base, u, a, v, xu) => {
                    let dv = self.engine.matmul(&dy, &v, false, false, DType::F32)?;
                    if self.needs(&[base]) {
                        let target = self.engine.matmul(&xu, &dv, true, false, DType::F32)?;
                        self.accumulate_projected(base, target)?;
                        let gradient = self.engine.matmul(
                            &self.nodes[x.0].value,
                            &dy,
                            true,
                            false,
                            DType::F32,
                        )?;
                        self.accumulate(base, gradient)?;
                    }
                    if self.needs(&[x]) {
                        let dense = self.engine.matmul(
                            &dy,
                            &self.nodes[base.0].value,
                            false,
                            true,
                            DType::F32,
                        )?;
                        let da = self.engine.matmul(&dv, &a, false, true, DType::F32)?;
                        let correction = self.engine.matmul(&da, &u, false, true, DType::F32)?;
                        let dx = self.engine.add(&dense, &correction, 1., DType::F32)?;
                        self.accumulate(x, dx)?;
                    }
                }
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
                    nested.parameter_policy = self.parameter_policy.clone();
                    nested.token_bases = self.token_bases.clone();
                    nested.feature_selections = self.feature_selections.clone();
                    let variables: Vec<_> = tensors
                        .into_iter()
                        .map(|(t, required)| nested.leaf(t, required))
                        .collect();
                    for (&source, &destination) in inputs.iter().zip(&variables) {
                        nested.nodes[destination.0].parameter = self.nodes[source.0].parameter;
                    }
                    let output = forward(&mut nested, &variables)?;
                    nested.backward_seeded(&[(output, dy)])?;
                    let gradients: Vec<_> = variables
                        .iter()
                        .map(|v| nested.parameter_gradient(*v))
                        .collect();
                    let projected: Vec<_> = variables
                        .iter()
                        .map(|v| nested.projected_gradient(*v).cloned())
                        .collect();
                    self.parameter_policy = nested.parameter_policy.take();
                    self.token_bases = std::mem::take(&mut nested.token_bases);
                    self.feature_selections = std::mem::take(&mut nested.feature_selections);
                    self.parameter_statistics.add(&nested.parameter_statistics);
                    drop(nested);
                    for ((input, gradient), target) in
                        inputs.into_iter().zip(gradients).zip(projected)
                    {
                        if let Some(gradient) = gradient.dense {
                            self.accumulate(input, gradient)?;
                        }
                        self.nodes[input.0].factored.extend(gradient.terms);
                        if let Some(target) = target {
                            self.accumulate_projected(input, target)?;
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
                Op::Cast(x) => {
                    self.accumulate(x, dy)?;
                    if let Some(target) = self.nodes[i].projected_gradient.clone() {
                        self.accumulate_projected(x, target)?;
                    }
                }
                Op::Add(x, z) => {
                    self.accumulate(x, dy.clone())?;
                    self.accumulate(z, dy)?;
                }
                Op::Matmul(a, b) => {
                    if let Some(trace) = &mut self.linear_adjoints {
                        trace.push(LinearAdjoint {
                            input: a,
                            weight: b,
                            output: Var(i),
                            x: self.nodes[a.0].value.clone(),
                            w: self.nodes[b.0].value.clone(),
                            y: self.nodes[i].value.clone(),
                            dy: dy.clone(),
                        });
                    }
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
                        if let Some(term) = self.compressed_parameter_vjp(i, a, b, &dy)? {
                            self.nodes[b.0].factored.push(term);
                        } else {
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

    fn compressed_parameter_vjp(
        &mut self,
        operation: usize,
        a: Var,
        b: Var,
        dy: &Tensor,
    ) -> Result<Option<FactoredRows>> {
        if self.parameter_policy.is_none() || !self.nodes[b.0].parameter {
            return Ok(None);
        }
        if self.nodes[a.0].value.shape[0] < self.nodes[a.0].value.shape[1] {
            self.parameter_statistics.dense_matrices += 1;
            return Ok(None);
        }
        if !self.parameter_groups.contains_key(&a.0) {
            let members = &self.parameter_members[&a.0];
            let donor = *members
                .iter()
                .max_by_key(|&&i| self.nodes[i].value.shape[1])
                .unwrap();
            let x = &self.nodes[a.0].value;
            let n = x.shape[0];
            let width = x.shape[1];
            let donor_width = self.nodes[donor].value.shape[1];
            let config = &self.parameter_policy.as_ref().unwrap().0;
            let mut active: HashSet<_> = members.iter().copied().collect();
            loop {
                let count = active.len();
                if count == 0 {
                    break;
                }
                active.retain(|&i| {
                    selects_compression(
                        config,
                        n,
                        width,
                        self.nodes[i].value.shape[1],
                        donor_width,
                        count,
                    )
                });
                if active.len() == count {
                    break;
                }
            }
            let group = if active.is_empty() {
                None
            } else {
                let (config, rng) = self.parameter_policy.as_mut().unwrap();
                let rank = budget(config, n, width, donor_width).0;
                let basis = self.token_bases.get(&(n, rank));
                let prepared = self.engine.prepare_parameter_vjp_reusing(
                    &self.nodes[a.0].value,
                    &self.nodes[donor].value,
                    config,
                    rng,
                    basis,
                    self.feature_selections.get(&width),
                )?;
                if basis.is_none() {
                    self.parameter_statistics.basis_groups += 1;
                    self.token_bases.insert(
                        (n, rank),
                        TokenBasis {
                            q: prepared.q.clone(),
                            columns: prepared.selected_columns.clone(),
                            donor_width: prepared.basis_donor_width,
                        },
                    );
                }
                self.feature_selections
                    .entry(width)
                    .or_insert_with(|| FeatureSelection {
                        rows: prepared.selected_rows.clone(),
                        map: prepared.row_map.clone(),
                    });
                self.parameter_statistics.projection_groups += 1;
                Some((prepared, active))
            };
            self.parameter_groups.insert(a.0, group);
        }
        if let Some((prepared, active)) = self.parameter_groups.get(&a.0).unwrap() {
            if active.contains(&operation) {
                let term = self.engine.parameter_vjp(prepared, dy)?;
                self.parameter_statistics.compressed_matrices += 1;
                return Ok(Some(term));
            }
        }
        self.parameter_statistics.dense_matrices += 1;
        Ok(None)
    }
}
