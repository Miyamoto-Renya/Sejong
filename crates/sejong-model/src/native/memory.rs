//! Independently trained query bodies and bounded reads of their K/V weights.
use super::{atomic_directory, signature};
use crate::{Parameter, load_tensors, save_tensors};
use sejong_core::{
    DType, Result, Rng, ensure, json,
    native::{ByteValue, MemoryConfig, ReadMode, inventory_bytes, softmax},
    numeric::{next_down as down, next_up as up},
};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub format: String,
    pub config: MemoryConfig,
    pub requests: Vec<ByteValue>,
    pub values: Vec<ByteValue>,
    pub contexts: Vec<Vec<f64>>,
    pub value_targets: Vec<Vec<f64>>,
    pub query_radius: Vec<f64>,
    pub row_radius: Vec<f64>,
    pub body_signature: String,
    pub query_steps: usize,
    pub fact_steps: usize,
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub revisions: Vec<RevisionProvenance>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct RevisionProvenance {
    pub parent_signature: String,
    pub data_fingerprint: String,
    pub steps: usize,
    pub learning_rate: f32,
}
pub struct Memory {
    pub parameters: Vec<Parameter>,
    pub metadata: MemoryMetadata,
    pub port: Option<Tensor>,
    seal: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct RevisionGuard {
    pub parent_signature: String,
    pub body_signature: String,
    pub port_signature: String,
    pub contract_signature: String,
}
pub const FORMAT: &str = "sejong-prepared-memory-v3";

fn layout(c: &MemoryConfig, values: usize) -> Vec<(String, Vec<usize>, bool)> {
    let d = c.dimension;
    let mut out = vec![];
    for layer in 0..c.layers {
        for name in ["anorm", "fnorm"] {
            out.push((format!("body.{layer}.{name}.weight"), vec![d], true));
            out.push((format!("body.{layer}.{name}.bias"), vec![1, d], false));
        }
        for name in ["q", "k", "v", "o"] {
            out.push((format!("body.{layer}.{name}.weight"), vec![d, d], false));
            if name != "k" {
                out.push((format!("body.{layer}.{name}.bias"), vec![1, d], false));
            }
        }
        out.push((
            format!("body.{layer}.up.weight"),
            vec![d, c.ffn_width],
            false,
        ));
        out.push((format!("body.{layer}.up.bias"), vec![1, c.ffn_width], false));
        out.push((
            format!("body.{layer}.down.weight"),
            vec![c.ffn_width, d],
            false,
        ));
        out.push((format!("body.{layer}.down.bias"), vec![1, d], false));
    }
    out.extend([
        ("body.out".into(), vec![d, d], false),
        ("key".into(), vec![d, c.hidden], false),
        ("value".into(), vec![c.hidden, d], false),
        ("readout".into(), vec![d, values], false),
        ("bias".into(), vec![1, values], false),
    ]);
    out
}
impl Memory {
    pub fn initialize(e: &mut Engine, metadata: MemoryMetadata) -> Result<Self> {
        metadata.config.validate()?;
        ensure(metadata.format == FORMAT, "prepared memory format mismatch")?;
        inventory_bytes(&metadata.requests, "memory requests")?;
        inventory_bytes(&metadata.values, "memory values")?;
        let c = &metadata.config;
        ensure(
            metadata.contexts.len() == metadata.requests.len()
                && metadata
                    .contexts
                    .iter()
                    .all(|r| r.len() == c.tokens * c.dimension && r.iter().all(|x| x.is_finite())),
            "declared context dimensions differ",
        )?;
        ensure(
            metadata.value_targets.len() == metadata.values.len()
                && metadata
                    .value_targets
                    .iter()
                    .all(|r| r.len() == c.dimension && r.iter().all(|x| x.is_finite())),
            "one finite target prototype per native value required",
        )?;
        let mut rng = Rng::new(c.seed);
        let mut parameters = vec![];
        for (name, shape, norm) in layout(c, metadata.values.len()) {
            let scale = 1. / (shape[0] as f32).sqrt();
            let data: Vec<_> = (0..shape.iter().product())
                .map(|_| {
                    if name == "bias" || name.ends_with(".bias") {
                        0.
                    } else if norm {
                        1.
                    } else {
                        rng.normal() * scale
                    }
                })
                .collect();
            parameters.push(Parameter {
                name,
                value: e.from_f32(&shape, &data)?,
            });
        }
        let mut memory = Self {
            parameters,
            metadata,
            port: None,
            seal: None,
        };
        let d = memory.metadata.config.dimension;
        let values = memory.metadata.values.len();
        let mut readout = vec![0.; d * values];
        let mut bias = vec![0.; values];
        for (i, target) in memory.metadata.value_targets.iter().enumerate() {
            for j in 0..d {
                readout[j * values + i] =
                    2. * target[j] / memory.metadata.config.readout_temperature;
            }
            bias[i] = -target.iter().map(|x| x * x).sum::<f64>()
                / memory.metadata.config.readout_temperature;
        }
        memory.replace_parameter("readout", e.from_f64(&[d, values], &readout)?)?;
        memory.replace_parameter("bias", e.from_f64(&[1, values], &bias)?)?;
        Ok(memory)
    }
    pub fn parameter(&self, name: &str) -> &Tensor {
        &self
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap()
            .value
    }
    pub fn bind(&self, g: &mut Graph, body: bool, facts: bool) -> BTreeMap<String, Var> {
        self.parameters
            .iter()
            .map(|p| {
                (
                    p.name.clone(),
                    g.leaf(
                        p.value.clone(),
                        if p.name.starts_with("body.") {
                            body
                        } else {
                            facts && (p.name == "key" || p.name == "value")
                        },
                    ),
                )
            })
            .collect()
    }
    pub fn query(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        contexts: &Tensor,
    ) -> Result<Var> {
        let c = &self.metadata.config;
        let batch = contexts.shape[0] / c.tokens;
        ensure(
            contexts.shape[1] == c.dimension && contexts.shape[0].is_multiple_of(c.tokens),
            "query context shape mismatch",
        )?;
        let mut x = g.leaf(contexts.clone(), false);
        for layer in 0..c.layers {
            let weight = |name: &str| w[&format!("body.{layer}.{name}")];
            let h = g.layer_norm(
                x,
                weight("anorm.weight"),
                weight("anorm.bias"),
                c.norm_epsilon,
            )?;
            let q = g.affine(h, weight("q.weight"), weight("q.bias"))?;
            let k = g.matmul(h, weight("k.weight"))?;
            let v = g.affine(h, weight("v.weight"), weight("v.bias"))?;
            let a = g.attention_mode(q, k, v, c.tokens, c.heads, false)?;
            let a = g.affine(a, weight("o.weight"), weight("o.bias"))?;
            let attention_state = g.add(x, a)?;
            let h = g.layer_norm(
                attention_state,
                weight("fnorm.weight"),
                weight("fnorm.bias"),
                c.norm_epsilon,
            )?;
            let h = g.affine(h, weight("up.weight"), weight("up.bias"))?;
            let h = g.relu(h)?;
            let f = g.affine(h, weight("down.weight"), weight("down.bias"))?;
            let delta = g.add(a, f)?;
            let delta = g.scale(delta, c.residual_scale)?;
            x = g.add(x, delta)?;
        }
        let indices = g.engine.from_i32(
            &[batch],
            &(0..batch)
                .map(|b| (b * c.tokens) as i32)
                .collect::<Vec<_>>(),
        )?;
        let first = g.gather(x, &indices)?;
        g.matmul(first, w["body.out"])
    }
    pub fn read_query(&self, g: &mut Graph, w: &BTreeMap<String, Var>, query: Var) -> Result<Var> {
        let h = g.matmul(query, w["key"])?;
        let h = g.gelu(h)?;
        g.matmul(h, w["value"])
    }
    pub fn replace_parameter(&mut self, name: &str, value: Tensor) -> Result<()> {
        ensure(
            self.seal.is_none(),
            "open a local revision before replacing a parameter",
        )?;
        let p = self
            .parameters
            .iter_mut()
            .find(|p| p.name == name)
            .ok_or_else(|| {
                sejong_core::Error::Message(format!("unknown memory parameter {name}"))
            })?;
        ensure(
            p.value.shape == value.shape,
            "replacement parameter shape mismatch",
        )?;
        p.value = value;
        Ok(())
    }
    pub fn logits(&self, g: &mut Graph, w: &BTreeMap<String, Var>, query: Var) -> Result<Var> {
        let h = self.read_query(g, w, query)?;
        let logits = g.matmul(h, w["readout"])?;
        let ones = g
            .engine
            .full(&[g.tensor(logits).shape[0], 1], g.tensor(logits).dtype, 1.)?;
        let ones = g.leaf(ones, false);
        let bias = g.matmul(ones, w["bias"])?;
        g.add(logits, bias)
    }
    pub fn body_signature(&self, e: &mut Engine) -> Result<String> {
        let params = self
            .parameters
            .iter()
            .filter(|p| p.name.starts_with("body."))
            .map(|p| Parameter {
                name: p.name.clone(),
                value: p.value.clone(),
            })
            .collect::<Vec<_>>();
        signature(e, &params, &self.metadata.config)
    }
    pub fn to_precision(&mut self, e: &mut Engine, dtype: DType) -> Result<()> {
        ensure(
            self.seal.is_none(),
            "open a local revision before changing precision",
        )?;
        for p in &mut self.parameters {
            if p.name != "readout" && p.name != "bias" {
                p.value = e.cast(&p.value, dtype)?;
            }
        }
        Ok(())
    }
    pub fn fit_port(&mut self, e: &mut Engine) -> Result<()> {
        self.to_precision(e, DType::F64)?;
        let c = &self.metadata.config;
        let d = c.dimension;
        let mut qr = StreamingQr::new(d + 1, d)?;
        for context in &self.metadata.contexts {
            let input = e.from_f64(&[c.tokens, d], context)?;
            let mut g = Graph::new(e, false);
            let weights = self.bind(&mut g, false, false);
            let query = self.query(&mut g, &weights, &input)?;
            let tensor = g.tensor(query).clone();
            let query = g.engine.to_f64(&tensor)?;
            let mut request = context[..d].to_vec();
            request.push(1.);
            qr.observe(request, query)?;
        }
        self.port = Some(e.from_f64(&[d + 1, d], &qr.solve()?)?);
        self.metadata.body_signature = self.body_signature(e)?;
        self.calibrate_internal(e)?;
        self.seal = Some(self.signature(e)?);
        Ok(())
    }
    pub fn row(&self, e: &mut Engine, index: usize, mode: ReadMode) -> Result<(Vec<f64>, f64)> {
        self.verify(e)?;
        let (law, _) = self.forward_row(e, index, mode, false)?;
        Ok((
            law,
            if mode == ReadMode::Direct {
                self.metadata.row_radius[index]
            } else {
                0.
            },
        ))
    }
    /// The same per-row forward is used by preparation and inference.
    fn forward_row(
        &self,
        e: &mut Engine,
        index: usize,
        mode: ReadMode,
        observe_query: bool,
    ) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
        ensure(
            index < self.metadata.requests.len(),
            "request outside declared input inventory",
        )?;
        let c = &self.metadata.config;
        let context = &self.metadata.contexts[index];
        let mut g = Graph::new(e, false);
        let weights = self.bind(&mut g, false, false);
        let query = match mode {
            ReadMode::Full => {
                let input = g.engine.from_f64(&[c.tokens, c.dimension], context)?;
                self.query(&mut g, &weights, &input)?
            }
            ReadMode::Direct => {
                let port = self.port.as_ref().ok_or_else(|| {
                    sejong_core::Error::Message("prepared affine port required".into())
                })?;
                let mut request = context[..c.dimension].to_vec();
                request.push(1.);
                let input = g.engine.from_f64(&[1, c.dimension + 1], &request)?;
                let input = g.leaf(input, false);
                let port = g.leaf(port.clone(), false);
                g.matmul(input, port)?
            }
        };
        let observed = if observe_query {
            let tensor = g.tensor(query).clone();
            Some(g.engine.to_f64(&tensor)?)
        } else {
            None
        };
        let logits = self.logits(&mut g, &weights, query)?;
        let tensor = g.tensor(logits).clone();
        Ok((softmax(&g.engine.to_f64(&tensor)?)?, observed))
    }
    pub fn calibrate(&mut self, e: &mut Engine) -> Result<()> {
        self.verify(e)?;
        self.calibrate_internal(e)?;
        self.seal = Some(self.signature(e)?);
        Ok(())
    }
    /// Radius against the normalized FP64 row law on every declared input.
    /// Preparation executes both paths and keeps scalar radii rather than row outputs.
    fn calibrate_internal(&mut self, e: &mut Engine) -> Result<()> {
        ensure(
            self.parameters.iter().all(|p| p.value.dtype == DType::F64),
            "calibration uses FP64 originals",
        )?;
        ensure(
            self.body_signature(e)? == self.metadata.body_signature,
            "query body changed: prepare a new port",
        )?;
        let mut query_radius = Vec::with_capacity(self.metadata.requests.len());
        let mut row_radius = Vec::with_capacity(self.metadata.requests.len());
        for index in 0..self.metadata.requests.len() {
            let (full, query) = self.forward_row(e, index, ReadMode::Full, true)?;
            let (direct, short) = self.forward_row(e, index, ReadMode::Direct, true)?;
            let query = query.unwrap();
            let short = short.unwrap();
            let mut squared = 0.;
            for (a, b) in query.iter().zip(&short) {
                let delta = up((a - b).abs());
                squared = up(squared + up(delta * delta));
            }
            query_radius.push(up(squared.sqrt()));
            row_radius.push(normalized_tv_upper(&full, &direct)?);
        }
        ensure(
            query_radius
                .iter()
                .chain(&row_radius)
                .all(|x| x.is_finite()),
            "nonfinite declared-row calibration",
        )?;
        self.metadata.query_radius = query_radius;
        self.metadata.row_radius = row_radius;
        Ok(())
    }
    pub fn verify(&self, e: &mut Engine) -> Result<()> {
        let seal = self.seal.as_ref().ok_or_else(|| {
            sejong_core::Error::Message("memory requires a sealed prepared snapshot".into())
        })?;
        ensure(
            self.signature(e)? == *seal,
            "memory changed after calibration; prepare a local revision",
        )
    }
    pub fn port_signature(&self, e: &mut Engine) -> Result<String> {
        let port = self
            .port
            .as_ref()
            .ok_or_else(|| sejong_core::Error::Message("prepared affine port required".into()))?;
        signature(
            e,
            &[Parameter {
                name: "port".into(),
                value: port.clone(),
            }],
            &self.metadata.body_signature,
        )
    }
    pub fn contract_signature(&self, e: &mut Engine) -> Result<String> {
        signature(
            e,
            &[
                Parameter {
                    name: "readout".into(),
                    value: self.parameter("readout").clone(),
                },
                Parameter {
                    name: "bias".into(),
                    value: self.parameter("bias").clone(),
                },
            ],
            &(
                &self.metadata.config,
                &self.metadata.requests,
                &self.metadata.values,
                &self.metadata.contexts,
                &self.metadata.value_targets,
            ),
        )
    }
    pub fn fork_revision(&self, e: &mut Engine) -> Result<(Self, RevisionGuard)> {
        self.verify(e)?;
        let mut parameters = vec![];
        for p in &self.parameters {
            let data = e.to_f64(&p.value)?;
            parameters.push(Parameter {
                name: p.name.clone(),
                value: e.from_f64(&p.value.shape, &data)?,
            });
        }
        let port = if let Some(p) = &self.port {
            {
                let data = e.to_f64(p)?;
                Some(e.from_f64(&p.shape, &data)?)
            }
        } else {
            None
        };
        let mut copy = Self {
            parameters,
            metadata: self.metadata.clone(),
            port,
            seal: self.seal.clone(),
        };
        let guard = copy.begin_revision(e)?;
        Ok((copy, guard))
    }
    pub fn begin_revision(&mut self, e: &mut Engine) -> Result<RevisionGuard> {
        self.verify(e)?;
        let guard = RevisionGuard {
            parent_signature: self.signature(e)?,
            body_signature: self.body_signature(e)?,
            port_signature: self.port_signature(e)?,
            contract_signature: self.contract_signature(e)?,
        };
        self.seal = None;
        Ok(guard)
    }
    pub fn finalize_revision(&mut self, e: &mut Engine, guard: &RevisionGuard) -> Result<()> {
        ensure(
            self.seal.is_none(),
            "revision requires an open local memory",
        )?;
        self.to_precision(e, DType::F64)?;
        ensure(
            self.body_signature(e)? == guard.body_signature
                && self.port_signature(e)? == guard.port_signature
                && self.contract_signature(e)? == guard.contract_signature,
            "local K/V revision must preserve the query body and affine port",
        )?;
        self.calibrate_internal(e)?;
        self.seal = Some(self.signature(e)?);
        Ok(())
    }
    pub fn read(
        &self,
        e: &mut Engine,
        inputs: &[Vec<f64>],
        mode: ReadMode,
    ) -> Result<Vec<Vec<f64>>> {
        self.verify(e)?;
        let c = &self.metadata.config;
        let d = c.dimension;
        let width = if mode == ReadMode::Full {
            d * c.tokens
        } else {
            d
        };
        ensure(
            !inputs.is_empty()
                && inputs
                    .iter()
                    .all(|r| r.len() == width && r.iter().all(|x| x.is_finite())),
            "continuous memory input shape mismatch",
        )?;
        let mut outputs = Vec::with_capacity(inputs.len());
        for row in inputs {
            let mut g = Graph::new(e, false);
            let weights = self.bind(&mut g, false, false);
            let query = if mode == ReadMode::Full {
                let x = g.engine.from_f64(&[c.tokens, d], row)?;
                self.query(&mut g, &weights, &x)?
            } else {
                let mut features = row.clone();
                features.push(1.);
                let x = g.engine.from_f64(&[1, d + 1], &features)?;
                let x = g.leaf(x, false);
                let p = g.leaf(self.port.as_ref().unwrap().clone(), false);
                g.matmul(x, p)?
            };
            let result = self.read_query(&mut g, &weights, query)?;
            let tensor = g.tensor(result).clone();
            let data = g.engine.to_f64(&tensor)?;
            ensure(
                data.iter().all(|x| x.is_finite()),
                "nonfinite continuous memory read",
            )?;
            outputs.push(data);
        }
        Ok(outputs)
    }

    pub fn save_training(&self, e: &mut Engine, path: &Path) -> Result<()> {
        atomic_directory(path, |dir| {
            let tensors = self
                .parameters
                .iter()
                .map(|p| (p.name.as_str(), &p.value))
                .chain(self.port.as_ref().map(|p| ("port", p)));
            save_tensors(e, &dir.join("weights.safetensors"), tensors)?;
            std::fs::write(
                dir.join("memory.json"),
                json::to_vec_pretty(&self.metadata)?,
            )?;
            Ok(())
        })
    }
    pub fn load_training(e: &mut Engine, path: &Path) -> Result<Self> {
        let metadata: MemoryMetadata = json::from_slice(&std::fs::read(path.join("memory.json"))?)?;
        let mut model = Self::initialize(e, metadata)?;
        let mut tensors = load_tensors(e, &path.join("weights.safetensors"))?;
        for p in &mut model.parameters {
            let value = tensors
                .remove(&p.name)
                .ok_or_else(|| sejong_core::Error::Message(format!("missing {}", p.name)))?;
            ensure(
                value.shape == p.value.shape && matches!(value.dtype, DType::F32 | DType::F64),
                "training memory tensor shape/dtype mismatch",
            )?;
            p.value = value;
        }
        model.port = tensors.remove("port");
        ensure(
            tensors.is_empty()
                && model.port.as_ref().is_none_or(|p| {
                    p.shape
                        == [
                            model.metadata.config.dimension + 1,
                            model.metadata.config.dimension,
                        ]
                        && p.dtype == DType::F64
                }),
            "training port contract mismatch",
        )?;
        Ok(model)
    }
    pub fn signature(&self, e: &mut Engine) -> Result<String> {
        let mut parameters = self
            .parameters
            .iter()
            .map(|p| Parameter {
                name: p.name.clone(),
                value: p.value.clone(),
            })
            .collect::<Vec<_>>();
        if let Some(port) = &self.port {
            parameters.push(Parameter {
                name: "port".into(),
                value: port.clone(),
            });
        }
        signature(e, &parameters, &self.metadata)
    }
    pub fn save(&self, e: &mut Engine, path: &Path) -> Result<()> {
        self.verify(e)?;
        atomic_directory(path, |dir| {
            let tensors = self
                .parameters
                .iter()
                .map(|p| (p.name.as_str(), &p.value))
                .chain(std::iter::once(("port", self.port.as_ref().unwrap())));
            save_tensors(e, &dir.join("weights.safetensors"), tensors)?;
            std::fs::write(
                dir.join("memory.json"),
                json::to_vec_pretty(&self.metadata)?,
            )?;
            std::fs::write(dir.join("signature"), self.signature(e)?)?;
            Ok(())
        })
    }
    pub fn load(e: &mut Engine, path: &Path) -> Result<Self> {
        let metadata: MemoryMetadata = json::from_slice(&std::fs::read(path.join("memory.json"))?)?;
        metadata.config.validate()?;
        ensure(metadata.format == FORMAT, "prepared memory format mismatch")?;
        inventory_bytes(&metadata.requests, "requests")?;
        inventory_bytes(&metadata.values, "values")?;
        let mut tensors = load_tensors(e, &path.join("weights.safetensors"))?;
        let port = tensors.remove("port");
        let mut parameters = vec![];
        for (name, shape, _) in layout(&metadata.config, metadata.values.len()) {
            let value = tensors
                .remove(&name)
                .ok_or_else(|| sejong_core::Error::Message(format!("missing {name}")))?;
            ensure(
                value.dtype == DType::F64 && value.shape == shape,
                "memory tensor contract mismatch",
            )?;
            parameters.push(Parameter { name, value });
        }
        let d = metadata.config.dimension;
        ensure(
            tensors.is_empty()
                && port
                    .as_ref()
                    .is_some_and(|p| p.shape == [d + 1, d] && p.dtype == DType::F64),
            "port tensor contract mismatch",
        )?;
        ensure(
            metadata.contexts.len() == metadata.requests.len()
                && metadata.contexts.iter().all(|r| {
                    r.len() == d * metadata.config.tokens && r.iter().all(|x| x.is_finite())
                }),
            "stored context contract mismatch",
        )?;
        let mut memory = Self {
            parameters,
            metadata,
            port,
            seal: None,
        };
        ensure(
            memory.signature(e)? == std::fs::read_to_string(path.join("signature"))?,
            "memory signature mismatch",
        )?;
        let stored = memory.metadata.row_radius.clone();
        memory.calibrate_internal(e)?;
        ensure(
            stored.len() == memory.metadata.row_radius.len()
                && stored
                    .iter()
                    .zip(&memory.metadata.row_radius)
                    .all(|(stored, current)| stored >= current),
            "stored bounds differ from current original",
        )?;
        memory.metadata.row_radius = stored;
        memory.seal = Some(memory.signature(e)?);
        Ok(memory)
    }
}

fn normalization_defect(law: &[f64]) -> f64 {
    let (mut lower, mut upper) = (0., 0.);
    for &p in law {
        lower = down(lower + p);
        upper = up(upper + p);
    }
    up((1. - lower).abs().max((upper - 1.).abs()))
}
fn normalized_tv_upper(full: &[f64], direct: &[f64]) -> Result<f64> {
    ensure(
        full.len() == direct.len()
            && !full.is_empty()
            && full.iter().chain(direct).all(|x| x.is_finite() && *x >= 0.),
        "finite row laws required",
    )?;
    let mut l1 = 0.;
    for (p, q) in full.iter().zip(direct) {
        l1 = up(l1 + up((p - q).abs()));
    }
    let with_full = up(l1 + normalization_defect(full));
    let with_both = up(with_full + normalization_defect(direct));
    Ok(up(with_both * 0.5).min(1.))
}
/// Online Givens QR retains storage determined by coordinate dimension.
struct StreamingQr {
    columns: usize,
    outputs: usize,
    rows: usize,
    r: Vec<f64>,
    z: Vec<f64>,
}
impl StreamingQr {
    fn new(columns: usize, outputs: usize) -> Result<Self> {
        let r = columns
            .checked_mul(columns)
            .ok_or_else(|| sejong_core::Error::Message("QR shape overflow".into()))?;
        let z = columns
            .checked_mul(outputs)
            .ok_or_else(|| sejong_core::Error::Message("QR target shape overflow".into()))?;
        Ok(Self {
            columns,
            outputs,
            rows: 0,
            r: vec![0.; r],
            z: vec![0.; z],
        })
    }
    fn observe(&mut self, mut x: Vec<f64>, mut y: Vec<f64>) -> Result<()> {
        ensure(
            x.len() == self.columns
                && y.len() == self.outputs
                && x.iter().chain(&y).all(|x| x.is_finite()),
            "finite paired query coordinates required",
        )?;
        let n = self.columns;
        for j in 0..n {
            let diagonal = self.r[j * n + j];
            let incoming = x[j];
            if incoming == 0. {
                continue;
            }
            let norm = diagonal.hypot(incoming);
            let cosine = diagonal / norm;
            let sine = incoming / norm;
            for k in j..n {
                let old = self.r[j * n + k];
                self.r[j * n + k] = cosine * old + sine * x[k];
                x[k] = -sine * old + cosine * x[k];
            }
            for k in 0..self.outputs {
                let old = self.z[j * self.outputs + k];
                self.z[j * self.outputs + k] = cosine * old + sine * y[k];
                y[k] = -sine * old + cosine * y[k];
            }
        }
        self.rows = self.rows.checked_add(1).ok_or_else(|| {
            sejong_core::Error::Message("query observation count overflow".into())
        })?;
        Ok(())
    }
    fn solve(self) -> Result<Vec<f64>> {
        let n = self.columns;
        ensure(
            self.rows >= n,
            "query observations must identify all affine coordinates",
        )?;
        let scale = self.r.iter().fold(0f64, |norm, x| norm.hypot(*x));
        let threshold = f64::EPSILON * (self.rows.max(n) as f64) * scale;
        let mut solution = vec![0.; n * self.outputs];
        for j in (0..n).rev() {
            ensure(
                self.r[j * n + j].abs() > threshold,
                "query observations are numerically rank deficient at matrix precision",
            )?;
            for output in 0..self.outputs {
                let mut value = self.z[j * self.outputs + output];
                for k in j + 1..n {
                    value -= self.r[j * n + k] * solution[k * self.outputs + output];
                }
                solution[j * self.outputs + output] = value / self.r[j * n + j];
            }
        }
        ensure(
            solution.iter().all(|x| x.is_finite()),
            "finite affine query fit required",
        )?;
        Ok(solution)
    }
}
