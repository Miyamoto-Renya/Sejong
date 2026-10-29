//! Variable-cardinality semantic pointers and compositional query decoding.
use super::{
    atomic_directory,
    bert::{BertEncoder, Features},
    signature,
};
use crate::{Parameter, load_tensors, save_tensors};
use sejong_core::{
    DType, Error, ModelConfig, Result, Rng, ensure, json,
    native::{LocalExample, Program, Role, Step, inventory, softmax},
};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, rc::Rc};

pub const FORMAT: &str = "sejong-role-dynamic-v3";
#[derive(Clone, Serialize, Deserialize)]
pub struct ClassifierMetadata {
    pub role: Role,
    pub phrases: Vec<String>,
    pub meanings: Vec<String>,
    pub steps: usize,
    pub source_hash: String,
    pub loss: f64,
}
#[derive(Serialize, Deserialize)]
struct Stored {
    format: String,
    metadata: ClassifierMetadata,
    config: ModelConfig,
    encoder_digest: Option<String>,
}
#[derive(Clone)]
struct Encoded {
    tokens: Vec<i32>,
    ids: Vec<i32>,
    present: Vec<bool>,
    occurrences: usize,
    people: usize,
    predicates: usize,
}
pub struct Classifier {
    pub metadata: ClassifierMetadata,
    pub config: ModelConfig,
    parameters: Vec<Parameter>,
    encoder: Option<Rc<BertEncoder>>,
}
#[derive(Clone)]
enum Output {
    Direct {
        var: Var,
        indices: Vec<usize>,
        probabilities: Vec<f64>,
    },
    Factor {
        active: Var,
        probability: f64,
        roles: Var,
        pair: Vec<(usize, usize)>,
        conditional: Vec<f64>,
        rows: usize,
    },
    Request {
        active: Var,
        probability: f64,
        goal: Var,
        goal_probability: Vec<f64>,
        selector: Var,
        selector_probability: Vec<f64>,
    },
    Sequence(Vec<(Option<Var>, Vec<f64>, usize)>),
    Constant,
}
pub struct Evaluation {
    pub log_probabilities: Vec<f64>,
    output: Output,
}
impl Evaluation {
    pub fn conditionals(&self) -> Option<Vec<&[f64]>> {
        match &self.output {
            Output::Sequence(nodes) => Some(nodes.iter().map(|(_, p, _)| p.as_slice()).collect()),
            _ => None,
        }
    }
    pub fn seed_conditionals(
        &self,
        g: &mut Graph,
        gradients: &[Vec<f64>],
    ) -> Result<Vec<(Var, Tensor)>> {
        let Output::Sequence(nodes) = &self.output else {
            return Err(Error::Message("conditional query output required".into()));
        };
        ensure(
            nodes.len() == gradients.len(),
            "conditional derivative count differs",
        )?;
        let mut roots = vec![];
        for ((var, p, _), gradient) in nodes.iter().zip(gradients) {
            ensure(
                p.len() == gradient.len(),
                "conditional derivative shape differs",
            )?;
            if let Some(var) = var {
                let sum: f64 = gradient.iter().sum();
                let values: Vec<_> = gradient
                    .iter()
                    .zip(p)
                    .map(|(a, p)| (a - p * sum) as f32)
                    .collect();
                roots.push((*var, g.engine.from_f32(&g.tensor(*var).shape, &values)?));
            }
        }
        Ok(roots)
    }
    pub fn seed(&self, g: &mut Graph, gradient: &[f64]) -> Result<Vec<(Var, Tensor)>> {
        ensure(
            gradient.len() == self.log_probabilities.len(),
            "operator derivative shape differs",
        )?;
        let make = |g: &mut Graph, v: Var, x: Vec<f32>| -> Result<(Var, Tensor)> {
            Ok((v, g.engine.from_f32(&g.tensor(v).shape, &x)?))
        };
        match &self.output {
            Output::Constant => Ok(vec![]),
            Output::Direct {
                var,
                indices,
                probabilities,
            } => {
                let sum: f64 = indices.iter().map(|i| gradient[*i]).sum();
                Ok(vec![make(
                    g,
                    *var,
                    indices
                        .iter()
                        .zip(probabilities)
                        .map(|(i, p)| (gradient[*i] - p * sum) as f32)
                        .collect(),
                )?])
            }
            Output::Factor {
                active,
                probability,
                roles,
                pair,
                conditional,
                rows,
            } => {
                let none = gradient.len() - 1;
                let sum: f64 = gradient[..none].iter().sum();
                let mut scores = vec![0.; rows * 2];
                for (i, (a, b)) in pair.iter().enumerate() {
                    let d = (gradient[i] - conditional[i] * sum) as f32;
                    scores[a * 2] += d;
                    scores[b * 2 + 1] += d;
                }
                Ok(vec![
                    make(
                        g,
                        *active,
                        vec![(sum * (1. - probability) - gradient[none] * probability) as f32],
                    )?,
                    make(g, *roles, scores)?,
                ])
            }
            Output::Request {
                active,
                probability,
                goal,
                goal_probability,
                selector,
                selector_probability,
            } => {
                let n = selector_probability.len();
                let none = 2 * n;
                let sum: f64 = gradient[..none].iter().sum();
                let dg = (0..2)
                    .map(|a| {
                        ((0..n).map(|b| gradient[a * n + b]).sum::<f64>()
                            - goal_probability[a] * sum) as f32
                    })
                    .collect();
                let ds = (0..n)
                    .map(|b| {
                        ((0..2).map(|a| gradient[a * n + b]).sum::<f64>()
                            - selector_probability[b] * sum) as f32
                    })
                    .collect();
                Ok(vec![
                    make(
                        g,
                        *active,
                        vec![(sum * (1. - probability) - gradient[none] * probability) as f32],
                    )?,
                    make(g, *goal, dg)?,
                    make(g, *selector, ds)?,
                ])
            }
            Output::Sequence(nodes) => nodes
                .iter()
                .filter_map(|(v, p, selected)| {
                    v.map(|v| {
                        make(
                            g,
                            v,
                            p.iter()
                                .enumerate()
                                .map(|(i, p)| {
                                    (gradient[0] * (f64::from(i == *selected) - p)) as f32
                                })
                                .collect(),
                        )
                    })
                })
                .collect(),
        }
    }
}
#[derive(Clone)]
struct QueryFeatures {
    global: Tensor,
    people: Tensor,
    predicates: Tensor,
    present: Vec<bool>,
    count: usize,
    occurrences: usize,
    engine: usize,
}
pub struct QueryPlan {
    pub roots: Vec<(Option<Program>, f64)>,
    pub predicates: Vec<usize>,
    context: QueryFeatures,
}
impl QueryPlan {
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.roots.capacity() * std::mem::size_of::<(Option<Program>, f64)>()
            + self
                .roots
                .iter()
                .filter_map(|(p, _)| p.as_ref())
                .map(|p| p.steps.capacity() * std::mem::size_of::<Step>())
                .sum::<usize>()
            + self.predicates.capacity() * std::mem::size_of::<usize>()
            + self.context.present.capacity() * std::mem::size_of::<bool>()
            + [
                &self.context.global,
                &self.context.people,
                &self.context.predicates,
            ]
            .iter()
            .map(|t| t.len() * t.dtype.bytes() + t.shape.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
    }
}

fn host_admit(operation: &str, items: &[(usize, usize)]) -> Result<()> {
    let required = items.iter().try_fold(0usize, |sum, (count, bytes)| {
        count
            .checked_mul(*bytes)
            .and_then(|n| sum.checked_add(n))
            .ok_or_else(|| Error::Message(format!("{operation} host size overflow")))
    })?;
    let available = sejong_core::native::host_available_bytes()?;
    if required > available {
        return Err(Error::Capacity {
            operation: operation.into(),
            required,
            available,
        });
    }
    Ok(())
}

fn layout(role: Role, c: &ModelConfig, input: usize) -> Vec<(String, Vec<usize>, bool)> {
    let d = c.width;
    let mut out = vec![];
    fn affine(out: &mut Vec<(String, Vec<usize>, bool)>, name: &str, a: usize, b: usize) {
        out.push((format!("{name}.weight"), vec![a, b], false));
        out.push((format!("{name}.bias"), vec![1, b], false));
    }
    if matches!(role, Role::Scene | Role::Instruction | Role::Request) {
        affine(&mut out, "project.linear", input, d);
        affine(&mut out, "active", d, 1);
        if role == Role::Request {
            affine(&mut out, "goal", d, 2);
            affine(&mut out, "explicit.0", 4 * d, d);
            affine(&mut out, "explicit.1", d, 1);
            affine(&mut out, "abstract", d, 3);
        } else {
            affine(&mut out, "roles.0", 4 * d, d);
            affine(&mut out, "roles.1", d, 2);
        }
        for (name, width) in [("project.input_norm", input), ("project.output_norm", d)] {
            out.push((format!("{name}.weight"), vec![width], true));
            out.push((format!("{name}.bias"), vec![1, width], false));
        }
        if role == Role::Instruction {
            out.push(("null".into(), vec![1, d], false));
        }
    } else {
        out.push(("embedding".into(), vec![261, d], false));
        for layer in 0..c.layers {
            let p = format!("encoder.{layer}");
            // A shared key bias cancels from every softmax row, so the projection trains Q/V biases.
            out.push((format!("{p}.qkv.weight"), vec![d, 3 * d], false));
            out.push((format!("{p}.query.bias"), vec![1, d], false));
            out.push((format!("{p}.value.bias"), vec![1, d], false));
            for (name, a, b) in [
                ("out", d, d),
                ("ff.0", d, c.ffn_width),
                ("ff.1", c.ffn_width, d),
            ] {
                affine(&mut out, &format!("{p}.{name}"), a, b);
            }
            for name in ["norm1", "norm2"] {
                out.push((format!("{p}.{name}.weight"), vec![d], true));
                out.push((format!("{p}.{name}.bias"), vec![1, d], false));
            }
            out.push((format!("{p}.distance"), vec![2, c.heads], false));
            out.push((format!("{p}.same"), vec![1, c.heads], false));
        }
        out.push(("final_norm.weight".into(), vec![d], true));
        out.push(("final_norm.bias".into(), vec![1, d], false));
        if role == Role::Lexical {
            affine(&mut out, "lexical", d, 2);
        } else {
            out.push(("goal_embedding".into(), vec![2, d], false));
            out.push(("operation_embedding".into(), vec![4, d], false));
            affine(&mut out, "root.0", 3 * d, d);
            affine(&mut out, "root.1", d, 1);
            affine(&mut out, "none", d, 1);
            affine(&mut out, "initial", 3 * d, d);
            affine(&mut out, "recurrent", 3 * d, d);
            affine(&mut out, "step.0", 4 * d, d);
            affine(&mut out, "step.1", d, 1);
            affine(&mut out, "stop", 2 * d, 1);
        }
    }
    out
}
fn encoded(text: &str, people: &[String], predicates: &[String]) -> Result<Encoded> {
    let names: Vec<_> = people.iter().chain(predicates).collect();
    let set: std::collections::BTreeSet<_> = names.iter().collect();
    ensure(
        set.len() == names.len() && names.iter().all(|n| !n.is_empty()),
        "distinct nonempty query identities required",
    )?;
    let mut order: Vec<_> = (0..names.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(names[*i].len()));
    let mut tokens = vec![258];
    let mut ids = vec![-1];
    let mut present = vec![false; names.len()];
    let mut cursor = 0;
    let mut occurrences = 0;
    while cursor < text.len() {
        let before = text[..cursor].chars().next_back();
        let found = order.iter().copied().find(|i| {
            let n = names[*i].as_str();
            let end = cursor + n.len();
            text[cursor..].starts_with(n)
                && before.is_none_or(|c| !c.is_alphanumeric())
                && (end == text.len()
                    || text[end..]
                        .chars()
                        .next()
                        .is_none_or(|c| !c.is_alphanumeric()))
        });
        if let Some(slot) = found {
            tokens.push(if slot < people.len() { 256 } else { 260 });
            ids.push(slot as i32);
            present[slot] = true;
            occurrences += usize::from(slot >= people.len());
            cursor += names[slot].len();
        } else {
            let c = text[cursor..].chars().next().unwrap();
            for b in c.to_string().bytes() {
                tokens.push(b as i32);
                ids.push(-1);
            }
            cursor += c.len_utf8();
        }
    }
    tokens.push(259);
    ids.push(-1);
    Ok(Encoded {
        tokens,
        ids,
        present,
        occurrences,
        people: people.len(),
        predicates: predicates.len(),
    })
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
fn logsigmoid(x: f64) -> f64 {
    if x >= 0. {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}
fn logsoftmax(xs: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let p = softmax(xs)?;
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let logz = max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    Ok((xs.iter().map(|x| x - logz).collect(), p))
}
impl Classifier {
    pub fn initialize(e: &mut Engine, c: ModelConfig, m: ClassifierMetadata) -> Result<Self> {
        Self::initialize_with_encoder(e, c, m, None)
    }
    pub fn initialize_with_encoder(
        e: &mut Engine,
        c: ModelConfig,
        mut metadata: ClassifierMetadata,
        path: Option<&Path>,
    ) -> Result<Self> {
        ensure(
            c.width > 0
                && c.layers > 0
                && c.heads > 0
                && c.width.is_multiple_of(c.heads)
                && c.ffn_width > 0
                && c.norm_epsilon.is_finite()
                && c.norm_epsilon > 0.,
            "positive head-aligned local model configuration required",
        )?;
        if metadata.role == Role::Lexical {
            metadata.phrases = metadata.phrases.iter().map(|p| p.to_lowercase()).collect();
            inventory(&metadata.phrases, "lexical phrases")?;
            ensure(
                metadata.meanings == ["use", "ignore"],
                "lexical operation inventory is [use,ignore]",
            )?;
        }
        let encoder = if matches!(
            metadata.role,
            Role::Scene | Role::Instruction | Role::Request
        ) {
            Some(BertEncoder::load_shared(
                e,
                path.unwrap_or(&BertEncoder::default_path()),
            )?)
        } else {
            None
        };
        let input = encoder.as_ref().map_or(0, |e| e.config.hidden_size);
        let mut rng = Rng::new(c.seed);
        let mut parameters = vec![];
        for (name, shape, norm) in layout(metadata.role, &c, input) {
            let zero = name.ends_with(".bias")
                || name.ends_with(".distance")
                || name.ends_with(".same")
                || name == "null";
            let scale = 1. / (shape[0] as f32).sqrt();
            let mut values: Vec<_> = (0..shape.iter().product())
                .map(|_| {
                    if norm {
                        1.
                    } else if zero {
                        0.
                    } else {
                        rng.normal() * scale
                    }
                })
                .collect();
            if name == "embedding" {
                let d = shape[1];
                values[257 * d..258 * d].fill(0.);
            }
            parameters.push(Parameter {
                name,
                value: e.from_f32(&shape, &values)?,
            });
        }
        Ok(Self {
            metadata,
            config: c,
            parameters,
            encoder,
        })
    }
    pub fn parameters(&self) -> Vec<Parameter> {
        self.parameters
            .iter()
            .map(|p| Parameter {
                name: p.name.clone(),
                value: p.value.clone(),
            })
            .collect()
    }
    pub fn parameter_count(&self) -> usize {
        self.trainable_parameter_count()
            + self
                .encoder
                .as_ref()
                .map_or(0, |e| e.parameters.iter().map(|p| p.value.len()).sum())
    }
    pub fn trainable_parameter_count(&self) -> usize {
        self.parameters.iter().map(|p| p.value.len()).sum()
    }
    pub fn encoder_identity(&self) -> Option<&super::bert::EncoderIdentity> {
        self.encoder.as_ref().map(|e| &e.identity)
    }
    pub fn example_memory_bytes(&self, row: &LocalExample) -> Result<usize> {
        fn allocation(elements: u128) -> Result<usize> {
            if elements == 0 {
                return Ok(0);
            }
            let bytes = usize::try_from(
                elements
                    .checked_mul(std::mem::size_of::<f32>() as u128)
                    .ok_or_else(|| Error::Message("parser workspace byte overflow".into()))?,
            )
            .map_err(|_| Error::Message("parser workspace shape overflow".into()))?;
            Engine::allocation_bytes(bytes)
        }
        fn sum(terms: &[(u128, usize)]) -> Result<usize> {
            usize::try_from(terms.iter().try_fold(0u128, |s, (n, bytes)| {
                n.checked_mul(*bytes as u128)
                    .and_then(|n| s.checked_add(n))
                    .ok_or_else(|| Error::Message("parser workspace sum overflow".into()))
            })?)
            .map_err(|_| Error::Message("parser workspace size overflow".into()))
        }
        let d = self.config.width as u128;
        let n = row.people.len() as u128;
        let parameter_gradients = allocation(self.trainable_parameter_count() as u128)?;
        if let Some(encoder) = &self.encoder {
            let (text, _, _) = super::bert::aliased(&row.text, &row.people, &[])?;
            let t = encoder.tokenizer.tokenize(&text)?.ids.len() as u128;
            let h = encoder.config.hidden_size as u128;
            let f = encoder.config.intermediate_size as u128;
            // Six embedding/residual buffers and eight per-block Q/K/V/output/residual buffers;
            // two live FFN tensors, the largest affine temporary, and attention normalizers.
            let frozen = sum(&[
                (6 + 8, allocation(t * h)?),
                (2, allocation(t * f)?),
                (1, allocation(t * h.max(f))?),
                (
                    1,
                    allocation(t * encoder.config.num_attention_heads as u128)?,
                ),
                (3, allocation(t)?),
            ])?;
            let rows = n + u128::from(self.metadata.role == Role::Instruction);
            // Projection: four row tensors. Role features: repeat/product/negative/difference/concat.
            // The role MLP retains matmul/bias/GELU; coordinate scores retain matmul and bias.
            let local = sum(&[
                (4, allocation((n + 1) * d)?),
                (1, allocation((n + 1) * h)?),
                (1, allocation(d)?),
                (1, allocation(n * d)?),
                (1, allocation(rows * d)?),
                (4 + 4 + 3, allocation(rows * d)?),
                (2, allocation(rows * 2)?),
                (2, allocation(1)?),
                (4, allocation(3)?),
            ])?;
            return Ok(frozen.max(sum(&[(2, local), (1, parameter_gradients)])?));
        }
        let input = encoded(&row.text, &row.people, &row.predicates)?;
        let t = input.tokens.len() as u128;
        let p = row.predicates.len() as u128;
        let layers = self.config.layers as u128;
        let heads = self.config.heads as u128;
        // Each block retains 17 width tensors (Q/V-only biases), three FFN tensors,
        // three projected position/identity bias tensors and the attention normalizer.
        let encoder = sum(&[
            (2 + 17 * layers, allocation(t * d)?),
            (3 * layers, allocation(t * self.config.ffn_width as u128)?),
            (3 * layers, allocation(t * t * heads)?),
            (layers, allocation(t * heads)?),
            (1, allocation(t * t * 2)?),
            (1, allocation(t * t)?),
            (1, allocation((n + p) * t)?),
            (1, allocation((n + p) * d)?),
            (2, allocation(t)?),
        ])?;
        let decoder = if self.metadata.role == Role::Query {
            let transitions = row
                .program
                .as_ref()
                .map_or(0, |q| q.steps.len() as u128 + 1);
            let prefix_steps = transitions * transitions.saturating_sub(1) / 2;
            // Root gathers/concatenation/MLP: 18 person-width tensors. Each action
            // uses four gathers, four-column concat, matmul/bias/GELU (11 widths).
            // Initial and recurrent states each retain eight width tensors.
            sum(&[
                (18, allocation(n * d)?),
                (4, allocation(n)?),
                (2, allocation(1)?),
                (11 * transitions, allocation(4 * p * d)?),
                (8 * (transitions + prefix_steps), allocation(d)?),
                (2 * transitions, allocation(d)?),
                (4 * transitions, allocation(4 * p + 1)?),
            ])?
        } else {
            sum(&[(2, allocation(2)?)])?
        };
        sum(&[(2, encoder), (2, decoder), (1, parameter_gradients)])
    }

    fn bind(&self, g: &mut Graph, training: bool) -> (BTreeMap<String, Var>, Vec<Var>) {
        let leaves: Vec<_> = self
            .parameters
            .iter()
            .map(|p| g.leaf(p.value.clone(), training))
            .collect();
        (
            self.parameters
                .iter()
                .zip(&leaves)
                .map(|(p, v)| (p.name.clone(), *v))
                .collect(),
            leaves,
        )
    }
    fn affine(g: &mut Graph, w: &BTreeMap<String, Var>, x: Var, name: &str) -> Result<Var> {
        g.affine(x, w[&format!("{name}.weight")], w[&format!("{name}.bias")])
    }
    fn norm(&self, g: &mut Graph, w: &BTreeMap<String, Var>, x: Var, name: &str) -> Result<Var> {
        g.layer_norm(
            x,
            w[&format!("{name}.weight")],
            w[&format!("{name}.bias")],
            self.config.norm_epsilon,
        )
    }
    fn gather(g: &mut Graph, x: Var, ids: &[i32]) -> Result<Var> {
        let ids = g.engine.from_i32(&[ids.len()], ids)?;
        g.gather(x, &ids)
    }
    fn repeat(g: &mut Graph, x: Var, n: usize) -> Result<Var> {
        Self::gather(g, x, &vec![0; n])
    }
    fn features(&self, e: &mut Engine, row: &LocalExample) -> Result<Features> {
        let key = json::to_string(&(&row.text, &row.people, &row.predicates))?;
        let encoder = self.encoder.as_ref().unwrap();
        let digest = blake3::hash(&json::to_vec(&(FORMAT, &encoder.identity.digest, &key))?)
            .to_hex()
            .to_string();
        let root = sejong_core::root().join(".cache/parser-features-v3");
        let path = root.join(format!("{digest}.json"));
        let slots = row.people.len();
        let d = encoder.config.hidden_size;
        let f = if path.exists() {
            let (stored, f): (String, Features) = json::from_slice(&std::fs::read(&path)?)?;
            ensure(
                stored == digest
                    && f.present.len() == slots
                    && f.rows.len() == (1 + slots) * d
                    && f.rows.iter().all(|x| x.is_finite()),
                "frozen feature cache differs from input/model dimensions",
            )?;
            f
        } else {
            let f = encoder.features(e, &row.text, &row.people, &[])?;
            std::fs::create_dir_all(&root)?;
            let temp = root.join(format!("{digest}.{}.tmp", std::process::id()));
            std::fs::write(&temp, json::to_vec(&(digest, f.clone()))?)?;
            std::fs::rename(temp, path)?;
            f
        };
        Ok(f)
    }
    fn role_features(g: &mut Graph, entities: Var, global: Var, n: usize) -> Result<Var> {
        let repeated = Self::repeat(g, global, n)?;
        let product = g.mul(entities, repeated)?;
        let negative = g.scale(repeated, -1.)?;
        let difference = g.add(entities, negative)?;
        g.concat_columns(&[entities, repeated, product, difference])
    }
    fn direct(
        g: &mut Graph,
        scores: Var,
        indices: Vec<usize>,
        size: usize,
        leaves: Vec<Var>,
    ) -> Result<(Evaluation, Vec<Var>)> {
        let values: Vec<f64> = g.value(scores)?.into_iter().map(f64::from).collect();
        let (log, p) = logsoftmax(&values)?;
        let mut law = vec![f64::NEG_INFINITY; size];
        for (i, value) in indices.iter().zip(log) {
            law[*i] = value;
        }
        Ok((
            Evaluation {
                log_probabilities: law,
                output: Output::Direct {
                    var: scores,
                    indices,
                    probabilities: p,
                },
            },
            leaves,
        ))
    }
    pub fn supports_program(&self, row: &LocalExample, program: &Option<Program>) -> Result<bool> {
        let input = encoded(&row.text, &row.people, &row.predicates)?;
        Ok(program.as_ref().is_none_or(|p| {
            p.goal < 2
                && p.anchor < input.people
                && input.present[p.anchor]
                && p.steps.len() <= input.occurrences
                && p.steps.iter().all(|s| {
                    s.predicate < input.predicates && input.present[input.people + s.predicate]
                })
        }))
    }
    pub fn target_index(&self, row: &LocalExample) -> usize {
        if self.metadata.role == Role::Query {
            0
        } else {
            row.label
        }
    }
    pub fn evaluate(
        &self,
        g: &mut Graph,
        row: &LocalExample,
        training: bool,
    ) -> Result<(Evaluation, Vec<Var>)> {
        let (w, leaves) = self.bind(g, training);
        if self.metadata.role == Role::Query {
            return self.teacher_query(g, &w, row, leaves);
        }
        if self.metadata.role == Role::Lexical {
            let phrase = row.text.to_lowercase();
            ensure(
                self.metadata.phrases.contains(&phrase),
                "unregistered lexical phrase",
            )?;
            let input = encoded(&phrase, &[], &[])?;
            let (global, _, _) = self.encode(g, &w, &input)?;
            let scores = Self::affine(g, &w, global, "lexical")?;
            return Self::direct(g, scores, vec![0, 1], 2, leaves);
        }
        let n = row.people.len();
        let classes = self
            .metadata
            .role
            .classes_for(n, self.metadata.meanings.len())?;
        // Categorical law, conditional law, logits and two identity coordinates per pair.
        host_admit(
            "semantic categorical law",
            &[(
                classes,
                3 * std::mem::size_of::<f64>() + 2 * std::mem::size_of::<usize>(),
            )],
        )?;
        if n == 0 && self.metadata.role != Role::Request {
            return Ok((
                Evaluation {
                    log_probabilities: vec![0.],
                    output: Output::Constant,
                },
                leaves,
            ));
        }
        let features = self.features(g.engine, row)?;
        let d = self.encoder.as_ref().unwrap().config.hidden_size;
        let input = g.engine.from_f32(&[n + 1, d], &features.rows)?;
        let input = g.leaf(input, false);
        let input = self.norm(g, &w, input, "project.input_norm")?;
        let input = Self::affine(g, &w, input, "project.linear")?;
        let input = g.gelu(input)?;
        let x = self.norm(g, &w, input, "project.output_norm")?;
        let global = Self::gather(g, x, &[0])?;
        let entities = if n > 0 {
            Some(Self::gather(
                g,
                x,
                &(1..=n).map(|i| i as i32).collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        let active = Self::affine(g, &w, global, "active")?;
        let t = g.value(active)?[0] as f64;
        if self.metadata.role == Role::Request {
            let abs = Self::affine(g, &w, global, "abstract")?;
            let selector = if let Some(entities) = entities {
                let f = Self::role_features(g, entities, global, n)?;
                let h = Self::affine(g, &w, f, "explicit.0")?;
                let h = g.gelu(h)?;
                let v = Self::affine(g, &w, h, "explicit.1")?;
                let v = g.reshape(v, &[1, n])?;
                g.concat_columns(&[v, abs])?
            } else {
                abs
            };
            let goal = Self::affine(g, &w, global, "goal")?;
            let mut values: Vec<_> = g.value(selector)?.into_iter().map(f64::from).collect();
            for (i, p) in features.present.iter().enumerate() {
                if !p {
                    values[i] = f64::NEG_INFINITY;
                }
            }
            let (ls, sp) = logsoftmax(&values)?;
            let (lg, gp) = logsoftmax(
                &g.value(goal)?
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>(),
            )?;
            let mut law = vec![];
            for a in 0..2 {
                for b in 0..n + 3 {
                    law.push(logsigmoid(t) + lg[a] + ls[b]);
                }
            }
            law.push(logsigmoid(-t));
            return Ok((
                Evaluation {
                    log_probabilities: law,
                    output: Output::Request {
                        active,
                        probability: sigmoid(t),
                        goal,
                        goal_probability: gp,
                        selector,
                        selector_probability: sp,
                    },
                },
                leaves,
            ));
        }
        if !features.present.iter().any(|p| *p) {
            let mut law = vec![f64::NEG_INFINITY; classes];
            *law.last_mut().unwrap() = 0.;
            return Ok((
                Evaluation {
                    log_probabilities: law,
                    output: Output::Constant,
                },
                leaves,
            ));
        }
        let instruction = self.metadata.role == Role::Instruction;
        let mut entities = entities.unwrap();
        if instruction {
            entities = g.concat_rows(&[entities, w["null"]])?;
        }
        let rows = n + usize::from(instruction);
        let f = Self::role_features(g, entities, global, rows)?;
        let h = Self::affine(g, &w, f, "roles.0")?;
        let h = g.gelu(h)?;
        let roles = Self::affine(g, &w, h, "roles.1")?;
        let scores = g.value(roles)?;
        let mut pair = vec![];
        let mut logits = vec![];
        for a in 0..rows {
            for b in 0..rows {
                if instruction && a == n && b == n {
                    continue;
                }
                pair.push((a, b));
                logits.push(
                    if (a == n || features.present[a]) && (b == n || features.present[b]) {
                        scores[a * 2] as f64 + scores[b * 2 + 1] as f64
                    } else {
                        f64::NEG_INFINITY
                    },
                );
            }
        }
        let (log, prob) = logsoftmax(&logits)?;
        let mut law: Vec<_> = log.into_iter().map(|v| logsigmoid(t) + v).collect();
        law.push(logsigmoid(-t));
        Ok((
            Evaluation {
                log_probabilities: law,
                output: Output::Factor {
                    active,
                    probability: sigmoid(t),
                    roles,
                    pair,
                    conditional: prob,
                    rows,
                },
            },
            leaves,
        ))
    }
    fn encode(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        input: &Encoded,
    ) -> Result<(Var, Option<Var>, Option<Var>)> {
        let t = input.tokens.len();
        let d = self.config.width;
        let tt = t
            .checked_mul(t)
            .ok_or_else(|| Error::Message("parser attention size overflow".into()))?;
        let slots = input
            .people
            .checked_add(input.predicates)
            .ok_or_else(|| Error::Message("parser identity size overflow".into()))?;
        let pooling = slots
            .checked_mul(t)
            .ok_or_else(|| Error::Message("parser pooling size overflow".into()))?;
        host_admit(
            "parser positions and identity pooling",
            &[
                (tt, 3 * std::mem::size_of::<f32>()),
                (pooling, std::mem::size_of::<f32>()),
            ],
        )?;
        let ids = g.engine.from_i32(&[t], &input.tokens)?;
        let mut x = g.embedding(w["embedding"], &ids, DType::F32)?;
        let mut distance = Vec::with_capacity(tt * 2);
        let mut same = Vec::with_capacity(tt);
        for a in 0..t {
            for b in 0..t {
                let delta = b as f32 - a as f32;
                let log = delta.abs().ln_1p();
                distance.extend([delta.signum() * log, log]);
                same.push(f32::from(input.ids[a] >= 0 && input.ids[a] == input.ids[b]));
            }
        }
        let distance = g.engine.from_f32(&[t * t, 2], &distance)?;
        let distance = g.leaf(distance, false);
        let same = g.engine.from_f32(&[t * t, 1], &same)?;
        let same = g.leaf(same, false);
        for layer in 0..self.config.layers {
            let p = format!("encoder.{layer}");
            let h = self.norm(g, w, x, &format!("{p}.norm1"))?;
            let qkv = g.matmul(h, w[&format!("{p}.qkv.weight")])?;
            let q = g.slice_columns(qkv, 0, d)?;
            let q = g.bias(q, w[&format!("{p}.query.bias")])?;
            let k = g.slice_columns(qkv, d, d)?;
            let v = g.slice_columns(qkv, 2 * d, d)?;
            let v = g.bias(v, w[&format!("{p}.value.bias")])?;
            let position = g.matmul(distance, w[&format!("{p}.distance")])?;
            let identity = g.matmul(same, w[&format!("{p}.same")])?;
            let bias = g.add(position, identity)?;
            let a = g.biased_attention(q, k, v, bias, t, self.config.heads)?;
            let a = Self::affine(g, w, a, &format!("{p}.out"))?;
            x = g.add(x, a)?;
            let h = self.norm(g, w, x, &format!("{p}.norm2"))?;
            let h = Self::affine(g, w, h, &format!("{p}.ff.0"))?;
            let h = g.gelu(h)?;
            let h = Self::affine(g, w, h, &format!("{p}.ff.1"))?;
            x = g.add(x, h)?;
        }
        x = self.norm(g, w, x, "final_norm")?;
        let global = Self::gather(g, x, &[0])?;
        let slots = input.people + input.predicates;
        if slots == 0 {
            return Ok((global, None, None));
        }
        let mut weights = vec![0.; slots * t];
        for slot in 0..slots {
            let count = input.ids.iter().filter(|i| **i == slot as i32).count();
            if count > 0 {
                for (i, id) in input.ids.iter().enumerate() {
                    if *id == slot as i32 {
                        weights[slot * t + i] = 1. / count as f32;
                    }
                }
            }
        }
        let weights = g.engine.from_f32(&[slots, t], &weights)?;
        let weights = g.leaf(weights, false);
        let pooled = g.matmul(weights, x)?;
        let people = if input.people > 0 {
            Some(Self::gather(
                g,
                pooled,
                &(0..input.people).map(|x| x as i32).collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        let predicates = if input.predicates > 0 {
            Some(Self::gather(
                g,
                pooled,
                &(input.people..slots).map(|x| x as i32).collect::<Vec<_>>(),
            )?)
        } else {
            None
        };
        Ok((global, people, predicates))
    }
    fn root_scores(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        global: Var,
        people: Option<Var>,
        present: &[bool],
    ) -> Result<(Var, Vec<f64>, Vec<f64>)> {
        let n = present.len();
        let classes = n
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Message("query root size overflow".into()))?;
        host_admit(
            "query root distribution",
            &[(
                classes,
                3 * std::mem::size_of::<f64>() + std::mem::size_of::<(Option<Program>, f64)>(),
            )],
        )?;
        let none = Self::affine(g, w, global, "none")?;
        if n == 0 {
            return Ok((none, vec![1.], vec![0.]));
        }
        let persons = Self::gather(
            g,
            people.unwrap(),
            &(0..2 * n).map(|i| (i % n) as i32).collect::<Vec<_>>(),
        )?;
        let goals = Self::gather(
            g,
            w["goal_embedding"],
            &(0..2 * n).map(|i| (i / n) as i32).collect::<Vec<_>>(),
        )?;
        let globals = Self::repeat(g, global, 2 * n)?;
        let feature = g.concat_columns(&[globals, persons, goals])?;
        let h = Self::affine(g, w, feature, "root.0")?;
        let h = g.gelu(h)?;
        let scores = Self::affine(g, w, h, "root.1")?;
        let scores = g.concat_rows(&[scores, none])?;
        let mut values: Vec<_> = g.value(scores)?.into_iter().map(f64::from).collect();
        for i in 0..2 * n {
            if !present[i % n] {
                values[i] = f64::NEG_INFINITY;
            }
        }
        let (log, p) = logsoftmax(&values)?;
        Ok((scores, p, log))
    }
    fn state(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        global: Var,
        people: Var,
        predicates: Option<Var>,
        prefix: &Program,
    ) -> Result<Var> {
        let person = Self::gather(g, people, &[prefix.anchor as i32])?;
        let goal = Self::gather(g, w["goal_embedding"], &[prefix.goal as i32])?;
        let feature = g.concat_columns(&[global, person, goal])?;
        let h = Self::affine(g, w, feature, "initial")?;
        let mut state = g.gelu(h)?;
        for step in &prefix.steps {
            let pred = Self::gather(g, predicates.unwrap(), &[step.predicate as i32])?;
            let op = Self::gather(
                g,
                w["operation_embedding"],
                &[(usize::from(step.reverse) * 2 + usize::from(step.last)) as i32],
            )?;
            let f = g.concat_columns(&[state, pred, op])?;
            let h = Self::affine(g, w, f, "recurrent")?;
            state = g.gelu(h)?;
        }
        Ok(state)
    }
    fn next_scores(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        global: Var,
        people: Var,
        predicates: Option<Var>,
        present: &[bool],
        prefix: &Program,
    ) -> Result<(Var, Vec<f64>, Vec<f64>)> {
        let state = self.state(g, w, global, people, predicates, prefix)?;
        let f = g.concat_columns(&[global, state])?;
        let stop = Self::affine(g, w, f, "stop")?;
        let p = present.len();
        let classes = p
            .checked_mul(4)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Message("query transition size overflow".into()))?;
        host_admit(
            "query transition distribution",
            &[(
                classes,
                3 * std::mem::size_of::<f64>() + std::mem::size_of::<(Option<Step>, f64)>(),
            )],
        )?;
        if p == 0 {
            return Ok((stop, vec![1.], vec![0.]));
        }
        let allpred = Self::gather(
            g,
            predicates.unwrap(),
            &(0..p * 4).map(|i| (i / 4) as i32).collect::<Vec<_>>(),
        )?;
        let ops = Self::gather(
            g,
            w["operation_embedding"],
            &(0..p * 4).map(|i| (i % 4) as i32).collect::<Vec<_>>(),
        )?;
        let globals = Self::repeat(g, global, p * 4)?;
        let states = Self::repeat(g, state, p * 4)?;
        let f = g.concat_columns(&[globals, states, allpred, ops])?;
        let h = Self::affine(g, w, f, "step.0")?;
        let h = g.gelu(h)?;
        let step = Self::affine(g, w, h, "step.1")?;
        let scores = g.concat_rows(&[step, stop])?;
        let mut values: Vec<_> = g.value(scores)?.into_iter().map(f64::from).collect();
        for i in 0..p * 4 {
            if !present[i / 4] {
                values[i] = f64::NEG_INFINITY;
            }
        }
        let (log, probabilities) = logsoftmax(&values)?;
        Ok((scores, probabilities, log))
    }
    fn teacher_query(
        &self,
        g: &mut Graph,
        w: &BTreeMap<String, Var>,
        row: &LocalExample,
        leaves: Vec<Var>,
    ) -> Result<(Evaluation, Vec<Var>)> {
        let input = encoded(&row.text, &row.people, &row.predicates)?;
        ensure(
            row.label == 0,
            "query training label is carried by program; categorical label must be zero",
        )?;
        if let Some(program) = &row.program {
            ensure(
                program.goal < 2
                    && program.anchor < input.people
                    && input.present[program.anchor]
                    && program.steps.len() <= input.occurrences,
                "query target exceeds observed root/input-derived grammar",
            )?;
        }
        let (global, people, predicates) = self.encode(g, w, &input)?;
        let (root, p, lp) =
            self.root_scores(g, w, global, people, &input.present[..input.people])?;
        let n = input.people;
        let selected = row
            .program
            .as_ref()
            .map_or(2 * n, |p| p.goal * n + p.anchor);
        ensure(
            selected < p.len() && lp[selected].is_finite(),
            "query target root outside observed identities",
        )?;
        let mut log_probability = lp[selected];
        let mut nodes = vec![(Some(root), p, selected)];
        if let Some(program) = &row.program {
            let mut prefix = Program {
                goal: program.goal,
                anchor: program.anchor,
                steps: vec![],
            };
            for step in &program.steps {
                ensure(
                    step.predicate < input.predicates && input.present[n + step.predicate],
                    "query target predicate absent",
                )?;
                let (scores, p, lp) = self.next_scores(
                    g,
                    w,
                    global,
                    people.unwrap(),
                    predicates,
                    &input.present[n..],
                    &prefix,
                )?;
                let chosen =
                    step.predicate * 4 + usize::from(step.reverse) * 2 + usize::from(step.last);
                log_probability += lp[chosen];
                nodes.push((Some(scores), p, chosen));
                prefix.steps.push(step.clone());
            }
            if prefix.steps.len() < input.occurrences {
                let (scores, p, lp) = self.next_scores(
                    g,
                    w,
                    global,
                    people.unwrap(),
                    predicates,
                    &input.present[n..],
                    &prefix,
                )?;
                let stop = input.predicates * 4;
                log_probability += lp[stop];
                nodes.push((Some(scores), p, stop));
            } else {
                let mut p = vec![0.; input.predicates * 4 + 1];
                p[input.predicates * 4] = 1.;
                nodes.push((None, p, input.predicates * 4));
            }
        }
        Ok((
            Evaluation {
                log_probabilities: vec![log_probability],
                output: Output::Sequence(nodes),
            },
            leaves,
        ))
    }
    pub fn query_plan(
        &self,
        e: &mut Engine,
        text: &str,
        people: &[String],
        predicates: &[String],
    ) -> Result<QueryPlan> {
        ensure(
            self.metadata.role == Role::Query,
            "query decoder role required",
        )?;
        let input = encoded(text, people, predicates)?;
        let mut g = Graph::new(e, false);
        let (w, _) = self.bind(&mut g, false);
        let (global, persons, preds) = self.encode(&mut g, &w, &input)?;
        let (_, law, _) =
            self.root_scores(&mut g, &w, global, persons, &input.present[..input.people])?;
        let mut roots = vec![];
        for (goal_anchor, p) in law.iter().take(2 * input.people).enumerate() {
            if *p > 0. {
                roots.push((
                    Some(Program {
                        goal: goal_anchor / input.people,
                        anchor: goal_anchor % input.people,
                        steps: vec![],
                    }),
                    *p,
                ));
            }
        }
        roots.push((None, law[2 * input.people]));
        let zero = g.engine.full(&[1, self.config.width], DType::F32, 0.)?;
        let context = QueryFeatures {
            global: g.tensor(global).clone(),
            people: persons.map_or_else(|| zero.clone(), |v| g.tensor(v).clone()),
            predicates: preds.map_or_else(|| zero.clone(), |v| g.tensor(v).clone()),
            present: input.present.clone(),
            count: input.people,
            occurrences: input.occurrences,
            engine: g.engine.context_id(),
        };
        Ok(QueryPlan {
            roots,
            predicates: (0..input.predicates)
                .filter(|i| input.present[input.people + *i])
                .collect(),
            context,
        })
    }
    pub fn query_expand(
        &self,
        e: &mut Engine,
        plan: &QueryPlan,
        prefix: &Program,
    ) -> Result<Vec<(Option<Step>, f64)>> {
        let c = &plan.context;
        ensure(
            c.engine == e.context_id()
                && prefix.goal < 2
                && prefix.anchor < c.count
                && c.present[prefix.anchor]
                && prefix.steps.len() <= c.occurrences,
            "query plan context/root mismatch",
        )?;
        if prefix.steps.len() >= c.occurrences {
            return Ok(vec![(None, 1.)]);
        }
        let p = c.present.len() - c.count;
        ensure(
            prefix
                .steps
                .iter()
                .all(|s| s.predicate < p && c.present[c.count + s.predicate]),
            "query prefix refers to unobserved predicate",
        )?;
        let mut g = Graph::new(e, false);
        let (w, _) = self.bind(&mut g, false);
        let global = g.leaf(c.global.clone(), false);
        let people = g.leaf(c.people.clone(), false);
        let predicates = if p > 0 {
            Some(g.leaf(c.predicates.clone(), false))
        } else {
            None
        };
        let (_, law, _) = self.next_scores(
            &mut g,
            &w,
            global,
            people,
            predicates,
            &c.present[c.count..],
            prefix,
        )?;
        let mut actions = vec![];
        for (i, p) in law.iter().take(4 * p).enumerate() {
            if *p > 0. {
                actions.push((
                    Some(Step {
                        predicate: i / 4,
                        reverse: i % 4 / 2 == 1,
                        last: i % 2 == 1,
                    }),
                    *p,
                ));
            }
        }
        actions.push((None, law[4 * p]));
        Ok(actions)
    }
    pub fn law(
        &self,
        e: &mut Engine,
        text: &str,
        people: &[String],
        predicates: &[String],
    ) -> Result<Vec<f64>> {
        ensure(
            self.metadata.role != Role::Query,
            "query distributions use lazy query_plan/query_expand",
        )?;
        let row = LocalExample {
            text: text.into(),
            people: people.to_vec(),
            predicates: predicates.to_vec(),
            label: 0,
            program: None,
        };
        let mut g = Graph::new(e, false);
        let (v, _) = self.evaluate(&mut g, &row, false)?;
        Ok(v.log_probabilities.iter().map(|x| x.exp()).collect())
    }
    pub fn signature(&self, e: &mut Engine) -> Result<String> {
        if let Some(encoder) = &self.encoder {
            encoder.verify(e)?;
        }
        signature(
            e,
            &self.parameters,
            &(
                &self.metadata,
                &self.config,
                FORMAT,
                self.encoder_identity(),
            ),
        )
    }
    pub fn save(&self, e: &mut Engine, path: &Path) -> Result<()> {
        atomic_directory(path, |dir| {
            save_tensors(
                e,
                &dir.join("head.safetensors"),
                self.parameters.iter().map(|p| (p.name.as_str(), &p.value)),
            )?;
            if let Some(encoder) = &self.encoder {
                encoder.save(e, &dir.join("encoder"))?;
            }
            std::fs::write(
                dir.join("role.json"),
                json::to_vec_pretty(&Stored {
                    format: FORMAT.into(),
                    metadata: self.metadata.clone(),
                    config: self.config.clone(),
                    encoder_digest: self.encoder_identity().map(|x| x.digest.clone()),
                })?,
            )?;
            std::fs::write(dir.join("signature"), self.signature(e)?)?;
            Ok(())
        })
    }
    pub fn load(e: &mut Engine, path: &Path) -> Result<Self> {
        Self::load_format(e, path, FORMAT)
    }
    /// Import an authenticated, shared semantic head into the dynamic coordinate implementation.
    pub fn migrate_semantic(e: &mut Engine, path: &Path) -> Result<Self> {
        let stored: Stored = json::from_slice(&std::fs::read(path.join("role.json"))?)?;
        ensure(
            matches!(
                stored.metadata.role,
                Role::Scene | Role::Instruction | Role::Request
            ),
            "semantic head role required for factor-v2 migration",
        )?;
        Self::load_format(e, path, "sejong-role-factor-v2")
    }
    fn load_format(e: &mut Engine, path: &Path, format: &str) -> Result<Self> {
        let stored: Stored = json::from_slice(&std::fs::read(path.join("role.json"))?)?;
        ensure(
            stored.format == format,
            "role artifact architecture version differs",
        )?;
        let encoder = if stored.encoder_digest.is_some() {
            Some(BertEncoder::load_shared(e, &path.join("encoder"))?)
        } else {
            None
        };
        ensure(
            encoder.as_ref().map(|x| x.identity.digest.clone()) == stored.encoder_digest,
            "encoder identity differs",
        )?;
        let mut tensors = load_tensors(e, &path.join("head.safetensors"))?;
        let input = encoder.as_ref().map_or(0, |e| e.config.hidden_size);
        let mut parameters = vec![];
        for (name, shape, _) in layout(stored.metadata.role, &stored.config, input) {
            let value = tensors
                .remove(&name)
                .ok_or_else(|| Error::Message(format!("missing {name}")))?;
            ensure(
                value.shape == shape && value.dtype == DType::F32,
                "role parameter shape differs",
            )?;
            parameters.push(Parameter { name, value });
        }
        ensure(tensors.is_empty(), "unexpected role parameter")?;
        let model = Self {
            metadata: stored.metadata,
            config: stored.config,
            parameters,
            encoder,
        };
        ensure(
            signature(
                e,
                &model.parameters,
                &(
                    &model.metadata,
                    &model.config,
                    format,
                    model.encoder_identity(),
                ),
            )? == std::fs::read_to_string(path.join("signature"))?,
            "role signature differs",
        )?;
        Ok(model)
    }
}
