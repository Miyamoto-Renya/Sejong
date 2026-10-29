//! Versioned finite semantic and direct-memory contracts.
use crate::{ModelConfig, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub const FORMAT: &str = "sejong-native-composite-v6";
pub use bytes::ByteValue;
mod bytes;
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Scene,
    Instruction,
    Request,
    Query,
    Lexical,
}
impl Role {
    pub fn classes_for(self, n: usize, meanings: usize) -> Result<usize> {
        let overflow = || crate::Error::Message("operator inventory size overflow".into());
        match self {
            Self::Scene => n
                .checked_mul(n)
                .and_then(|x| x.checked_add(1))
                .ok_or_else(overflow),
            Self::Instruction => n
                .checked_add(1)
                .and_then(|x| x.checked_mul(x))
                .ok_or_else(overflow),
            Self::Request => n
                .checked_add(3)
                .and_then(|x| x.checked_mul(2))
                .and_then(|x| x.checked_add(1))
                .ok_or_else(overflow),
            Self::Lexical => Ok(meanings),
            Self::Query => Err(crate::Error::Message(
                "query uses conditional program decoding".into(),
            )),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalExample {
    pub text: String,
    #[serde(default)]
    pub people: Vec<String>,
    #[serde(default)]
    pub predicates: Vec<String>,
    pub label: usize,
    #[serde(default)]
    pub program: Option<Program>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolePair {
    pub left: LocalExample,
    pub right: LocalExample,
    pub equivalent: bool,
}
fn default_batch() -> usize {
    0
}
fn default_checkpoint() -> usize {
    0
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleTraining {
    #[serde(default)]
    pub loss: LossPolicy,
    #[serde(default)]
    pub optimizer: OptimizerPolicy,
    #[serde(default)]
    pub batch: BatchPolicy,
    pub role: Role,
    pub model: ModelConfig,
    pub steps: usize,
    pub learning_rate: f32,
    pub examples: Vec<LocalExample>,
    #[serde(default)]
    pub phrases: Vec<String>,
    #[serde(default)]
    pub meanings: Vec<String>,
    #[serde(default)]
    pub encoder: Option<PathBuf>,
    #[serde(default)]
    pub pairs: Vec<RolePair>,
    #[serde(default)]
    pub resume: Option<PathBuf>,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_every: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    pub id: String,
    pub dimension: usize,
    pub tokens: usize,
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub ffn_width: usize,
    pub norm_epsilon: f32,
    pub readout_temperature: f64,
    pub residual_scale: f32,
    pub seed: u64,
}
impl MemoryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            !self.id.is_empty()
                && self.dimension > 0
                && self.heads > 0
                && self.dimension.is_multiple_of(self.heads)
                && self.tokens > 0
                && self.layers > 0
                && self.hidden > 0
                && self.ffn_width > 0
                && self.norm_epsilon.is_finite()
                && self.norm_epsilon > 0.
                && self.readout_temperature.is_finite()
                && self.readout_temperature > 0.
                && self.residual_scale.is_finite(),
            "memory requires a named, positive, head-aligned configuration",
        )
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryTraining {
    #[serde(default)]
    pub query_loss: LossPolicy,
    #[serde(default)]
    pub background: BackgroundPolicy,
    #[serde(default)]
    pub optimizer: OptimizerPolicy,
    #[serde(default)]
    pub batch: BatchPolicy,
    pub query_learning_rate: f32,
    pub min_query_learning_rate: f32,
    pub config: MemoryConfig,
    pub requests: Vec<ByteValue>,
    pub values: Vec<ByteValue>,
    pub contexts: Vec<Vec<f64>>,
    pub target_queries: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub value_targets: Vec<Vec<f64>>,
    #[serde(default)]
    pub resume: Option<PathBuf>,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_every: usize,
    pub query_steps: usize,
    pub fact_steps: usize,
    pub learning_rate: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionTraining {
    #[serde(default)]
    pub optimizer: OptimizerPolicy,
    #[serde(default)]
    pub batch: BatchPolicy,
    pub labels: Vec<usize>,
    #[serde(default)]
    pub resume: Option<PathBuf>,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_every: usize,
    pub steps: usize,
    pub learning_rate: f32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Budgets {
    pub memory_bytes: Option<usize>,
}
impl Budgets {
    pub fn resolve(&self) -> Result<usize> {
        let available = host_available_bytes()?;
        Ok(self.memory_bytes.map_or(available, |b| b.min(available)))
    }
}
pub fn host_available_bytes() -> Result<usize> {
    let info = std::fs::read_to_string("/proc/meminfo")?;
    let available = info
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| crate::Error::Message("host memory availability required".into()))?;
    let mut limit = available;
    if let (Ok(max), Ok(current)) = (
        std::fs::read_to_string("/sys/fs/cgroup/memory.max"),
        std::fs::read_to_string("/sys/fs/cgroup/memory.current"),
    ) {
        if let (Ok(max), Ok(current)) =
            (max.trim().parse::<usize>(), current.trim().parse::<usize>())
        {
            limit = limit.min(max.saturating_sub(current));
        }
    }
    ensure(limit > 0, "available host memory exhausted")?;
    Ok(limit)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FactSource {
    Memory { path: PathBuf },
    Native { path: PathBuf },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assembly {
    pub scene: PathBuf,
    pub instruction: PathBuf,
    pub request: PathBuf,
    pub query: PathBuf,
    #[serde(default)]
    pub lexica: Vec<PathBuf>,
    pub a: FactSource,
    pub b: FactSource,
    #[serde(default)]
    pub extra: BTreeMap<String, FactSource>,
    #[serde(default)]
    pub goal_routes: Vec<Vec<String>>,
    pub catalog: Vec<String>,
    #[serde(default)]
    pub budgets: Budgets,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    Full,
    Direct,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Query {
    pub question: String,
    pub story: Vec<String>,
    pub people: Vec<String>,
    pub predicates: Vec<String>,
    pub joint: Vec<String>,
    pub goal: usize,
    pub modes: BTreeMap<String, ReadMode>,
    pub max_rows: Option<usize>,
    pub max_components: Option<usize>,
    pub full: bool,
    pub tolerance: Option<f64>,
}
impl Default for Query {
    fn default() -> Self {
        Self {
            question: String::new(),
            story: vec![],
            people: vec![],
            predicates: vec![],
            joint: vec![],
            goal: 1,
            modes: BTreeMap::new(),
            max_rows: None,
            max_components: None,
            full: false,
            tolerance: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Answer {
    pub values: Vec<ByteValue>,
    pub probabilities: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub remaining: f64,
    pub read_error: f64,
    #[serde(default)]
    pub numerical_error: f64,
    pub unresolved: f64,
    pub certified: bool,
    pub prediction: Option<usize>,
    pub output: Option<ByteValue>,
    pub reason: String,
    pub full_rows: usize,
    pub direct_rows: usize,
    pub components: usize,
    pub event_nodes: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointAnswer {
    pub values: Vec<ByteValue>,
    pub shape: Vec<usize>,
    pub probabilities: Vec<f64>,
    pub read_error: f64,
    #[serde(default)]
    pub numerical_error: f64,
    pub full_rows: usize,
    pub direct_rows: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Step {
    pub predicate: usize,
    pub reverse: bool,
    pub last: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Program {
    pub goal: usize,
    pub anchor: usize,
    pub steps: Vec<Step>,
}
pub fn inventory(xs: &[String], name: &str) -> Result<()> {
    let set: std::collections::BTreeSet<_> = xs.iter().collect();
    ensure(
        !xs.is_empty() && set.len() == xs.len() && xs.iter().all(|x| !x.is_empty()),
        format!("{name} requires unique nonempty entries"),
    )
}
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    ensure(
        !logits.is_empty()
            && logits
                .iter()
                .all(|x| x.is_finite() || *x == f64::NEG_INFINITY),
        "invalid logits",
    )?;
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ensure(max.is_finite(), "at least one allowed outcome required")?;
    let mut p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = p.iter().sum();
    for x in &mut p {
        *x /= sum;
    }
    Ok(p)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueTraining {
    #[serde(default)]
    pub optimizer: OptimizerPolicy,
    #[serde(default)]
    pub batch: BatchPolicy,
    pub model: ModelConfig,
    pub requests: Vec<ByteValue>,
    pub prompts: Vec<ByteValue>,
    pub values: Vec<ByteValue>,
    pub labels: Vec<usize>,
    pub steps: usize,
    pub learning_rate: f32,
    #[serde(default)]
    pub resume: Option<PathBuf>,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_every: usize,
}
pub fn inventory_bytes(xs: &[ByteValue], name: &str) -> Result<()> {
    let set: std::collections::BTreeSet<_> = xs.iter().collect();
    ensure(
        !xs.is_empty() && set.len() == xs.len() && xs.iter().all(|x| !x.is_empty()),
        format!("{name} requires unique nonempty byte entries"),
    )
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightRequest {
    pub inputs: Vec<Vec<f64>>,
    pub route: Vec<String>,
    #[serde(default)]
    pub modes: Vec<ReadMode>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LossPolicy {
    pub weights: BTreeMap<String, f64>,
}
impl LossPolicy {
    pub fn normalized(&self, names: &[&str]) -> Result<Vec<f64>> {
        ensure(!names.is_empty(), "loss components required")?;
        let raw: Vec<_> = names
            .iter()
            .map(|name| {
                self.weights
                    .get(*name)
                    .copied()
                    .unwrap_or(if self.weights.is_empty() { 1. } else { 0. })
            })
            .collect();
        ensure(
            raw.iter().all(|x| x.is_finite() && *x >= 0.),
            "loss weights must be finite and nonnegative",
        )?;
        let total: f64 = raw.iter().sum();
        ensure(total > 0., "active loss weights sum to zero")?;
        Ok(raw.into_iter().map(|x| x / total).collect())
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BackgroundPolicy {
    #[default]
    Empirical,
    Gaussian {
        standard_deviation: f32,
        zero_probability: f32,
    },
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BatchPolicy {
    pub max_examples: Option<usize>,
    pub memory_bytes: Option<usize>,
}
impl BatchPolicy {
    pub fn resolve(
        &self,
        available: usize,
        per_example: usize,
        observations: usize,
    ) -> Result<usize> {
        ensure(
            per_example > 0 && observations > 0,
            "positive batch cost and observations required",
        )?;
        let bytes = self.memory_bytes.map_or(available, |b| b.min(available));
        let count = (bytes / per_example)
            .min(observations)
            .min(self.max_examples.unwrap_or(observations));
        if count == 0 {
            return Err(crate::Error::Capacity {
                operation: "training batch".into(),
                required: per_example,
                available: bytes,
            });
        }
        Ok(count)
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OptimizerPolicy {
    pub momentum_examples: Option<usize>,
    pub variance_examples: Option<usize>,
    pub epsilon: Option<f32>,
    pub weight_decay: f32,
    pub clip_norm: Option<f32>,
}
impl OptimizerPolicy {
    pub fn configuration(
        &self,
        steps: usize,
        lr: f32,
        batch: usize,
        observations: usize,
    ) -> Result<crate::TrainConfig> {
        ensure(
            steps > 0 && batch > 0 && observations > 0 && lr.is_finite() && lr > 0.,
            "positive optimizer dimensions and learning rate required",
        )?;
        let effective = |examples: usize| -> Result<f32> {
            ensure(examples > 0, "optimizer memory horizon must be positive")?;
            let h = examples.div_ceil(batch) as f64;
            Ok((((h - 1.) / (h + 1.)) as f32).min(f32::from_bits(1f32.to_bits() - 1)))
        };
        let planned = steps
            .checked_mul(batch)
            .ok_or_else(|| crate::Error::Message("optimizer horizon overflow".into()))?;
        let beta1 = effective(self.momentum_examples.unwrap_or(observations))?;
        let beta2 = effective(self.variance_examples.unwrap_or(planned))?;
        let epsilon = self.epsilon.unwrap_or(0.);
        ensure(
            epsilon.is_finite()
                && epsilon >= 0.
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.,
            "optimizer scale parameters invalid",
        )?;
        if let Some(clip) = self.clip_norm {
            ensure(
                clip.is_finite() && clip > 0.,
                "gradient clipping norm must be positive",
            )?;
        }
        Ok(crate::TrainConfig {
            steps,
            learning_rate: lr,
            min_learning_rate: lr,
            warmup_steps: 0,
            beta1,
            beta2,
            epsilon,
            weight_decay: self.weight_decay,
            clip_enabled: self.clip_norm.is_some(),
            clip_norm: self.clip_norm.unwrap_or(0.),
            loss_scale: 1.,
            accumulation: 1,
            ..Default::default()
        })
    }
}
