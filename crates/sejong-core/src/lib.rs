//! Configuration, tensor types and artifact contracts shared by Sejong.
pub mod native;
pub mod numeric;
use serde::{Deserialize, Serialize};
pub use serde_json as json;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("{operation}: required {required} bytes, available {available} bytes")]
    Capacity {
        operation: String,
        required: usize,
        available: usize,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn ensure(condition: bool, message: impl std::fmt::Display) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Message(message.to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F64,
    F16,
    I32,
}
impl DType {
    pub fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F64 => 8,
            _ => 4,
        }
    }
    pub fn pointer(self) -> &'static str {
        match self {
            Self::F32 => "*fp32",
            Self::F64 => "*fp64",
            Self::F16 => "*fp16",
            Self::I32 => "*i32",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn_width: usize,
    pub context: usize,
    pub rope_theta: f32,
    pub norm_epsilon: f32,
    pub precision: DType,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hierarchy: Option<HierarchyConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts: Option<ExpertConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HierarchyConfig {
    pub patch_size: usize,
    pub encoder_width: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    pub encoder_ffn_width: usize,
    pub decoder_width: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ffn_width: usize,
    pub local_window: usize,
    #[serde(default)]
    pub checkpoint_encoder: bool,
    #[serde(default)]
    pub checkpoint_decoder: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertConfig {
    pub count: usize,
    pub top_k: usize,
    pub ffn_width: usize,
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            id: "sejong".into(),
            width: 512,
            layers: 8,
            heads: 8,
            ffn_width: 1536,
            context: 2048,
            rope_theta: 10000.0,
            norm_epsilon: f32::EPSILON,
            precision: DType::F16,
            seed: 42,
            hierarchy: None,
            kv_heads: None,
            experts: None,
        }
    }
}
impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(!self.id.is_empty(), "model.id must contain a name")?;
        ensure(
            self.width > 0 && self.heads > 0 && self.width.is_multiple_of(self.heads),
            "width must be divisible by heads",
        )?;
        ensure(
            (self.width / self.heads).is_multiple_of(2),
            "RoPE requires an even head width",
        )?;
        ensure(
            self.layers > 0 && self.ffn_width > 0 && self.context > 0,
            "layers, ffn_width and context must be positive",
        )?;
        ensure(
            matches!(self.precision, DType::F32 | DType::F16),
            "model precision must be f32 or f16",
        )?;
        ensure(
            self.norm_epsilon.is_finite()
                && self.norm_epsilon > 0.0
                && self.rope_theta.is_finite()
                && self.rope_theta > 0.0,
            "norm_epsilon and rope_theta must be finite and positive",
        )?;
        let kv = self.kv_heads.unwrap_or(self.heads);
        ensure(
            kv > 0 && self.heads.is_multiple_of(kv),
            "kv_heads must divide heads",
        )?;
        if let Some(e) = &self.experts {
            ensure(
                e.count > 0 && e.top_k > 0 && e.top_k <= e.count && e.ffn_width > 0,
                "expert count/width must be positive and top_k must be in 1..=count",
            )?;
        }
        if let Some(h) = &self.hierarchy {
            ensure(
                h.patch_size > 0 && h.patch_size <= self.context && h.local_window > 0,
                "patch_size must be in 1..=context and local_window must be positive",
            )?;
            for (width, layers, heads, ffn) in [
                (
                    h.encoder_width,
                    h.encoder_layers,
                    h.encoder_heads,
                    h.encoder_ffn_width,
                ),
                (
                    h.decoder_width,
                    h.decoder_layers,
                    h.decoder_heads,
                    h.decoder_ffn_width,
                ),
            ] {
                ensure(
                    width > 0
                        && layers > 0
                        && heads > 0
                        && ffn > 0
                        && width.is_multiple_of(heads)
                        && (width / heads).is_multiple_of(2),
                    "hierarchy stages require positive dimensions and an even head width",
                )?;
            }
        }
        self.parameter_tensor_groups()?
            .iter()
            .try_fold(0usize, |total, (elements, count)| {
                elements
                    .checked_mul(*count)
                    .and_then(|n| total.checked_add(n))
                    .ok_or_else(|| {
                        Error::Message("model parameter count exceeds addressable range".into())
                    })
            })?;
        Ok(())
    }
    /// 같은 원소 수의 파라미터 텐서를 묶어 초기화 전 저장 용량을 계산한다.
    pub fn parameter_tensor_groups(&self) -> Result<Vec<(usize, usize)>> {
        fn matrix(
            groups: &mut Vec<(usize, usize)>,
            a: usize,
            b: usize,
            count: usize,
        ) -> Result<()> {
            let elements = a
                .checked_mul(b)
                .ok_or_else(|| Error::Message("model tensor shape overflow".into()))?;
            ensure(
                elements > 0 && elements <= i32::MAX as usize && count > 0,
                "model parameter tensors require 1..=i32::MAX elements",
            )?;
            groups.push((elements, count));
            Ok(())
        }
        fn repeated(layers: usize, count: usize) -> Result<usize> {
            layers
                .checked_mul(count)
                .ok_or_else(|| Error::Message("model tensor count overflow".into()))
        }
        fn stage(
            groups: &mut Vec<(usize, usize)>,
            width: usize,
            layers: usize,
            heads: usize,
            kv_heads: usize,
            ffn: usize,
            experts: Option<&ExpertConfig>,
        ) -> Result<()> {
            ensure(
                heads > 0
                    && kv_heads > 0
                    && width.is_multiple_of(heads)
                    && heads.is_multiple_of(kv_heads),
                "stage head dimensions mismatch",
            )?;
            let kv = width / heads * kv_heads;
            matrix(groups, width, 1, repeated(layers, 2)?)?;
            matrix(groups, width, width, repeated(layers, 2)?)?;
            matrix(groups, width, kv, repeated(layers, 2)?)?;
            if let Some(e) = experts {
                matrix(
                    groups,
                    width,
                    e.ffn_width,
                    repeated(repeated(layers, e.count)?, 3)?,
                )?;
                matrix(groups, width, e.count, layers)?;
            } else {
                matrix(groups, width, ffn, repeated(layers, 3)?)?;
            }
            Ok(())
        }
        let mut groups = Vec::new();
        stage(
            &mut groups,
            self.width,
            self.layers,
            self.heads,
            self.kv_heads.unwrap_or(self.heads),
            self.ffn_width,
            self.experts.as_ref(),
        )?;
        matrix(&mut groups, self.width, 1, 1)?;
        matrix(&mut groups, self.width, VOCAB, 1)?;
        if let Some(h) = &self.hierarchy {
            stage(
                &mut groups,
                h.encoder_width,
                h.encoder_layers,
                h.encoder_heads,
                h.encoder_heads,
                h.encoder_ffn_width,
                None,
            )?;
            stage(
                &mut groups,
                h.decoder_width,
                h.decoder_layers,
                h.decoder_heads,
                h.decoder_heads,
                h.decoder_ffn_width,
                None,
            )?;
            matrix(&mut groups, VOCAB, h.encoder_width, 1)?;
            matrix(&mut groups, h.encoder_width, 1, 1)?;
            matrix(&mut groups, h.decoder_width, 1, 1)?;
            matrix(&mut groups, self.width, 1, 2)?;
            matrix(&mut groups, self.width, h.encoder_width, 2)?;
            matrix(&mut groups, self.width, h.decoder_width, 2)?;
            matrix(&mut groups, h.encoder_width, h.decoder_width, 1)?;
        } else {
            matrix(&mut groups, VOCAB, self.width, 1)?;
        }
        Ok(groups)
    }
}
pub const VOCAB: usize = 257;
pub const EOS: i32 = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainConfig {
    pub optimizer: OptimizerKind,
    pub control_variate: Option<ControlVariateConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_vjp: Option<ParameterVjpConfig>,
    pub gradient_diagnostics_every: usize,
    pub activation_checkpointing: bool,
    pub steps: usize,
    pub sequence: usize,
    pub batch: usize,
    pub accumulation: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub warmup_steps: usize,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub clip_norm: f32,
    pub clip_enabled: bool,
    pub loss_scale: f32,
    pub checkpoint_every: usize,
    pub log_every: usize,
    pub seed: u64,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerKind::Adamw,
            control_variate: None,
            parameter_vjp: None,
            gradient_diagnostics_every: 0,
            activation_checkpointing: true,
            steps: 10000,
            sequence: 512,
            batch: 1,
            accumulation: 8,
            learning_rate: 3e-4,
            min_learning_rate: 3e-5,
            warmup_steps: 100,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 0.,
            weight_decay: 0.1,
            clip_norm: 1.0,
            clip_enabled: true,
            loss_scale: 1.0,
            checkpoint_every: 500,
            log_every: 10,
            seed: 123,
        }
    }
}
impl TrainConfig {
    pub fn validate(&self, model: &ModelConfig) -> Result<()> {
        ensure(
            self.steps > 0
                && self.sequence > 0
                && self.sequence <= model.context
                && self.batch > 0
                && self.accumulation > 0,
            "training requires positive steps/batch/accumulation and sequence within context",
        )?;
        ensure(self.log_every > 0, "log_every must be positive")?;
        if let Some(policy) = &self.parameter_vjp {
            ensure(
                self.optimizer == OptimizerKind::Sgd,
                "parameter_vjp uses the compact SGD optimizer",
            )?;
            policy.validate()?;
        }
        if self.optimizer == OptimizerKind::ControlVariateSgd {
            if let Some(hierarchy) = &model.hierarchy {
                ensure(
                    self.sequence > hierarchy.patch_size,
                    "control-variate training sequence must exceed patch_size to observe global gradients",
                )?;
            }
            self.control_variate
                .as_ref()
                .ok_or_else(|| {
                    Error::Message(
                        "control_variate_sgd requires an explicit training.control_variate budget"
                            .into(),
                    )
                })?
                .validate(self.accumulation)?;
        }
        ensure(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.min_learning_rate.is_finite()
                && self.min_learning_rate >= 0.0
                && self.min_learning_rate <= self.learning_rate,
            "learning rates must be finite, with 0 <= min_learning_rate <= learning_rate",
        )?;
        ensure(
            (0.0..1.0).contains(&self.beta1) && (0.0..1.0).contains(&self.beta2),
            "Adam beta values must be in [0, 1)",
        )?;
        ensure(
            self.epsilon.is_finite()
                && self.epsilon >= 0.0
                && self.loss_scale.is_finite()
                && self.loss_scale > 0.0
                && self.loss_scale.recip().is_finite()
                && (!self.clip_enabled || (self.clip_norm.is_finite() && self.clip_norm > 0.0))
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.0,
            "optimizer requires nonnegative epsilon/weight decay, positive loss scale and positive enabled clip norm",
        )
    }
    pub fn lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.learning_rate * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            let span = self.steps.saturating_sub(self.warmup_steps).max(1);
            let phase = (step.saturating_sub(self.warmup_steps) as f32 / span as f32).min(1.0);
            self.min_learning_rate
                + 0.5
                    * (self.learning_rate - self.min_learning_rate)
                    * (1.0 + (std::f32::consts::PI * phase).cos())
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    Adamw,
    Sgd,
    ControlVariateSgd,
}

/// 현재 순전파의 token 기저와 정확한 feature 잔차 표집의 실행 예산.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterVjpConfig {
    pub rank: usize,
    pub residual_fraction: f64,
    pub block_rows: usize,
    pub seed: u64,
}
impl ParameterVjpConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.rank > 0
                && self.rank <= i32::MAX as usize
                && self.block_rows > 0
                && self.block_rows <= i32::MAX as usize
                && self.residual_fraction.is_finite()
                && self.residual_fraction > 0.
                && self.residual_fraction <= 1.,
            "parameter VJP requires positive rank/block_rows and residual_fraction in (0,1]",
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlSampling {
    Uniform,
    Stratified,
}

/// 제어 배치와 저랭크 예측기의 학습 예산을 지정한다.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlVariateConfig {
    pub controls: usize,
    pub rank: usize,
    pub sampling: ControlSampling,
    pub strata: usize,
    pub basis_refresh: usize,
    pub predictor_learning_rate: f32,
}
impl ControlVariateConfig {
    pub fn validate(&self, observations: usize) -> Result<()> {
        ensure(
            self.controls > 0 && self.controls <= observations,
            "control_variate.controls must be in 1..=accumulation",
        )?;
        ensure(
            self.rank > 0 && self.basis_refresh > 0,
            "control_variate.rank and basis_refresh must be positive",
        )?;
        ensure(
            self.predictor_learning_rate.is_finite() && self.predictor_learning_rate > 0.,
            "control_variate.predictor_learning_rate must be finite and positive",
        )?;
        if self.sampling == ControlSampling::Stratified {
            ensure(
                self.strata > 0 && self.strata <= self.controls && self.strata <= observations,
                "stratified sampling requires 1 <= strata <= controls <= accumulation",
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    pub device: i32,
    pub architecture: String,
    pub wavefront: usize,
    pub memory_mb: usize,
}
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device: 0,
            architecture: "gfx1032".into(),
            wavefront: 32,
            memory_mb: 0,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainConfig,
    pub gpu: GpuConfig,
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        c.model.validate()?;
        c.training.validate(&c.model)?;
        ensure(
            [32, 64].contains(&c.gpu.wavefront),
            "GPU wavefront must be 32 or 64",
        )?;
        Ok(c)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub model: ModelConfig,
    pub step: usize,
    pub weights: String,
    pub optimizer: Option<String>,
    pub training: Option<TrainConfig>,
    pub data_fingerprint: Option<String>,
    pub rng_state: Option<u64>,
}
impl Manifest {
    pub fn load(directory: &Path) -> Result<Self> {
        let m: Self = json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
        ensure(
            m.format == 1,
            format_args!("unsupported artifact format {}", m.format),
        )?;
        m.model.validate()?;
        for filename in std::iter::once(&m.weights).chain(m.optimizer.iter()) {
            ensure(
                Path::new(filename).components().count() == 1
                    && Path::new(filename)
                        .file_name()
                        .is_some_and(|v| v == filename.as_str()),
                "artifact file names must be local basenames",
            )?;
        }
        Ok(m)
    }
}
pub fn root() -> PathBuf {
    std::env::var_os("SEJONG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
}

/// Portable seeded generator used for initialization and checkpointable sampling.
#[derive(Clone, Debug)]
pub struct Rng {
    pub state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        (((self.next_u64() >> 40) as f64 + 0.5) / 16777216.0) as f32
    }
    pub fn index(&mut self, size: usize) -> usize {
        let n = size as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return (x % n) as usize;
            }
        }
    }
    pub fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}
