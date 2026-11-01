//! Memory-mapped byte corpora, AdamW and resumable Transformer training.
pub mod control_variate;
pub mod memory;
pub mod native;
pub mod roles;
use memmap2::Mmap;
use sejong_core::{Config, DType, EOS, Error, Manifest, Result, Rng, TrainConfig, ensure, json};
use sejong_gpu::{Engine, Tensor, graph::Graph};
use sejong_model::{Model, load_tensors};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

pub struct Corpus {
    documents: Vec<Mmap>,
    ends: Vec<usize>,
    pub tokens: usize,
    pub fingerprint: String,
}
impl Corpus {
    pub fn open(path: &Path) -> Result<Self> {
        let mut files = vec![];
        collect_text(path, &mut files)?;
        files.sort();
        ensure(!files.is_empty(), "corpus requires at least one text file")?;
        let mut documents = vec![];
        let mut ends = vec![];
        let mut tokens = 0usize;
        let mut hash = blake3::Hasher::new();
        hash.update(b"Sejong byte corpus v1: EOS + document bytes + EOS");
        for file in files {
            let f = std::fs::File::open(&file)?;
            if f.metadata()?.len() == 0 {
                continue;
            }
            let map = unsafe { Mmap::map(&f)? };
            hash.update(&(map.len() as u64).to_le_bytes());
            hash.update(&map);
            tokens = tokens
                .checked_add(map.len() + 1)
                .ok_or_else(|| Error::Message("corpus size overflow".into()))?;
            ends.push(tokens);
            documents.push(map);
        }
        ensure(tokens > 1, "corpus contains no text")?;
        tokens = tokens
            .checked_add(1)
            .ok_or_else(|| Error::Message("corpus size overflow".into()))?;
        Ok(Self {
            documents,
            ends,
            tokens,
            fingerprint: hash.finalize().to_hex().to_string(),
        })
    }
    fn token(&self, index: usize) -> i32 {
        if index == self.tokens - 1 {
            return EOS;
        }
        let doc = self.ends.partition_point(|end| *end <= index);
        let start = if doc == 0 { 0 } else { self.ends[doc - 1] };
        let local = index - start;
        if local == 0 {
            EOS
        } else {
            self.documents[doc][local - 1] as i32
        }
    }
    pub fn batch(
        &self,
        rng: &mut Rng,
        batch: usize,
        sequence: usize,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut input = Vec::with_capacity(batch * sequence);
        let mut target = Vec::with_capacity(batch * sequence);
        for _ in 0..batch {
            let start = rng.index(self.tokens - sequence);
            for i in 0..sequence {
                input.push(self.token(start + i));
                target.push(self.token(start + i + 1));
            }
        }
        Ok((input, target))
    }
}
fn collect_text(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if path.is_dir() {
                collect_text(&path, files)?;
            } else if path.extension().is_some_and(|ext| ext == "txt") {
                files.push(path);
            }
        }
    } else {
        return Err(Error::Message(format!(
            "corpus path does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

pub struct Adam {
    pub step: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    gradients: Vec<Option<Tensor>>,
}
impl Adam {
    pub fn new(engine: &mut Engine, parameters: &[sejong_model::Parameter]) -> Result<Self> {
        let mut m = vec![];
        let mut v = vec![];
        for p in parameters {
            m.push(engine.full(&p.value.shape, DType::F32, 0.)?);
            v.push(engine.full(&p.value.shape, DType::F32, 0.)?);
        }
        Ok(Self {
            step: 0,
            m,
            v,
            gradients: vec![None; parameters.len()],
        })
    }
    pub fn load(
        engine: &mut Engine,
        model: &Model,
        directory: &Path,
        manifest: &Manifest,
    ) -> Result<Self> {
        let name = manifest
            .optimizer
            .as_ref()
            .ok_or_else(|| Error::Message("resume requires an optimizer checkpoint".into()))?;
        Self::load_parameters(
            engine,
            &model.parameters,
            &directory.join(name),
            manifest.step,
        )
    }
    pub fn load_parameters(
        engine: &mut Engine,
        parameters: &[sejong_model::Parameter],
        path: &Path,
        step: usize,
    ) -> Result<Self> {
        let state = load_tensors(engine, path)?;
        let mut m = vec![];
        let mut v = vec![];
        ensure(
            state.len() == parameters.len() * 2,
            "optimizer tensor count mismatch",
        )?;
        for p in parameters {
            for (prefix, destination) in [("m", &mut m), ("v", &mut v)] {
                let key = format!("{prefix}.{}", p.name);
                let t = state
                    .get(&key)
                    .ok_or_else(|| Error::Message(format!("missing optimizer tensor {key}")))?;
                ensure(t.shape == p.value.shape, "optimizer tensor shape mismatch")?;
                destination.push(t.clone());
            }
        }
        Ok(Self {
            step,
            m,
            v,
            gradients: vec![None; parameters.len()],
        })
    }
    pub fn accumulate(
        &mut self,
        engine: &mut Engine,
        gradients: Vec<Option<Tensor>>,
    ) -> Result<()> {
        ensure(
            gradients.len() == self.gradients.len(),
            "optimizer gradient count mismatch",
        )?;
        for (slot, incoming) in self.gradients.iter_mut().zip(gradients) {
            if let Some(incoming) = incoming {
                if let Some(current) = slot {
                    engine.add_into(current, &incoming, current, 1.)?;
                } else {
                    *slot = Some(incoming);
                }
            }
        }
        Ok(())
    }
    pub fn update(
        &mut self,
        engine: &mut Engine,
        parameters: &[sejong_model::Parameter],
        config: &TrainConfig,
        lr: f32,
    ) -> Result<Option<f64>> {
        let diagnostics = config.gradient_diagnostics_every > 0
            && (self.step + 1).is_multiple_of(config.gradient_diagnostics_every);
        let norm = if config.clip_enabled || diagnostics {
            let value = engine.gradient_norm(&self.gradients)? / config.loss_scale as f64;
            ensure(value.is_finite(), "nonfinite clipping gradient")?;
            Some(value)
        } else {
            None
        };
        let clipping = norm
            .filter(|n| config.clip_enabled && *n > config.clip_norm as f64)
            .map_or(1., |n| config.clip_norm as f64 / n);
        let scale = clipping as f32 / config.loss_scale;
        self.step += 1;
        for (i, p) in parameters.iter().enumerate() {
            let gradient = match self.gradients[i].take() {
                Some(value) => value,
                None => engine.full(&p.value.shape, DType::F32, 0.)?,
            };
            engine.adamw(
                &p.value,
                &gradient,
                &self.m[i],
                &self.v[i],
                self.step,
                config,
                lr,
                scale,
                p.value.shape.len() > 1,
            )?;
        }
        Ok(norm)
    }
    pub fn tensors(&self, parameters: &[sejong_model::Parameter]) -> BTreeMap<String, Tensor> {
        let mut state = BTreeMap::new();
        for (i, p) in parameters.iter().enumerate() {
            state.insert(format!("m.{}", p.name), self.m[i].clone());
            state.insert(format!("v.{}", p.name), self.v[i].clone());
        }
        state
    }
}

pub struct TrainingSummary {
    pub steps: usize,
    pub last_loss: f64,
    pub model_directory: PathBuf,
}

pub(crate) fn counter_metrics(
    engine: &Engine,
    counters: &sejong_gpu::ExecutionCounters,
) -> Option<json::Value> {
    engine.metrics_enabled().then(|| json::json!({
        "launches":counters.launches,"logical_flops":counters.logical_flops,"issued_flops":counters.issued_flops,
        "gemm_launches":counters.gemm_launches,"attention_launches":counters.attention_launches,
        "compilations":counters.compilations,"compile_seconds":counters.compile_seconds,
    }))
}

fn save_execution_metrics(engine: &Engine, output: &Path, elapsed_seconds: f64) -> Result<()> {
    if let Some(profile) = engine.kernel_profile()? {
        std::fs::write(
            output.join("kernel-profile.json"),
            json::to_vec_pretty(&profile)?,
        )?;
    }
    let counters = engine.execution_counters();
    let metrics = if engine.metrics_enabled() {
        json::json!({
            "elapsed_seconds":elapsed_seconds,"metrics_collected":true,
            "scope":"initialization_training_basis_refresh_checkpoint_flush_and_save",
            "logical_flops":counters.logical_flops,"issued_flops":counters.issued_flops,
            "launches":counters.launches,"gemm_launches":counters.gemm_launches,
            "attention_launches":counters.attention_launches,"compilations":counters.compilations,
            "compile_seconds":counters.compile_seconds,
        })
    } else {
        json::json!({"elapsed_seconds":elapsed_seconds,"metrics_collected":false,
        "scope":"initialization_training_checkpoint_and_save"})
    };
    std::fs::write(
        output.join("execution.json"),
        json::to_vec_pretty(&metrics)?,
    )?;
    Ok(())
}
pub fn train(
    config: &Config,
    data: &Path,
    output: &Path,
    resume: Option<&Path>,
) -> Result<TrainingSummary> {
    let corpus = Corpus::open(data)?;
    train_configured(config, &corpus, output, resume, false)
}
pub fn train_measured(
    config: &Config,
    corpus: &Corpus,
    output: &Path,
    resume: Option<&Path>,
) -> Result<TrainingSummary> {
    train_configured(config, corpus, output, resume, true)
}
fn train_configured(
    config: &Config,
    corpus: &Corpus,
    output: &Path,
    resume: Option<&Path>,
    metrics: bool,
) -> Result<TrainingSummary> {
    if config.training.optimizer != sejong_core::OptimizerKind::Adamw {
        return control_variate::train_configured(config, corpus, output, resume, metrics);
    }
    let complete_started = Instant::now();
    config.model.validate()?;
    config.training.validate(&config.model)?;
    ensure(
        !output.exists(),
        format_args!("training destination already exists: {}", output.display()),
    )?;
    ensure(
        corpus.tokens > config.training.sequence,
        "corpus is shorter than the training sequence",
    )?;
    let mut engine = Engine::new(config.gpu.clone())?;
    engine.enable_execution_metrics(metrics);
    let (model, mut optimizer, mut rng) = if let Some(directory) = resume {
        let (model, manifest) = Model::load(&mut engine, directory)?;
        ensure(
            json::to_value(&model.config)? == json::to_value(&config.model)?,
            "resume model configuration differs from checkpoint",
        )?;
        ensure(
            manifest.data_fingerprint.as_deref() == Some(&corpus.fingerprint),
            "resume corpus differs from checkpoint",
        )?;
        let previous = manifest
            .training
            .as_ref()
            .ok_or_else(|| Error::Message("checkpoint training configuration missing".into()))?;
        let mut before = json::to_value(previous)?;
        let mut after = json::to_value(&config.training)?;
        for field in ["checkpoint_every", "log_every"] {
            before.as_object_mut().unwrap().remove(field);
            after.as_object_mut().unwrap().remove(field);
        }
        ensure(
            before == after,
            "resume optimizer, schedule or batch settings differ from checkpoint",
        )?;
        let optimizer = Adam::load(&mut engine, &model, directory, &manifest)?;
        let rng = Rng {
            state: manifest
                .rng_state
                .ok_or_else(|| Error::Message("checkpoint RNG state missing".into()))?,
        };
        (model, optimizer, rng)
    } else {
        let model = Model::initialize(&mut engine, config.model.clone())?;
        let optimizer = Adam::new(&mut engine, &model.parameters)?;
        (model, optimizer, Rng::new(config.training.seed))
    };
    ensure(
        optimizer.step < config.training.steps,
        "checkpoint already reached the configured training steps",
    )?;
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("config.json"), json::to_vec_pretty(config)?)?;
    let mut log = std::io::BufWriter::new(std::fs::File::create(output.join("metrics.jsonl"))?);
    eprintln!(
        "Sejong {} | {} parameters | {} | {} corpus tokens",
        model.config.id,
        model.parameter_count(),
        engine.name(),
        corpus.tokens
    );
    let start = Instant::now();
    let mut last_loss = 0.;
    let tc = &config.training;
    engine.synchronize()?;
    while optimizer.step < tc.steps {
        let counters_before = engine.execution_counters();
        let step_started = Instant::now();
        let summaries = engine.empty(&[tc.accumulation], DType::F64)?;
        for observation in 0..tc.accumulation {
            let (input, target) = corpus.batch(&mut rng, tc.batch, tc.sequence)?;
            let ids = engine.from_i32(&[input.len()], &input)?;
            let targets = engine.from_i32(&[target.len()], &target)?;
            let mut graph = Graph::new(&mut engine, true);
            graph.checkpointing = tc.activation_checkpointing;
            let binding = model.bind(&mut graph, true)?;
            let logits = model.logits(&mut graph, &binding, &ids, tc.sequence)?;
            let objective = graph.cross_entropy(logits, &targets)?;
            let loss_tensor = graph.tensor(objective).clone();
            graph.engine.loss_summary(
                &loss_tensor,
                &summaries,
                observation,
                1. / tc.accumulation as f32,
            )?;
            graph.backward(&[(objective, tc.loss_scale / tc.accumulation as f32)])?;
            let gradients = binding
                .leaves
                .iter()
                .map(|v| graph.gradient(*v).cloned())
                .collect();
            drop(graph);
            optimizer.accumulate(&mut engine, gradients)?;
        }
        let loss = engine.to_f64(&summaries)?.iter().sum::<f64>();
        let lr = tc.lr(optimizer.step);
        let norm = optimizer.update(&mut engine, &model.parameters, tc, lr)?;
        last_loss = loss;
        engine.synchronize()?;
        let step_seconds = step_started.elapsed().as_secs_f64();
        let counters = engine.execution_counters().difference(&counters_before);
        if optimizer.step == 1
            || optimizer.step.is_multiple_of(tc.log_every)
            || optimizer.step == tc.steps
        {
            let record = json::json!({
                "step":optimizer.step,"optimizer":"adamw","loss":loss,"gradient_norm":norm,"learning_rate":lr,
                "step_seconds":step_seconds,"observed_bytes":tc.accumulation*tc.batch*tc.sequence,
                "exact_bytes":tc.accumulation*tc.batch*tc.sequence,"elapsed_seconds":start.elapsed().as_secs_f64(),
                "peak_gpu_mib":engine.peak_bytes()/1048576,"counters":counter_metrics(&engine,&counters)
            });
            writeln!(log, "{record}")?;
            eprintln!("{record}");
        }
        if tc.checkpoint_every > 0
            && optimizer.step.is_multiple_of(tc.checkpoint_every)
            && optimizer.step < tc.steps
        {
            let directory = output
                .join("checkpoints")
                .join(format!("step-{:08}", optimizer.step));
            save_training(
                &model,
                &mut engine,
                &optimizer,
                &directory,
                tc,
                &corpus.fingerprint,
                &rng,
            )?;
        }
    }
    log.flush()?;
    let directory = output.join("model");
    save_training(
        &model,
        &mut engine,
        &optimizer,
        &directory,
        tc,
        &corpus.fingerprint,
        &rng,
    )?;
    save_execution_metrics(&engine, output, complete_started.elapsed().as_secs_f64())?;
    Ok(TrainingSummary {
        steps: optimizer.step,
        last_loss,
        model_directory: directory,
    })
}
fn save_training(
    model: &Model,
    engine: &mut Engine,
    optimizer: &Adam,
    directory: &Path,
    training: &TrainConfig,
    fingerprint: &str,
    rng: &Rng,
) -> Result<()> {
    let mut manifest = model.manifest(optimizer.step);
    manifest.training = Some(training.clone());
    manifest.data_fingerprint = Some(fingerprint.into());
    manifest.rng_state = Some(rng.state);
    model.save(
        engine,
        directory,
        manifest,
        Some(&optimizer.tensors(&model.parameters)),
    )
}
