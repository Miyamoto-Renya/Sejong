//! Resumable local query preparation, vector memory learning and native value heads.
use crate::Adam;
use sejong_core::{
    DType, Result, Rng, TrainConfig, ensure, json,
    native::{BackgroundPolicy, BatchPolicy, MemoryTraining, RevisionTraining, ValueTraining},
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::{
    Parameter,
    native::{
        atomic_directory,
        memory::{Memory, MemoryMetadata, RevisionGuard, RevisionProvenance},
        value::{NativeValue, ValueMetadata},
    },
    save_tensors,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct Progress {
    format: String,
    phase: String,
    step: usize,
    optimizer_step: usize,
    rng: u64,
    source: String,
    guard: Option<RevisionGuard>,
    batch_size: usize,
}
fn fingerprint(spec: &impl Serialize, recipe: &str) -> Result<String> {
    let mut value = json::to_value(spec)?;
    if let Some(map) = value.as_object_mut() {
        map.remove("resume");
    }
    Ok(blake3::hash(&json::to_vec(
        &json::json!({"recipe":recipe,"specification":value}),
    )?)
    .to_hex()
    .to_string())
}
pub fn memory_fingerprint(spec: &MemoryTraining) -> Result<String> {
    fingerprint(spec, "memory-explicit-policies-streamed-rows-v2")
}
pub fn value_fingerprint(spec: &ValueTraining) -> Result<String> {
    fingerprint(spec, "native-value-resource-batch-explicit-policy-v1")
}
pub fn revision_fingerprint(spec: &RevisionTraining) -> Result<String> {
    fingerprint(spec, "memory-revision-resource-chunks-explicit-policy-v1")
}
fn allocation(elements: u128, dtype: DType) -> Result<usize> {
    let bytes = usize::try_from(elements.checked_mul(dtype.bytes() as u128).ok_or_else(|| {
        sejong_core::Error::Message("training workspace element overflow".into())
    })?)
    .map_err(|_| sejong_core::Error::Message("training workspace shape overflow".into()))?;
    Engine::allocation_bytes(bytes)
}
fn workspace_sum(terms: impl IntoIterator<Item = (u128, usize)>) -> Result<usize> {
    let bytes = terms.into_iter().try_fold(0u128, |sum, (count, size)| {
        count
            .checked_mul(size as u128)
            .and_then(|bytes| sum.checked_add(bytes))
            .ok_or_else(|| sejong_core::Error::Message("training workspace sum overflow".into()))
    })?;
    usize::try_from(bytes)
        .map_err(|_| sejong_core::Error::Message("training workspace size overflow".into()))
}
fn work_bytes(memory: &Memory, query_training: bool) -> Result<usize> {
    let c = &memory.metadata.config;
    let (t, d, f, l, h) = (
        c.tokens as u128,
        c.dimension as u128,
        c.ffn_width as u128,
        c.layers as u128,
        c.heads as u128,
    );
    // One query keeps 16 width-sized and three FF-sized tensors per block,
    // its input, first-row gather/output, indices and attention normalizers.
    let query = workspace_sum([
        (1 + 16 * l, allocation(t * d, DType::F32)?),
        (3 * l, allocation(t * f, DType::F32)?),
        (l, allocation(t * h, DType::F32)?),
        (2, allocation(d, DType::F32)?),
        (1, allocation(1, DType::I32)?),
    ])?;
    // Three query views retain their forward values and adjoints together.
    if query_training {
        workspace_sum([(6, query)])
    } else {
        workspace_sum([
            (1, query),
            (4, allocation(c.hidden as u128, DType::F32)?),
            (2, allocation(d, DType::F32)?),
        ])
    }
}
fn value_work_bytes(model: &NativeValue) -> Result<usize> {
    let c = &model.metadata.model;
    let t = model
        .metadata
        .prompts
        .iter()
        .map(|x| x.len() + 2)
        .max()
        .unwrap_or(0) as u128;
    let (d, f, l, h) = (
        c.width as u128,
        c.ffn_width as u128,
        c.layers as u128,
        c.heads as u128,
    );
    // Count each tensor allocation independently, including allocator rounding.
    // FP32 covers both model compute dtypes and their FP32 adjoints.
    workspace_sum([
        (2 * (2 + 12 * l), allocation(t * d, DType::F32)?),
        (6 * l, allocation(t * f, DType::F32)?),
        (2 * l, allocation(t * h, DType::F32)?),
        (4, allocation(d, DType::F32)?),
        (
            4,
            allocation(model.metadata.values.len() as u128, DType::F32)?,
        ),
        (2, allocation(1, DType::F32)?),
        (1, allocation(t, DType::I32)?),
        (2, allocation(1, DType::I32)?),
    ])
}
fn resolve_batch(
    e: &Engine,
    policy: &BatchPolicy,
    bytes: usize,
    observations: usize,
    parameters: &[Parameter],
    fixed_workspace: usize,
) -> Result<usize> {
    // Current, incoming and summed parameter adjoints can coexist.
    let reserve = workspace_sum(
        parameters
            .iter()
            .map(|p| allocation(p.value.len() as u128, DType::F32).map(|size| (3, size)))
            .collect::<Result<Vec<_>>>()?,
    )?;
    policy.resolve(
        e.available_memory_bytes()?
            .saturating_sub(reserve)
            .saturating_sub(fixed_workspace),
        bytes,
        observations,
    )
}
fn resume_batch(available: usize, saved: Option<&Progress>) -> Result<usize> {
    let batch = saved.map_or(available, |p| p.batch_size);
    ensure(
        batch > 0 && batch <= available,
        "checkpoint batch exceeds current workspace; select matching resources or start a new learning run",
    )?;
    Ok(batch)
}
fn parameters(memory: &Memory, body: bool) -> Vec<Parameter> {
    memory
        .parameters
        .iter()
        .filter(|p| {
            if body {
                p.name.starts_with("body.")
            } else {
                p.name == "key" || p.name == "value"
            }
        })
        .map(|p| Parameter {
            name: p.name.clone(),
            value: p.value.clone(),
        })
        .collect()
}
fn checkpoint(
    e: &mut Engine,
    memory: &Memory,
    optimizer: &Adam,
    parameters: &[Parameter],
    out: &Path,
    progress: &Progress,
) -> Result<()> {
    atomic_directory(out, |dir| {
        memory.save_training(e, &dir.join("original"))?;
        let tensors = optimizer.tensors(parameters);
        save_tensors(
            e,
            &dir.join("optimizer.safetensors"),
            tensors.iter().map(|(n, t)| (n.as_str(), t)),
        )?;
        std::fs::write(dir.join("training.json"), json::to_vec_pretty(progress)?)?;
        Ok(())
    })
}
fn load_progress(path: &Path, source: &str) -> Result<Progress> {
    let p: Progress = json::from_slice(&std::fs::read(path.join("training.json"))?)?;
    ensure(
        p.format == "sejong-local-training-v3" && p.source == source,
        "local checkpoint source/format mismatch",
    )?;
    Ok(p)
}
fn progress(
    phase: &str,
    step: usize,
    optimizer: &Adam,
    rng: &Rng,
    source: &str,
    guard: Option<RevisionGuard>,
    batch_size: usize,
) -> Progress {
    Progress {
        format: "sejong-local-training-v3".into(),
        phase: phase.into(),
        step,
        optimizer_step: optimizer.step,
        rng: rng.state,
        source: source.into(),
        guard,
        batch_size,
    }
}
fn open_run(out: &Path) -> Result<()> {
    ensure(
        !out.exists(),
        format!("output already exists: {}", out.display()),
    )?;
    std::fs::create_dir_all(out.join("checkpoints"))?;
    Ok(())
}

pub fn train_memory(e: &mut Engine, spec: &MemoryTraining, out: &Path) -> Result<Memory> {
    let source = memory_fingerprint(spec)?;
    let d = spec.config.dimension;
    let n = spec.requests.len();
    spec.optimizer
        .configuration(spec.query_steps, spec.query_learning_rate, n, n)?;
    spec.optimizer
        .configuration(spec.fact_steps, spec.learning_rate, n, n)?;
    ensure(
        spec.min_query_learning_rate.is_finite()
            && spec.min_query_learning_rate > 0.
            && spec.min_query_learning_rate <= spec.query_learning_rate,
        "query cosine minimum learning rate invalid",
    )?;
    ensure(
        n >= 2
            && spec.target_queries.len() == n
            && spec
                .target_queries
                .iter()
                .all(|q| q.len() == d && q.iter().all(|x| x.is_finite())),
        "local query targets invalid",
    )?;
    ensure(
        spec.labels.len() == n && spec.labels.iter().all(|i| *i < spec.values.len()),
        "memory labels must match the declared input/output inventory",
    )?;
    open_run(out)?;
    let resume = spec.resume.as_deref();
    let saved = resume.map(|p| load_progress(p, &source)).transpose()?;
    let mut memory = if let Some(path) = resume {
        Memory::load_training(e, &path.join("original"))?
    } else {
        Memory::initialize(
            e,
            MemoryMetadata {
                format: sejong_model::native::memory::FORMAT.into(),
                config: spec.config.clone(),
                requests: spec.requests.clone(),
                values: spec.values.clone(),
                contexts: spec.contexts.clone(),
                value_targets: spec.value_targets.clone(),
                query_radius: vec![],
                row_radius: vec![],
                body_signature: String::new(),
                query_steps: 0,
                fact_steps: 0,
                source_hash: source.clone(),
                revisions: vec![],
            },
        )?
    };
    ensure(
        memory.metadata.source_hash == source,
        "memory checkpoint training identity mismatch",
    )?;
    let (guard, fact_resume) = if saved.as_ref().is_some_and(|p| p.phase == "facts") {
        (
            saved.as_ref().unwrap().guard.clone().ok_or_else(|| {
                sejong_core::Error::Message("fact checkpoint revision guard required".into())
            })?,
            true,
        )
    } else {
        let params = parameters(&memory, true);
        let mut optimizer = if let (Some(path), Some(p)) = (resume, &saved) {
            ensure(
                p.phase == "query" && p.step <= spec.query_steps,
                "query checkpoint phase/step mismatch",
            )?;
            Adam::load_parameters(
                e,
                &params,
                &path.join("optimizer.safetensors"),
                p.optimizer_step,
            )?
        } else {
            Adam::new(e, &params)?
        };
        let mut rng = Rng::new(spec.config.seed ^ 0x7175657279);
        if let Some(p) = &saved {
            rng.state = p.rng;
        }
        let start = saved.as_ref().map_or(0, |p| p.step);
        let allowed = resolve_batch(e, &spec.batch, work_bytes(&memory, true)?, n, &params, 0)?;
        let batch = resume_batch(allowed, saved.as_ref())?;
        let settings =
            spec.optimizer
                .configuration(spec.query_steps, spec.query_learning_rate, batch, n)?;
        for step in start..spec.query_steps {
            let loss = query_update(
                e,
                &memory,
                spec,
                &mut rng,
                &mut optimizer,
                &params,
                &settings,
                batch,
            )?;
            memory.metadata.query_steps = step + 1;
            if step == start
                || (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
                || step + 1 == spec.query_steps
            {
                eprintln!("{} query {} loss {:.6}", spec.config.id, step + 1, loss);
            }
            if (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
                || step + 1 == spec.query_steps
            {
                checkpoint(
                    e,
                    &memory,
                    &optimizer,
                    &params,
                    &out.join(format!("checkpoints/query-{:08}", step + 1)),
                    &progress("query", step + 1, &optimizer, &rng, &source, None, batch),
                )?;
            }
        }
        memory.fit_port(e)?;
        let guard = memory.begin_revision(e)?;
        memory.to_precision(e, DType::F32)?;
        (guard, false)
    };
    let settings = spec
        .optimizer
        .configuration(spec.fact_steps, spec.learning_rate, n, n)?;
    let params = parameters(&memory, false);
    let mut optimizer = if fact_resume {
        let p = saved.as_ref().unwrap();
        Adam::load_parameters(
            e,
            &params,
            &resume.unwrap().join("optimizer.safetensors"),
            p.optimizer_step,
        )?
    } else {
        Adam::new(e, &params)?
    };
    let allowed = resolve_batch(e, &spec.batch, work_bytes(&memory, false)?, n, &params, 0)?;
    let chunks = resume_batch(allowed, if fact_resume { saved.as_ref() } else { None })?;
    let mut rng = Rng::new(spec.config.seed ^ 0x6661637473);
    if fact_resume {
        rng.state = saved.as_ref().unwrap().rng;
    }
    let start = if fact_resume {
        saved.as_ref().unwrap().step
    } else {
        0
    };
    ensure(
        start <= spec.fact_steps,
        "fact checkpoint step exceeds target",
    )?;
    for step in start..spec.fact_steps {
        let loss = fact_update(
            e,
            &memory,
            &spec.labels,
            &mut rng,
            &mut optimizer,
            &params,
            &settings,
            spec.learning_rate,
            chunks,
        )?;
        memory.metadata.fact_steps = step + 1;
        if step == start
            || (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.fact_steps
        {
            eprintln!("{} facts {} mse {:.6}", spec.config.id, step + 1, loss);
        }
        if (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.fact_steps
        {
            checkpoint(
                e,
                &memory,
                &optimizer,
                &params,
                &out.join(format!("checkpoints/facts-{:08}", step + 1)),
                &progress(
                    "facts",
                    step + 1,
                    &optimizer,
                    &rng,
                    &source,
                    Some(guard.clone()),
                    chunks,
                ),
            )?;
        }
    }
    memory.finalize_revision(e, &guard)?;
    memory.save(e, &out.join("model"))?;
    Ok(memory)
}
fn query_context(spec: &MemoryTraining, request: &[f32], rng: &mut Rng) -> Result<Vec<f32>> {
    let mut row = request.to_vec();
    let d = spec.config.dimension;
    let tokens = spec.config.tokens;
    match &spec.background {
        BackgroundPolicy::Empirical => {
            let observed = &spec.contexts[rng.index(spec.contexts.len())];
            row.extend(observed[d..].iter().map(|x| *x as f32));
        }
        BackgroundPolicy::Gaussian {
            standard_deviation,
            zero_probability,
        } => {
            ensure(
                standard_deviation.is_finite()
                    && *standard_deviation >= 0.
                    && zero_probability.is_finite()
                    && (0. ..=1.).contains(zero_probability),
                "invalid explicit background distribution",
            )?;
            let scale = if rng.uniform() < *zero_probability {
                0.
            } else {
                *standard_deviation
            };
            row.extend((d..tokens * d).map(|_| rng.normal() * scale));
        }
    }
    Ok(row)
}
fn query_update(
    e: &mut Engine,
    memory: &Memory,
    spec: &MemoryTraining,
    rng: &mut Rng,
    optimizer: &mut Adam,
    parameters: &[Parameter],
    settings: &TrainConfig,
    batch: usize,
) -> Result<f64> {
    let c = &spec.config;
    let d = c.dimension;
    let indices: Vec<_> = (0..batch).map(|_| rng.index(spec.requests.len())).collect();
    let mut first = vec![];
    let mut alternate = vec![];
    let mut target = vec![];
    for &i in &indices {
        let request: Vec<_> = spec.contexts[i][..d].iter().map(|x| *x as f32).collect();
        first.extend(query_context(spec, &request, rng)?);
        alternate.extend(query_context(spec, &request, rng)?);
        target.extend(spec.target_queries[i].iter().map(|x| *x as f32));
    }
    let mut middle = vec![];
    for row in 0..batch {
        let request: Vec<_> = (0..d)
            .map(|j| {
                0.5 * (first[row * c.tokens * d + j]
                    + first[((row + batch - 1) % batch) * c.tokens * d + j])
            })
            .collect();
        middle.extend(query_context(spec, &request, rng)?);
    }
    let a = e.from_f32(&[batch * c.tokens, d], &first)?;
    let b = e.from_f32(&[batch * c.tokens, d], &alternate)?;
    let m = e.from_f32(&[batch * c.tokens, d], &middle)?;
    let mut g = Graph::new(e, true);
    let weights = memory.bind(&mut g, true, false);
    let qa = memory.query(&mut g, &weights, &a)?;
    let qb = memory.query(&mut g, &weights, &b)?;
    let qm = memory.query(&mut g, &weights, &m)?;
    let av = g.value(qa)?;
    let bv = g.value(qb)?;
    let mv = g.value(qm)?;
    let count = (batch * d) as f32;
    let mut da = vec![0.; batch * d];
    let mut db = da.clone();
    let mut dm = da.clone();
    let mut loss = 0.;
    let loss_weights = spec
        .query_loss
        .normalized(&["target", "background", "midpoint"])?;
    let (wt, wc, wm) = (
        loss_weights[0] as f32,
        loss_weights[1] as f32,
        loss_weights[2] as f32,
    );
    for row in 0..batch {
        for j in 0..d {
            let k = row * d + j;
            let prev = ((row + batch - 1) % batch) * d + j;
            let difference = av[k] - bv[k];
            let midpoint = mv[k] - 0.5 * (av[k] + av[prev]);
            loss += (wt * 0.5 * ((av[k] - target[k]).powi(2) + (bv[k] - target[k]).powi(2))
                + wc * difference * difference
                + wm * midpoint * midpoint) as f64
                / count as f64;
            da[k] += (wt * (av[k] - target[k]) + 2. * wc * difference - wm * midpoint) / count;
            da[prev] -= wm * midpoint / count;
            db[k] = (wt * (bv[k] - target[k]) - 2. * wc * difference) / count;
            dm[k] = 2. * wm * midpoint / count;
        }
    }
    let shape = [batch, d];
    let roots = [
        (qa, g.engine.from_f32(&shape, &da)?),
        (qb, g.engine.from_f32(&shape, &db)?),
        (qm, g.engine.from_f32(&shape, &dm)?),
    ];
    g.backward_seeded(&roots)?;
    let gradients = parameters
        .iter()
        .map(|p| g.gradient(weights[&p.name]).cloned())
        .collect();
    drop(g);
    optimizer.accumulate(e, gradients)?;
    let position = optimizer.step as f64 / spec.query_steps as f64;
    let lr = spec.min_query_learning_rate as f64
        + 0.5
            * (spec.query_learning_rate - spec.min_query_learning_rate) as f64
            * (1. + (std::f64::consts::PI * position).cos());
    optimizer.update(e, parameters, settings, lr as f32)?;
    Ok(loss)
}
fn fact_update(
    e: &mut Engine,
    memory: &Memory,
    labels: &[usize],
    _rng: &mut Rng,
    optimizer: &mut Adam,
    parameters: &[Parameter],
    settings: &TrainConfig,
    lr: f32,
    chunks: usize,
) -> Result<f64> {
    let c = &memory.metadata.config;
    let d = c.dimension;
    let normalization = (labels.len() * d) as f32;
    let mut loss = 0.;
    for start in (0..labels.len()).step_by(chunks) {
        let end = (start + chunks).min(labels.len());
        let count = end - start;
        let contexts: Vec<f32> = memory.metadata.contexts[start..end]
            .iter()
            .flatten()
            .map(|x| *x as f32)
            .collect();
        let targets: Vec<f32> = labels[start..end]
            .iter()
            .flat_map(|label| {
                memory.metadata.value_targets[*label]
                    .iter()
                    .map(|x| *x as f32)
            })
            .collect();
        let input = e.from_f32(&[count * c.tokens, d], &contexts)?;
        let mut g = Graph::new(e, true);
        let w = memory.bind(&mut g, false, true);
        let q = memory.query(&mut g, &w, &input)?;
        let output = memory.read_query(&mut g, &w, q)?;
        let values = g.value(output)?;
        loss += values
            .iter()
            .zip(&targets)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / normalization as f64;
        let gradient: Vec<_> = values
            .iter()
            .zip(&targets)
            .map(|(a, b)| 2. * (a - b) / normalization)
            .collect();
        let seed = g.engine.from_f32(&[count, d], &gradient)?;
        g.backward_seeded(&[(output, seed)])?;
        let gradients = parameters
            .iter()
            .map(|p| g.gradient(w[&p.name]).cloned())
            .collect();
        drop(g);
        optimizer.accumulate(e, gradients)?;
    }
    optimizer.update(e, parameters, settings, lr)?;
    Ok(loss)
}
pub fn revise_memory(
    e: &mut Engine,
    original: &Memory,
    spec: &RevisionTraining,
    out: &Path,
    resume: Option<&Path>,
) -> Result<Memory> {
    let source = revision_fingerprint(spec)?;
    let n = spec.labels.len();
    let settings = spec
        .optimizer
        .configuration(spec.steps, spec.learning_rate, n, n)?;
    ensure(
        spec.labels.len() == original.metadata.requests.len()
            && spec
                .labels
                .iter()
                .all(|i| *i < original.metadata.values.len()),
        "revision labels invalid",
    )?;
    original.verify(e)?;
    open_run(out)?;
    let resume = resume.or(spec.resume.as_deref());
    let (mut memory, guard, saved) = if let Some(path) = resume {
        let p = load_progress(path, &source)?;
        ensure(
            p.phase == "revision"
                && p.guard.as_ref().is_some_and(|g| {
                    g.parent_signature == original.signature(e).unwrap_or_default()
                }),
            "revision checkpoint parent mismatch",
        )?;
        let memory = Memory::load_training(e, &path.join("original"))?;
        let guard = p.guard.clone().unwrap();
        (memory, guard, Some(p))
    } else {
        let (mut memory, guard) = original.fork_revision(e)?;
        memory.to_precision(e, DType::F32)?;
        (memory, guard, None)
    };
    let params = parameters(&memory, false);
    let mut optimizer = if let Some(p) = &saved {
        Adam::load_parameters(
            e,
            &params,
            &resume.unwrap().join("optimizer.safetensors"),
            p.optimizer_step,
        )?
    } else {
        Adam::new(e, &params)?
    };
    let allowed = resolve_batch(e, &spec.batch, work_bytes(&memory, false)?, n, &params, 0)?;
    let chunks = resume_batch(allowed, saved.as_ref())?;
    let mut rng =
        Rng::new(memory.metadata.config.seed ^ 0x726576697365 ^ memory.metadata.fact_steps as u64);
    if let Some(p) = &saved {
        rng.state = p.rng;
    }
    let start = saved.as_ref().map_or(0, |p| p.step);
    ensure(
        start <= spec.steps,
        "revision checkpoint step exceeds target",
    )?;
    let initial_steps = original.metadata.fact_steps;
    for step in start..spec.steps {
        let loss = fact_update(
            e,
            &memory,
            &spec.labels,
            &mut rng,
            &mut optimizer,
            &params,
            &settings,
            spec.learning_rate,
            chunks,
        )?;
        memory.metadata.fact_steps = initial_steps + step + 1;
        if step == start
            || (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.steps
        {
            eprintln!(
                "{} revision {} mse {:.6}",
                memory.metadata.config.id,
                step + 1,
                loss
            );
        }
        if (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.steps
        {
            checkpoint(
                e,
                &memory,
                &optimizer,
                &params,
                &out.join(format!("checkpoints/revision-{:08}", step + 1)),
                &progress(
                    "revision",
                    step + 1,
                    &optimizer,
                    &rng,
                    &source,
                    Some(guard.clone()),
                    chunks,
                ),
            )?;
        }
    }
    memory.metadata.revisions.push(RevisionProvenance {
        parent_signature: guard.parent_signature.clone(),
        data_fingerprint: source,
        steps: spec.steps,
        learning_rate: spec.learning_rate,
    });
    memory.finalize_revision(e, &guard)?;
    memory.save(e, &out.join("model"))?;
    Ok(memory)
}

pub fn train_value(e: &mut Engine, spec: &ValueTraining, out: &Path) -> Result<NativeValue> {
    spec.optimizer.configuration(
        spec.steps,
        spec.learning_rate,
        spec.requests.len(),
        spec.requests.len(),
    )?;
    ensure(
        spec.labels.len() == spec.requests.len()
            && spec.labels.iter().all(|i| *i < spec.values.len()),
        "native value labels invalid",
    )?;
    let source = value_fingerprint(spec)?;
    open_run(out)?;
    let saved = spec
        .resume
        .as_deref()
        .map(|p| load_progress(p, &source))
        .transpose()?;
    let mut model = if let Some(path) = &spec.resume {
        NativeValue::load_training(e, &path.join("original"))?
    } else {
        NativeValue::initialize(
            e,
            ValueMetadata {
                format: sejong_model::native::value::FORMAT.into(),
                requests: spec.requests.clone(),
                prompts: spec.prompts.clone(),
                values: spec.values.clone(),
                model: spec.model.clone(),
                steps: 0,
                source_hash: source.clone(),
                loss: 0.,
            },
        )?
    };
    ensure(
        model.metadata.source_hash == source,
        "native value checkpoint source mismatch",
    )?;
    let parameters = model.parameters();
    let mut optimizer = if let Some(p) = &saved {
        ensure(p.phase == "value", "native value checkpoint phase mismatch")?;
        Adam::load_parameters(
            e,
            &parameters,
            &spec.resume.as_ref().unwrap().join("optimizer.safetensors"),
            p.optimizer_step,
        )?
    } else {
        Adam::new(e, &parameters)?
    };
    let allowed = resolve_batch(
        e,
        &spec.batch,
        value_work_bytes(&model)?,
        spec.requests.len(),
        &parameters,
        if model.body.config.precision == DType::F32 {
            0
        } else {
            let casts = model
                .body
                .parameters
                .iter()
                .map(|p| p.value.len())
                .chain(std::iter::once(model.embedding.len()))
                .map(|elements| {
                    allocation(elements as u128, model.body.config.precision).map(|size| (1, size))
                })
                .collect::<Result<Vec<_>>>()?;
            workspace_sum(casts)?
        },
    )?;
    let batch = resume_batch(allowed, saved.as_ref())?;
    let settings =
        spec.optimizer
            .configuration(spec.steps, spec.learning_rate, batch, spec.requests.len())?;
    let mut rng = Rng::new(spec.model.seed ^ 0x6e6174697665);
    if let Some(p) = &saved {
        rng.state = p.rng;
    }
    let start = saved.as_ref().map_or(0, |p| p.step);
    ensure(
        start <= spec.steps,
        "native value checkpoint step exceeds target",
    )?;
    for step in start..spec.steps {
        let indices: Vec<_> = (0..batch).map(|_| rng.index(spec.requests.len())).collect();
        let label = e.from_i32(
            &[batch],
            &indices
                .iter()
                .map(|i| spec.labels[*i] as i32)
                .collect::<Vec<_>>(),
        )?;
        let mut g = Graph::new(e, true);
        let (logits, leaves) = model.forward_batch(&mut g, &indices, true)?;
        let loss = g.cross_entropy(logits, &label)?;
        let value = g.value(loss)?.iter().map(|x| *x as f64).sum::<f64>() / batch as f64;
        g.backward(&[(loss, 1.)])?;
        let gradients = leaves.iter().map(|v| g.gradient(*v).cloned()).collect();
        drop(g);
        optimizer.accumulate(e, gradients)?;
        optimizer.update(e, &parameters, &settings, spec.learning_rate)?;
        model.metadata.steps = step + 1;
        model.metadata.loss = value;
        if step == start
            || (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.steps
        {
            eprintln!("{} native {} loss {:.6}", spec.model.id, step + 1, value);
        }
        if (spec.checkpoint_every > 0 && (step + 1) % spec.checkpoint_every == 0)
            || step + 1 == spec.steps
        {
            let state = progress("value", step + 1, &optimizer, &rng, &source, None, batch);
            atomic_directory(
                &out.join(format!("checkpoints/value-{:08}", step + 1)),
                |dir| {
                    model.save_training(e, &dir.join("original"))?;
                    let tensors = optimizer.tensors(&parameters);
                    save_tensors(
                        e,
                        &dir.join("optimizer.safetensors"),
                        tensors.iter().map(|(n, t)| (n.as_str(), t)),
                    )?;
                    std::fs::write(dir.join("training.json"), json::to_vec_pretty(&state)?)?;
                    Ok(())
                },
            )?;
        }
    }
    model.seal(e)?;
    model.save(e, &out.join("model"))?;
    Ok(model)
}
