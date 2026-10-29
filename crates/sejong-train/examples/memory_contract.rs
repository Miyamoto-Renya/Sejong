//! One grouped GPU check of scalable memory preparation and resumable local updates.
use sejong_core::{
    DType, GpuConfig, ModelConfig, Result, Rng, ensure, json,
    native::{
        BackgroundPolicy, BatchPolicy, ByteValue, LossPolicy, MemoryConfig, MemoryTraining,
        OptimizerPolicy, ReadMode, RevisionTraining, ValueTraining,
    },
};
use sejong_gpu::{Engine, Tensor};
use sejong_model::{
    Parameter, load_tensors,
    native::{memory::Memory, value::NativeValue},
};
use sejong_train::{
    Adam,
    memory::{revise_memory, train_memory, train_value},
};
use std::path::{Path, PathBuf};

fn check_rows(e: &mut Engine, memory: &Memory) -> Result<f64> {
    let mut largest = 0f64;
    for index in 0..memory.metadata.requests.len() {
        let (full, _) = memory.row(e, index, ReadMode::Full)?;
        let (direct, radius) = memory.row(e, index, ReadMode::Direct)?;
        let tv = full
            .iter()
            .zip(&direct)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 2.;
        ensure(
            tv.is_finite() && radius.is_finite() && tv <= radius,
            "declared full/direct row exceeds its prepared radius",
        )?;
        largest = largest.max(tv);
    }
    Ok(largest)
}

fn resumed_tensor_delta(e: &mut Engine, a: &Tensor, b: &Tensor, updates: usize) -> Result<f64> {
    ensure(a.shape == b.shape, "resumed tensor shape differs")?;
    let a = e.to_f64(a)?;
    let b = e.to_f64(b)?;
    let scale = a.iter().chain(&b).map(|x| x.abs()).fold(0f64, f64::max) as f32;
    let adjacent = f32::from_bits(if scale == f32::MAX {
        scale.to_bits() - 1
    } else {
        scale.to_bits() + 1
    });
    let ulp = (adjacent as f64 - scale as f64).abs();
    let delta = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f64, f64::max);
    ensure(
        delta <= updates as f64 * ulp,
        format!(
            "resumed tensor delta {delta} exceeds {updates} update-sized FP32 ULPs at scale {scale}"
        ),
    )?;
    Ok(delta)
}
fn resumed_memory_delta(e: &mut Engine, a: &Memory, b: &Memory, updates: usize) -> Result<f64> {
    let mut delta = 0f64;
    for (a, b) in a.parameters.iter().zip(&b.parameters) {
        ensure(a.name == b.name, "resumed memory parameter order differs")?;
        delta = delta.max(resumed_tensor_delta(e, &a.value, &b.value, updates)?);
    }
    Ok(delta.max(resumed_tensor_delta(
        e,
        a.port.as_ref().unwrap(),
        b.port.as_ref().unwrap(),
        updates,
    )?))
}

fn check_checkpoint_restoration(e: &mut Engine, root: &Path) -> Result<()> {
    let path = root.join("memory/checkpoints/query-00000004");
    let progress: json::Value = json::from_slice(&std::fs::read(path.join("training.json"))?)?;
    let memory = Memory::load_training(e, &path.join("original"))?;
    ensure(
        progress["source"].as_str() == Some(memory.metadata.source_hash.as_str()),
        "checkpoint configuration identity differs",
    )?;
    let parameters = memory
        .parameters
        .iter()
        .filter(|p| p.name.starts_with("body."))
        .map(|p| Parameter {
            name: p.name.clone(),
            value: p.value.clone(),
        })
        .collect::<Vec<_>>();
    let step = progress["optimizer_step"]
        .as_u64()
        .ok_or_else(|| sejong_core::Error::Message("optimizer step missing".into()))?
        as usize;
    let optimizer =
        Adam::load_parameters(e, &parameters, &path.join("optimizer.safetensors"), step)?;
    let serialized = load_tensors(e, &path.join("optimizer.safetensors"))?;
    ensure(optimizer.step == step, "restored optimizer step differs")?;
    for (name, tensor) in optimizer.tensors(&parameters) {
        ensure(
            e.to_f64(&tensor)? == e.to_f64(&serialized[&name])?,
            format!("restored optimizer state {name} differs"),
        )?;
    }
    let read = |path: &Path| -> Result<json::Value> {
        Ok(json::from_slice(&std::fs::read(
            path.join("training.json"),
        )?)?)
    };
    let uninterrupted = read(&root.join("memory/checkpoints/query-00000008"))?;
    let resumed = read(&root.join("memory-resume/checkpoints/query-00000008"))?;
    ensure(
        uninterrupted == resumed,
        "resumed RNG/step/batch/configuration state differs",
    )?;
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--restore-only") {
        let root = PathBuf::from(std::env::args().nth(2).unwrap());
        let mut e = Engine::new(GpuConfig::default())?;
        check_checkpoint_restoration(&mut e, &root)?;
        let path = root.join("result.json");
        let mut result: json::Value = json::from_slice(&std::fs::read(&path)?)?;
        result["checkpoint_restore"] = json::json!("exact_optimizer_rng_step_batch_config");
        std::fs::write(path, json::to_vec_pretty(&result)?)?;
        println!("{}", json::to_string_pretty(&result)?);
        return Ok(());
    }
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!(".cache/scalable-memory-{}", std::process::id())));
    ensure(
        !root.exists(),
        "select a fresh output directory for the grouped memory check",
    )?;
    std::fs::create_dir_all(&root)?;
    let mut e = Engine::new(GpuConfig::default())?;
    let config = MemoryConfig {
        id: "sejong-scalable-memory".into(),
        dimension: 4,
        tokens: 129,
        layers: 1,
        heads: 1,
        hidden: 8,
        ffn_width: 8,
        norm_epsilon: f32::EPSILON,
        readout_temperature: 1.,
        residual_scale: 1.,
        seed: 6311,
    };
    let d = config.dimension;
    let n = d + 1;
    let mut rng = Rng::new(config.seed);
    let contexts: Vec<Vec<f64>> = (0..n)
        .map(|index| {
            let mut row: Vec<_> = (0..config.tokens * d)
                .map(|_| rng.normal() as f64 / (d as f64).sqrt())
                .collect();
            row[..d].fill(0.);
            if index < d {
                row[index] = 1.;
            }
            row
        })
        .collect();
    let batch = BatchPolicy {
        max_examples: Some(2),
        memory_bytes: None,
    };
    let mut spec = MemoryTraining {
        config,
        requests: (0..n)
            .map(|i| ByteValue::from(format!("request-{i}")))
            .collect(),
        values: vec![ByteValue::from("서울"), ByteValue(vec![255, 0, 128])],
        target_queries: contexts.iter().map(|row| row[..d].to_vec()).collect(),
        contexts,
        value_targets: (0..2)
            .map(|index| {
                let mut target = vec![0.; d];
                target[index] = 1.;
                target
            })
            .collect(),
        labels: (0..n).map(|index| index % 2).collect(),
        query_steps: 8,
        fact_steps: 8,
        query_learning_rate: 0.001,
        min_query_learning_rate: 0.0001,
        learning_rate: 0.001,
        resume: None,
        checkpoint_every: 4,
        query_loss: LossPolicy::default(),
        background: BackgroundPolicy::Empirical,
        batch: batch.clone(),
        optimizer: OptimizerPolicy::default(),
    };
    let memory = train_memory(&mut e, &spec, &root.join("memory"))?;
    let maximum_tv = check_rows(&mut e, &memory)?;
    let restored = Memory::load(&mut e, &root.join("memory/model"))?;
    ensure(
        memory.signature(&mut e)? == restored.signature(&mut e)?,
        "memory roundtrip differs",
    )?;
    spec.resume = Some(root.join("memory/checkpoints/query-00000004"));
    let resumed = train_memory(&mut e, &spec, &root.join("memory-resume"))?;
    check_checkpoint_restoration(&mut e, &root)?;
    let query_resume_delta = resumed_memory_delta(&mut e, &memory, &resumed, spec.query_steps / 2)?;
    let original = memory.signature(&mut e)?;
    let body = memory.body_signature(&mut e)?;
    let port = memory.port_signature(&mut e)?;
    let revision = RevisionTraining {
        labels: spec.labels.iter().map(|index| 1 - index).collect(),
        steps: 8,
        learning_rate: 0.001,
        resume: None,
        checkpoint_every: 0,
        batch: batch.clone(),
        optimizer: OptimizerPolicy::default(),
    };
    let revised = revise_memory(&mut e, &memory, &revision, &root.join("revision"), None)?;
    check_rows(&mut e, &revised)?;
    ensure(
        memory.signature(&mut e)? == original
            && revised.signature(&mut e)? != original
            && revised.body_signature(&mut e)? == body
            && revised.port_signature(&mut e)? == port,
        "local K/V revision changed its frozen query interface",
    )?;
    let revised_resume = revise_memory(
        &mut e,
        &memory,
        &revision,
        &root.join("revision-resume"),
        Some(&root.join("revision/checkpoints/revision-00000008")),
    )?;
    ensure(
        revised.signature(&mut e)? == revised_resume.signature(&mut e)?,
        "revision resume differs",
    )?;
    let mut value_spec = ValueTraining {
        model: ModelConfig {
            id: "sejong-native-resource-batch".into(),
            width: 8,
            layers: 1,
            heads: 1,
            ffn_width: 16,
            context: 8,
            rope_theta: 8.,
            norm_epsilon: f32::EPSILON,
            precision: DType::F32,
            seed: 8411,
        },
        requests: vec![
            ByteValue::from("one"),
            ByteValue::from("two"),
            ByteValue::from("three"),
        ],
        prompts: vec![
            ByteValue::from("a"),
            ByteValue::from("ab"),
            ByteValue::from("abc"),
        ],
        values: spec.values.clone(),
        labels: vec![0, 1, 0],
        steps: 4,
        learning_rate: 0.001,
        resume: None,
        checkpoint_every: 2,
        batch,
        optimizer: OptimizerPolicy::default(),
    };
    let value = train_value(&mut e, &value_spec, &root.join("value"))?;
    let loaded_value = NativeValue::load(&mut e, &root.join("value/model"))?;
    ensure(
        value.signature(&mut e)? == loaded_value.signature(&mut e)?,
        "native value roundtrip differs",
    )?;
    value_spec.resume = Some(root.join("value/checkpoints/value-00000002"));
    let value_resume = train_value(&mut e, &value_spec, &root.join("value-resume"))?;
    let mut native_resume_delta = 0f64;
    for (a, b) in value.parameters().iter().zip(value_resume.parameters()) {
        ensure(a.name == b.name, "resumed native parameter order differs")?;
        native_resume_delta = native_resume_delta.max(resumed_tensor_delta(
            &mut e,
            &a.value,
            &b.value,
            value_spec.steps / 2,
        )?);
    }
    ensure(
        value.row(&mut e, 0)?.iter().all(|x| x.is_finite()),
        "native categorical law is nonfinite",
    )?;
    let output = json::json!({"tokens":129,"declared_rows":n,"maximum_tv":maximum_tv,
        "query_resume_max_delta":query_resume_delta,"revision_resume":"identical",
        "native_resume_max_delta":native_resume_delta,"zero_cadence":"final_checkpoint_saved",
        "checkpoint_restore":"exact_optimizer_rng_step_batch_config","gpu_peak_bytes":e.peak_bytes()});
    std::fs::write(root.join("result.json"), json::to_vec_pretty(&output)?)?;
    println!("{}", json::to_string_pretty(&output)?);
    Ok(())
}
