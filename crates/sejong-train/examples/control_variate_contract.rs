//! 실제 GPU 학습에서 SGD 동치, basis 갱신과 체크포인트 재개를 검증한다.
use sejong_core::{
    Config, ControlSampling, ControlVariateConfig, DType, ExpertConfig, HierarchyConfig,
    ModelConfig, OptimizerKind, Result, TrainConfig, ensure, json, root,
};
use sejong_gpu::Engine;
use sejong_model::load_tensors;
use std::path::Path;

fn compare(engine: &mut Engine, a: &Path, b: &Path) -> Result<f32> {
    let a = load_tensors(engine, a)?;
    let b = load_tensors(engine, b)?;
    ensure(a.keys().eq(b.keys()), "checkpoint tensor names differ")?;
    let mut maximum = 0f32;
    for (name, tensor) in a {
        let av = engine.to_f32(&tensor)?;
        let bv = engine.to_f32(&b[&name])?;
        for (x, y) in av.into_iter().zip(bv) {
            maximum = maximum.max((x - y).abs());
        }
    }
    Ok(maximum)
}

fn main() -> Result<()> {
    let directory = root()
        .join(".cache")
        .join(format!("cv-contract-{}", std::process::id()));
    let data = root().join(".cache/flops-benchmark-data/train.txt");
    let mut config = Config {
        model: ModelConfig {
            id: "sejong-cv-contract".into(),
            width: 32,
            layers: 1,
            heads: 4,
            ffn_width: 64,
            context: 64,
            precision: DType::F32,
            kv_heads: Some(2),
            experts: Some(ExpertConfig {
                count: 2,
                top_k: 1,
                ffn_width: 32,
            }),
            hierarchy: Some(HierarchyConfig {
                patch_size: 4,
                encoder_width: 16,
                encoder_layers: 1,
                encoder_heads: 2,
                encoder_ffn_width: 32,
                decoder_width: 16,
                decoder_layers: 1,
                decoder_heads: 2,
                decoder_ffn_width: 32,
                local_window: 8,
                checkpoint_encoder: false,
                checkpoint_decoder: false,
            }),
            ..Default::default()
        },
        training: TrainConfig {
            optimizer: OptimizerKind::ControlVariateSgd,
            gradient_diagnostics_every: 2,
            control_variate: Some(ControlVariateConfig {
                controls: 2,
                rank: 2,
                sampling: ControlSampling::Stratified,
                strata: 2,
                basis_refresh: 2,
                predictor_learning_rate: 0.1,
            }),
            steps: 4,
            sequence: 16,
            batch: 1,
            accumulation: 4,
            learning_rate: 0.01,
            min_learning_rate: 0.01,
            warmup_steps: 0,
            checkpoint_every: 2,
            log_every: 4,
            weight_decay: 0.,
            clip_enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let continuous = sejong_train::train(&config, &data, &directory.join("continuous"), None)?;
    let metrics: Vec<json::Value> =
        std::fs::read_to_string(directory.join("continuous/metrics.jsonl"))?
            .lines()
            .map(json::from_str)
            .collect::<std::result::Result<_, _>>()?;
    for (index, metric) in metrics.iter().enumerate() {
        let enabled = (index + 1).is_multiple_of(config.training.gradient_diagnostics_every);
        ensure(
            metric["gradient_diagnostics"] == enabled,
            "gradient diagnostic cadence differs from configuration",
        )?;
        ensure(
            metric["predictor_mse"].is_number() == enabled
                && metric["diagnostic_seconds"].is_number() == enabled,
            "disabled gradient diagnostics require null metric values",
        )?;
    }
    let checkpoint = directory.join("continuous/checkpoints/step-00000002");
    let resumed = sejong_train::train(
        &config,
        &data,
        &directory.join("resumed"),
        Some(&checkpoint),
    )?;
    let mut engine = Engine::new(config.gpu.clone())?;
    let resume_weights = compare(
        &mut engine,
        &continuous.model_directory.join("weights.safetensors"),
        &resumed.model_directory.join("weights.safetensors"),
    )?;
    let resume_optimizer = compare(
        &mut engine,
        &continuous.model_directory.join("optimizer.safetensors"),
        &resumed.model_directory.join("optimizer.safetensors"),
    )?;
    ensure(
        resume_weights < 2e-5 && resume_optimizer < 2e-5,
        "resumed SGD/predictor state differs from continuous execution",
    )?;
    let tensors = load_tensors(
        &mut engine,
        &continuous.model_directory.join("optimizer.safetensors"),
    )?;
    let mut learned = 0.;
    for (name, tensor) in &tensors {
        if name.ends_with(".predictor") {
            learned += engine.squared_norm(tensor)?;
        }
    }
    ensure(
        learned > 0. && learned.is_finite(),
        "predictor coefficients require a finite learned update",
    )?;
    let at_refresh = load_tensors(&mut engine, &checkpoint.join("optimizer.safetensors"))?;
    let mut basis_change = 0.;
    for (name, tensor) in &tensors {
        if name.ends_with(".left") || name.ends_with(".right") {
            let delta = engine.add(tensor, &at_refresh[name], -1., DType::F32)?;
            basis_change += engine.squared_norm(&delta)?;
        }
    }
    ensure(
        basis_change > 0. && basis_change.is_finite(),
        "basis refresh requires a finite data-dependent update",
    )?;
    config.training.steps = 1;
    config.training.checkpoint_every = 0;
    config.training.optimizer = OptimizerKind::Sgd;
    let exact = sejong_train::train(&config, &data, &directory.join("exact"), None)?;
    config.training.optimizer = OptimizerKind::ControlVariateSgd;
    config.training.control_variate.as_mut().unwrap().controls = config.training.accumulation;
    let all = sejong_train::train(&config, &data, &directory.join("all-controls"), None)?;
    let all_controls_error = compare(
        &mut engine,
        &exact.model_directory.join("weights.safetensors"),
        &all.model_directory.join("weights.safetensors"),
    )?;
    ensure(
        all_controls_error < 2e-5,
        "all-control correction differs from full SGD",
    )?;
    let result = json::json!({"resume_weights_max_error":resume_weights,"resume_optimizer_max_error":resume_optimizer,"all_controls_sgd_max_error":all_controls_error,"predictor_squared_norm":learned,"basis_change_squared_norm":basis_change,"directory":directory});
    std::fs::write(directory.join("result.json"), json::to_vec_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
