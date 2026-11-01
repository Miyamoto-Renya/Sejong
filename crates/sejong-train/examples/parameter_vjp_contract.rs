//! 지정한 전체 모델의 압축 SGD 학습·저장·재개와 전체 관측의 정확 SGD를 대조한다.
use sejong_core::{Config, Result, ensure, json};
use sejong_gpu::Engine;
use sejong_model::load_tensors;
use std::path::Path;

fn compare(e: &mut Engine, a: &Path, b: &Path) -> Result<(f64, f64)> {
    let a = load_tensors(e, a)?;
    let b = load_tensors(e, b)?;
    ensure(
        a.keys().eq(b.keys()),
        "training artifact tensor names differ",
    )?;
    let mut error = 0f64;
    let mut magnitude = 0f64;
    for (name, tensor) in &a {
        let av = e.to_f32(tensor)?;
        let bv = e.to_f32(&b[name])?;
        for (x, y) in av.iter().zip(&bv) {
            ensure(
                x.is_finite() && y.is_finite(),
                "nonfinite checkpoint weights",
            )?;
            error = error.max((f64::from(*x) - f64::from(*y)).abs());
            magnitude = magnitude.max(f64::from(*x).abs()).max(f64::from(*y).abs());
        }
    }
    Ok((error, magnitude))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() == 3,
        "usage: parameter_vjp_contract CONFIG DATA OUT",
    )?;
    let mut config = Config::load(Path::new(&args[0]))?;
    ensure(
        config.training.parameter_vjp.is_some(),
        "contract requires a parameter_vjp policy",
    )?;
    let data = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    config.training.steps = 6;
    config.training.checkpoint_every = 3;
    config.training.log_every = 1;
    let full = sejong_train::train(&config, data, &out.join("continuous"), None)?;
    let metrics: Vec<json::Value> = std::fs::read_to_string(out.join("continuous/metrics.jsonl"))?
        .lines()
        .map(json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    ensure(
        metrics.iter().all(|r| {
            r["parameter_vjp_statistics"]["compressed_matrices"]
                .as_u64()
                .is_some_and(|n| n > 0)
        }),
        "full model training must exercise compressed parameter VJPs",
    )?;
    let checkpoint = out.join("continuous/checkpoints/step-00000003");
    let resumed = sejong_train::train(&config, data, &out.join("resumed"), Some(&checkpoint))?;
    let mut e = Engine::new(config.gpu.clone())?;
    let (resume_error, resume_magnitude) = compare(
        &mut e,
        &full.model_directory.join("weights.safetensors"),
        &resumed.model_directory.join("weights.safetensors"),
    )?;
    let (optimizer_error, _) = compare(
        &mut e,
        &full.model_directory.join("optimizer.safetensors"),
        &resumed.model_directory.join("optimizer.safetensors"),
    )?;
    let operations = 2. * (config.training.steps - config.training.checkpoint_every) as f64;
    let delta = operations * f32::EPSILON as f64;
    let budget = delta / (1. - delta) * resume_magnitude;
    ensure(
        resume_error <= budget && optimizer_error == 0.,
        "resumed parameter VJP weights or RNG state differ",
    )?;
    config.training.steps = 1;
    config.training.checkpoint_every = 0;
    config
        .training
        .parameter_vjp
        .as_mut()
        .unwrap()
        .residual_fraction = 1.;
    let covered = sejong_train::train(&config, data, &out.join("covered"), None)?;
    config.training.parameter_vjp = None;
    let exact = sejong_train::train(&config, data, &out.join("exact"), None)?;
    let (coverage_error, magnitude) = compare(
        &mut e,
        &covered.model_directory.join("weights.safetensors"),
        &exact.model_directory.join("weights.safetensors"),
    )?;
    ensure(
        coverage_error <= 2. * f32::EPSILON as f64 * magnitude,
        "full residual coverage differs from exact SGD",
    )?;
    let result = json::json!({"model":config.model,"resume_weights_max_abs_error":resume_error,
        "resume_optimizer_rng_max_abs_error":optimizer_error,"full_coverage_sgd_max_abs_error":coverage_error,
        "resume_roundoff_budget":budget,"continuous_loss":full.last_loss,"resumed_loss":resumed.last_loss,
        "steps":metrics.len(),"directory":out});
    std::fs::write(out.join("result.json"), json::to_vec_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
