//! NativeValue가 만든 동일한 gradient의 인수 갱신과 dense 복원 갱신을 대조한다.
use sejong_core::{GpuConfig, Result, Rng, ensure, json, native::ValueTraining};
use sejong_gpu::{
    Engine,
    graph::Graph,
    parameter_vjp::{GradientMaterializer, GradientPlan, ParameterGradient},
};
use sejong_model::native::value::{FORMAT, NativeValue, ValueMetadata};
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() == 2,
        "usage: native_sgd_update_check SPEC RESULT",
    )?;
    let spec: ValueTraining = json::from_slice(&std::fs::read(&args[0])?)?;
    let policy = spec.parameter_vjp.as_ref().unwrap();
    let mut e = Engine::new(GpuConfig::default())?;
    let model = NativeValue::initialize(
        &mut e,
        ValueMetadata {
            format: FORMAT.into(),
            requests: spec.requests.clone(),
            prompts: spec.prompts.clone(),
            values: spec.values.clone(),
            model: spec.model.clone(),
            steps: 0,
            source_hash: "native-update-comparison".into(),
            loss: 0.,
        },
    )?;
    let weights: Vec<_> = model.parameters().into_iter().map(|p| p.value).collect();
    let mut copies = Vec::new();
    for weight in &weights {
        copies.push(e.scale(weight, 1.)?);
    }
    let compact = GradientPlan::new(&mut e, &weights)?;
    let dense = GradientPlan::new(&mut e, &copies)?;
    let materializer = GradientMaterializer::new(&mut e, &weights)?;
    let batch = spec
        .batch
        .max_examples
        .unwrap_or(spec.requests.len())
        .min(spec.requests.len());
    let indices: Vec<_> = (0..batch).collect();
    let labels: Vec<_> = indices.iter().map(|i| spec.labels[*i] as i32).collect();
    let label = e.from_i32(&[batch], &labels)?;
    let mut g = Graph::new(&mut e, true);
    g.checkpointing = false;
    g.configure_parameter_vjp(policy.clone(), Rng::new(policy.seed))?;
    let (logits, leaves) = model.forward_batch(&mut g, &indices, true)?;
    let loss = g.cross_entropy(logits, &label)?;
    g.backward(&[(loss, 1.)])?;
    let stats = g.parameter_vjp_statistics().clone();
    let gradients: Vec<_> = leaves.iter().map(|v| g.parameter_gradient(*v)).collect();
    drop(g);
    let lr = spec.learning_rate;
    let settings = spec
        .optimizer
        .configuration(spec.steps, lr, batch, spec.requests.len())?;
    e.synchronize()?;
    let started = Instant::now();
    let packed = compact.pack(&mut e, &gradients)?;
    compact.update(&mut e, &packed, lr, 1., settings.weight_decay)?;
    e.synchronize()?;
    let compact_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let full = materializer.materialize(&mut e, &gradients)?;
    let restored: Vec<_> = full
        .into_iter()
        .map(|dense| ParameterGradient {
            dense,
            terms: vec![],
        })
        .collect();
    let packed_dense = dense.pack(&mut e, &restored)?;
    dense.update(&mut e, &packed_dense, lr, 1., settings.weight_decay)?;
    e.synchronize()?;
    let materialized_seconds = started.elapsed().as_secs_f64();
    let mut max_abs = 0f64;
    let mut squared = 0f64;
    let mut norm = 0f64;
    for (a, b) in weights.iter().zip(&copies) {
        for (x, y) in e.to_f32(a)?.iter().zip(e.to_f32(b)?) {
            let delta = f64::from(*x) - f64::from(y);
            max_abs = max_abs.max(delta.abs());
            squared += delta * delta;
            norm += f64::from(y).powi(2);
        }
    }
    let relative_l2 = (squared / norm).sqrt();
    let operations = 2 * (spec.model.width.max(spec.model.ffn_width) + policy.rank);
    let delta = operations as f64 * f64::from(f32::EPSILON) / 2.;
    let budget = delta / (1. - delta);
    ensure(
        stats.compressed_matrices > 0 && relative_l2 <= budget,
        "native factor SGD differs beyond the FP32 reduction comparison budget",
    )?;
    let result = json::json!({"scope":"one NativeValue backward, same gradients and initial weights; paired optimizer updates",
        "parameters":weights.iter().map(|w|w.len()).sum::<usize>(),
        "compressed_matrices":stats.compressed_matrices,"dense_matrices":stats.dense_matrices,
        "compact_update_seconds":compact_seconds,"materialized_update_seconds":materialized_seconds,
        "weights_max_abs_error":max_abs,"weights_relative_l2_error":relative_l2,
        "reduction_comparison_budget":budget});
    let path = Path::new(&args[1]);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json::to_vec_pretty(&result)?)?;
    println!("{}", json::to_string_pretty(&result)?);
    Ok(())
}
