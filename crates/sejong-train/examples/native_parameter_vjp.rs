//! 지정한 NativeValue 전체 본체의 압축 미분·Adam·봉인·저장·재개를 확인한다.
use sejong_core::{
    GpuConfig, Result, ensure, json,
    native::{ValueTraining, softmax},
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::native::value::NativeValue;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(args.len() == 2, "usage: native_parameter_vjp SPEC OUT")?;
    let mut spec: ValueTraining = json::from_slice(&std::fs::read(&args[0])?)?;
    ensure(
        spec.parameter_vjp.is_some(),
        "native contract requires parameter_vjp",
    )?;
    spec.steps = 2;
    spec.checkpoint_every = 1;
    spec.resume = None;
    let out = Path::new(&args[1]);
    let mut e = Engine::new(GpuConfig::default())?;
    let value = sejong_train::native::train_value(&mut e, &spec, &out.join("continuous"))?;
    let loaded = NativeValue::load(&mut e, &out.join("continuous/model"))?;
    let mut reload_error = 0f64;
    for (a, b) in value.parameters().iter().zip(loaded.parameters()) {
        for (x, y) in e.to_f32(&a.value)?.iter().zip(e.to_f32(&b.value)?) {
            reload_error = reload_error.max((f64::from(*x) - f64::from(y)).abs());
        }
    }
    ensure(
        reload_error == 0.,
        "sealed native value weights changed during reload",
    )?;
    spec.resume = Some(out.join("continuous/checkpoints/value-00000001"));
    let resumed = sejong_train::native::train_value(&mut e, &spec, &out.join("resumed"))?;
    let mut resume_error = 0f64;
    for (a, b) in value.parameters().iter().zip(resumed.parameters()) {
        ensure(a.name == b.name, "native parameter order differs")?;
        for (x, y) in e.to_f32(&a.value)?.iter().zip(e.to_f32(&b.value)?) {
            resume_error = resume_error.max((f64::from(*x) - f64::from(y)).abs());
        }
    }
    let first: json::Value = json::from_slice(&std::fs::read(
        out.join("continuous/checkpoints/value-00000002/training.json"),
    )?)?;
    let second: json::Value = json::from_slice(&std::fs::read(
        out.join("resumed/checkpoints/value-00000002/training.json"),
    )?)?;
    ensure(
        first["parameter_rng"] == second["parameter_rng"],
        "native VJP RNG resume differs",
    )?;
    let requests: Vec<_> = (0..spec.requests.len()).collect();
    let mut probabilities = Vec::new();
    for model in [&value, &resumed] {
        let mut graph = Graph::new(&mut e, false);
        let (output, _) = model.forward_batch(&mut graph, &requests, false)?;
        let logits = graph.value(output)?;
        let rows = logits
            .chunks(spec.values.len())
            .map(|row| softmax(&row.iter().copied().map(f64::from).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>>>()?;
        probabilities.push(rows.into_iter().flatten().collect::<Vec<_>>());
    }
    let probability_error = probabilities[0]
        .iter()
        .zip(&probabilities[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let reduction_roundoff =
        (2 * spec.model.width + spec.values.len()) as f64 * (f32::EPSILON as f64 / 2.);
    let budget = reduction_roundoff / (1. - reduction_roundoff);
    let result = json::json!({"model":spec.model,"parameter_vjp":spec.parameter_vjp,
        "steps":value.metadata.steps,"loss":value.metadata.loss,"reload_weights_max_abs_error":reload_error,
        "resume_weights_max_abs_error":resume_error,
        "resume_probability_max_abs_error":probability_error,"probability_comparison_budget":budget,
        "resumed_loss":resumed.metadata.loss,"parameter_rng_equal":true});
    std::fs::write(out.join("result.json"), json::to_vec_pretty(&result)?)?;
    ensure(
        probability_error <= budget
            && value.metadata.loss.is_finite()
            && resumed.metadata.loss.is_finite(),
        "native resumed output differs beyond FP32 readout reduction budget",
    )?;
    println!("{result}");
    Ok(())
}
