//! 같은 NativeValue batch의 보존·checkpoint gradient와 표집 상태를 대조한다.
use sejong_core::{GpuConfig, Result, Rng, ensure, json, native::ValueTraining};
use sejong_gpu::{Engine, graph::Graph, parameter_vjp::GradientMaterializer};
use sejong_model::native::value::{FORMAT, NativeValue, ValueMetadata};
use std::path::Path;

struct Sample {
    gradients: Vec<Option<Vec<f32>>>,
    details: json::Value,
}
fn sample(
    e: &mut Engine,
    model: &NativeValue,
    spec: &ValueTraining,
    indices: &[usize],
    checkpoint: bool,
    compressed: bool,
) -> Result<Sample> {
    let label = e.from_i32(
        &[indices.len()],
        &indices
            .iter()
            .map(|&i| spec.labels[i] as i32)
            .collect::<Vec<_>>(),
    )?;
    let mut g = Graph::new(e, true);
    g.checkpointing = checkpoint;
    if compressed {
        let policy = spec.parameter_vjp.as_ref().unwrap();
        g.configure_parameter_vjp(policy.clone(), Rng::new(policy.seed))?;
    }
    let (logits, leaves) = model.forward_batch(&mut g, indices, true)?;
    let loss = g.cross_entropy(logits, &label)?;
    let values = g.value(loss)?;
    g.backward(&[(loss, 1.)])?;
    let stats = g.parameter_vjp_statistics();
    let details = json::json!({"checkpoint":checkpoint,"compressed":compressed,
        "loss": values.iter().map(|&x| x as f64).sum::<f64>()/values.len() as f64,
        "parameter_rng":g.parameter_vjp_rng().map(|r|r.state),
        "compressed_matrices":stats.compressed_matrices,"dense_matrices":stats.dense_matrices,
        "basis_groups":stats.basis_groups,"projection_groups":stats.projection_groups});
    let tensors = if compressed {
        let gradients = leaves
            .iter()
            .map(|&v| g.parameter_gradient(v))
            .collect::<Vec<_>>();
        drop(g);
        let materializer = GradientMaterializer::new(
            e,
            &model
                .parameters()
                .iter()
                .map(|p| p.value.clone())
                .collect::<Vec<_>>(),
        )?;
        materializer.materialize(e, &gradients)?
    } else {
        let tensors = leaves
            .iter()
            .map(|&v| g.gradient(v).cloned())
            .collect::<Vec<_>>();
        drop(g);
        tensors
    };
    let gradients = tensors
        .iter()
        .map(|t| t.as_ref().map(|t| e.to_f32(t)).transpose())
        .collect::<Result<Vec<_>>>()?;
    Ok(Sample { gradients, details })
}
fn compare(model: &NativeValue, a: &Sample, b: &Sample) -> json::Value {
    let mut squared = 0.;
    let mut base = 0.;
    let mut maximum = 0f64;
    let mut changes = Vec::new();
    for ((p, x), y) in model
        .parameters()
        .iter()
        .zip(&a.gradients)
        .zip(&b.gradients)
    {
        let mut local = 0f64;
        let mut local_sq = 0.;
        let mut local_base = 0.;
        for i in 0..p.value.len() {
            let x = x.as_ref().map_or(0., |x| x[i]) as f64;
            let y = y.as_ref().map_or(0., |y| y[i]) as f64;
            local = local.max((x - y).abs());
            local_sq += (x - y).powi(2);
            local_base += x * x;
        }
        maximum = maximum.max(local);
        squared += local_sq;
        base += local_base;
        if local > 0. {
            changes.push(json::json!({"name":p.name,"max_abs":local,
            "relative_l2":local_sq.sqrt()/local_base.sqrt().max(f64::MIN_POSITIVE)}));
        }
    }
    json::json!({"max_abs":maximum,"relative_l2":squared.sqrt()/base.sqrt().max(f64::MIN_POSITIVE),
        "changed_parameters":changes})
}
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure(
        args.len() == 2,
        "usage: native_gradient_equivalence SPEC OUTPUT_JSON",
    )?;
    let spec: ValueTraining = json::from_slice(&std::fs::read(&args[0])?)?;
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
            source_hash: "gradient-comparison".into(),
            loss: 0.,
        },
    )?;
    let batch = spec
        .batch
        .max_examples
        .unwrap_or(spec.requests.len())
        .min(spec.requests.len());
    let mut rng = Rng::new(spec.model.seed ^ 0x6e6174697665);
    let indices = (0..batch)
        .map(|_| rng.index(spec.requests.len()))
        .collect::<Vec<_>>();
    let dense_checkpoint = sample(&mut e, &model, &spec, &indices, true, false)?;
    let dense_repeat = sample(&mut e, &model, &spec, &indices, true, false)?;
    let dense_retained = sample(&mut e, &model, &spec, &indices, false, false)?;
    let mut result = json::json!({"batch":batch,"parameters":model.parameters().iter().map(|p|p.value.len()).sum::<usize>(),
        "dense_checkpoint":dense_checkpoint.details,"dense_retained":dense_retained.details,
        "dense_checkpoint_repeat":compare(&model,&dense_checkpoint,&dense_repeat),
        "dense_checkpoint_vs_retained":compare(&model,&dense_checkpoint,&dense_retained)});
    drop(dense_checkpoint);
    drop(dense_repeat);
    drop(dense_retained);
    if spec.parameter_vjp.is_some() {
        let checkpoint = sample(&mut e, &model, &spec, &indices, true, true)?;
        let repeat = sample(&mut e, &model, &spec, &indices, true, true)?;
        let retained = sample(&mut e, &model, &spec, &indices, false, true)?;
        result["compressed_checkpoint"] = checkpoint.details.clone();
        result["compressed_retained"] = retained.details.clone();
        result["compressed_checkpoint_repeat"] = compare(&model, &checkpoint, &repeat);
        result["compressed_checkpoint_vs_retained"] = compare(&model, &checkpoint, &retained);
    }
    std::fs::write(Path::new(&args[1]), json::to_vec_pretty(&result)?)?;
    println!("{}", result);
    Ok(())
}
