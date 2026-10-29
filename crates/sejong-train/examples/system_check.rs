//! Integrated GPU model check: backward, KV caching, precision and artifact reload.
use sejong_core::{DType, GpuConfig, ModelConfig, Result, Rng, ensure};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::Model;
fn logits(engine: &mut Engine, model: &Model, tokens: &[i32]) -> Result<Vec<f32>> {
    let ids = engine.from_i32(&[tokens.len()], tokens)?;
    let mut g = Graph::new(engine, false);
    let b = model.bind(&mut g, false)?;
    let y = model.logits(&mut g, &b, &ids, tokens.len())?;
    g.value(y)
}
fn loss(engine: &mut Engine, model: &Model, tokens: &[i32], targets: &[i32]) -> Result<f64> {
    let ids = engine.from_i32(&[tokens.len()], tokens)?;
    let target = engine.from_i32(&[targets.len()], targets)?;
    let mut g = Graph::new(engine, false);
    let b = model.bind(&mut g, false)?;
    let y = model.logits(&mut g, &b, &ids, tokens.len())?;
    let l = g.cross_entropy(y, &target)?;
    let v = g.value(l)?;
    Ok(v.iter().map(|v| *v as f64).sum::<f64>() / v.len() as f64)
}
fn main() -> Result<()> {
    let mut e = Engine::new(GpuConfig::default())?;
    let c = ModelConfig {
        id: "system-check".into(),
        width: 32,
        layers: 2,
        heads: 2,
        ffn_width: 64,
        context: 64,
        precision: DType::F32,
        seed: 97,
        ..ModelConfig::default()
    };
    let model = Model::initialize(&mut e, c.clone())?;
    let tokens = vec![256, 83, 101, 106, 111, 110, 103, 32, 234, 184, 128];
    let targets = vec![83, 101, 106, 111, 110, 103, 32, 234, 184, 128, 46];
    let reference = logits(&mut e, &model, &tokens)?;
    let weights = model.inference_weights(&mut e)?;
    let mut cache = model.new_cache(&mut e)?;
    let mut max_cache_error = 0f32;
    for end in 1..=tokens.len() {
        let cached = model.forward_cached(&mut e, &weights, &tokens[end - 1..end], &mut cache)?;
        let got = e.to_f32(&cached)?;
        for (a, b) in got.iter().zip(&reference[(end - 1) * 257..end * 257]) {
            max_cache_error = max_cache_error.max((a - b).abs());
        }
    }
    ensure(
        max_cache_error < 1e-4,
        format!("KV cache mismatch {max_cache_error}"),
    )?;
    let ids = e.from_i32(&[tokens.len()], &tokens)?;
    let target = e.from_i32(&[targets.len()], &targets)?;
    let mut g = Graph::new(&mut e, true);
    let binding = model.bind(&mut g, true)?;
    let y = model.logits(&mut g, &binding, &ids, tokens.len())?;
    let l = g.cross_entropy(y, &target)?;
    g.backward(&[(l, 1.)])?;
    let gradients: Vec<_> = binding
        .leaves
        .iter()
        .map(|v| g.gradient(*v).unwrap().clone())
        .collect();
    drop(g);
    let mut rng = Rng::new(112);
    let mut analytical = 0f64;
    let mut original = vec![];
    let mut directions = vec![];
    for (p, gradient) in model.parameters.iter().zip(gradients) {
        let w = e.to_f32(&p.value)?;
        let d: Vec<f32> = (0..w.len())
            .map(|_| rng.normal() / (model.parameter_count() as f32).sqrt())
            .collect();
        let grad = e.to_f32(&gradient)?;
        analytical += grad
            .iter()
            .zip(&d)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum::<f64>();
        original.push(w);
        directions.push(d);
    }
    let epsilon = 0.02f32;
    let mut values = vec![];
    for sign in [-1., 1.] {
        for ((p, w), d) in model.parameters.iter().zip(&original).zip(&directions) {
            let changed: Vec<f32> = w
                .iter()
                .zip(d)
                .map(|(w, d)| w + sign * epsilon * d)
                .collect();
            let t = e.from_f32(&p.value.shape, &changed)?;
            e.copy_into(&t, &p.value)?;
        }
        values.push(loss(&mut e, &model, &tokens, &targets)?);
    }
    for (p, w) in model.parameters.iter().zip(&original) {
        let t = e.from_f32(&p.value.shape, w)?;
        e.copy_into(&t, &p.value)?;
    }
    let numerical = (values[1] - values[0]) / (2. * epsilon as f64);
    let error = (numerical - analytical).abs();
    ensure(
        error < 0.001 + 0.05 * analytical.abs(),
        format!("backward directional derivative: analytic={analytical}, numerical={numerical}"),
    )?;
    let mut half_config = c;
    half_config.precision = DType::F16;
    let half = Model::initialize(&mut e, half_config)?;
    let half_logits = logits(&mut e, &half, &tokens)?;
    let max_half_error = reference
        .iter()
        .zip(half_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure(
        max_half_error < 0.04,
        format!("mixed precision error {max_half_error}"),
    )?;
    let destination = sejong_core::root().join(".cache/system-check/roundtrip");
    if destination.exists() {
        std::fs::remove_dir_all(&destination)?;
    }
    model.save(&mut e, &destination, model.manifest(0), None)?;
    let (reloaded, _) = Model::load(&mut e, &destination)?;
    let roundtrip = logits(&mut e, &reloaded, &tokens)?;
    ensure(
        reference == roundtrip,
        "artifact reload changed model logits",
    )?;
    println!(
        "Integrated model check: cache_error={max_cache_error:.8}, gradient_analytic={analytical:.7}, gradient_numerical={numerical:.7}, fp16_error={max_half_error:.6}, artifact_reload=exact"
    );
    let mut optimizer = sejong_train::Adam::new(&mut e, &reloaded.parameters)?;
    let config = sejong_core::TrainConfig {
        clip_norm: 1000.,
        ..sejong_core::TrainConfig::default()
    };
    let before = e.to_f32(&reloaded.parameters[0].value)?[0];
    let gradients = reloaded
        .parameters
        .iter()
        .map(|p| e.full(&p.value.shape, DType::F32, 0.25).map(Some))
        .collect::<Result<Vec<_>>>()?;
    optimizer.accumulate(&mut e, gradients)?;
    optimizer.update(&mut e, &reloaded.parameters, &config, 0.001)?;
    optimizer.update(&mut e, &reloaded.parameters, &config, 0.001)?;
    let mut expected = before;
    let mut m = 0f32;
    let mut v = 0f32;
    for (step, gradient) in [0.25f32, 0.].into_iter().enumerate() {
        m = config.beta1 * m + (1. - config.beta1) * gradient;
        v = config.beta2 * v + (1. - config.beta2) * gradient * gradient;
        expected = expected * (1. - 0.001 * config.weight_decay)
            - 0.001 * (m / (1. - config.beta1.powi(step as i32 + 1)))
                / ((v / (1. - config.beta2.powi(step as i32 + 1))).sqrt() + config.epsilon);
    }
    let actual = e.to_f32(&reloaded.parameters[0].value)?[0];
    ensure(
        (actual - expected).abs() < 2e-7,
        format!("AdamW zero-gradient transition mismatch {actual} vs {expected}"),
    )?;
    println!(
        "AdamW across an inactive update: error={:.9}",
        (actual - expected).abs()
    );
    Ok(())
}
