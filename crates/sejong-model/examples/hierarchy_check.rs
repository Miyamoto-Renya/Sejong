//! 구성 가능한 계층 모델의 인과성, 미분, 스트리밍과 native value 저장 경로를 GPU에서 확인한다.
use sejong_core::{
    DType, ExpertConfig, GpuConfig, HierarchyConfig, ModelConfig, Result, Rng, ensure, json,
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::{
    Model,
    native::value::{FORMAT, NativeValue, ValueMetadata},
};
use std::time::{SystemTime, UNIX_EPOCH};

fn config() -> ModelConfig {
    ModelConfig {
        id: "hierarchy-contract".into(),
        width: 32,
        layers: 2,
        heads: 4,
        kv_heads: Some(2),
        ffn_width: 64,
        context: 64,
        precision: DType::F32,
        norm_epsilon: 1e-5,
        seed: 117,
        hierarchy: Some(HierarchyConfig {
            patch_size: 3,
            encoder_width: 16,
            encoder_layers: 1,
            encoder_heads: 2,
            encoder_ffn_width: 32,
            decoder_width: 24,
            decoder_layers: 1,
            decoder_heads: 3,
            decoder_ffn_width: 48,
            local_window: 5,
            checkpoint_encoder: false,
            checkpoint_decoder: false,
        }),
        experts: Some(ExpertConfig {
            count: 4,
            top_k: 2,
            ffn_width: 16,
        }),
        ..ModelConfig::default()
    }
}
fn error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
fn logits(e: &mut Engine, m: &Model, ids: &[i32], sequence: usize) -> Result<Vec<f32>> {
    let ids = e.from_i32(&[ids.len()], ids)?;
    let mut g = Graph::new(e, false);
    let b = m.bind(&mut g, false)?;
    let y = m.logits(&mut g, &b, &ids, sequence)?;
    g.value(y)
}
fn loss(e: &mut Engine, m: &Model, ids: &[i32], targets: &[i32], sequence: usize) -> Result<f64> {
    let ids = e.from_i32(&[ids.len()], ids)?;
    let targets = e.from_i32(&[targets.len()], targets)?;
    let mut g = Graph::new(e, false);
    let b = m.bind(&mut g, false)?;
    let y = m.logits(&mut g, &b, &ids, sequence)?;
    let loss = g.cross_entropy(y, &targets)?;
    let values = g.value(loss)?;
    Ok(values.iter().map(|x| f64::from(*x)).sum::<f64>() / values.len() as f64)
}
fn main() -> Result<()> {
    let mut e = Engine::new(GpuConfig::default())?;
    let model = Model::initialize(&mut e, config())?;
    let sequence = 11;
    let tokens: Vec<i32> = (0..2 * sequence).map(|i| (31 + i * 7) as i32).collect();
    let targets: Vec<i32> = tokens.iter().map(|x| (x + 1) % 257).collect();
    let expected = logits(&mut e, &model, &tokens, sequence)?;
    let mut separate = Vec::new();
    for row in tokens.chunks(sequence) {
        separate.extend(logits(&mut e, &model, row, sequence)?);
    }
    let batch_error = error(&expected, &separate);
    ensure(
        batch_error < 2e-4,
        format!("hierarchy batch separation error {batch_error}"),
    )?;
    let mut changed = tokens.clone();
    for row in 0..2 {
        for i in 6..sequence {
            changed[row * sequence + i] = 256 - changed[row * sequence + i];
        }
    }
    let modified = logits(&mut e, &model, &changed, sequence)?;
    let mut causal_error = 0f32;
    for row in 0..2 {
        let start = row * sequence * 257;
        causal_error = causal_error.max(error(
            &expected[start..start + 6 * 257],
            &modified[start..start + 6 * 257],
        ));
    }
    ensure(
        causal_error < 1e-6,
        format!("suffix changed prefix logits by {causal_error}"),
    )?;
    let weights = model.inference_weights(&mut e)?;
    let mut cache_error = 0f32;
    let mut cache = model.new_cache(&mut e)?;
    let mut cache_launches = Vec::new();
    let chunk_patterns = vec![
        vec![1; sequence],
        vec![1, 4, 2, 4],
        vec![sequence],
        vec![4, 1, 1, 1, 4],
        vec![8, 1, 2],
        vec![6, 1, 4],
    ];
    for chunks in &chunk_patterns {
        let counters = e.execution_counters();
        cache.length = 0;
        let mut start = 0;
        let mut result = Vec::new();
        for &count in chunks {
            let hidden = model.hidden_cached(
                &mut e,
                &weights,
                &tokens[start..start + count],
                &mut cache,
                None,
            )?;
            let hidden = e.cast(&hidden, DType::F32)?;
            let output = e.matmul(&hidden, &weights["output"], false, false, DType::F32)?;
            result.extend(e.to_f32(&output)?);
            start += count;
        }
        cache_error = cache_error.max(error(&expected[..sequence * 257], &result));
        cache_launches.push(e.execution_counters().difference(&counters).launches);
    }
    ensure(
        cache_error < 2e-4,
        format!("hierarchy streaming mismatch {cache_error}"),
    )?;
    let ids = e.from_i32(&[tokens.len()], &tokens)?;
    let target = e.from_i32(&[targets.len()], &targets)?;
    let mut g = Graph::new(&mut e, true);
    let binding = model.bind(&mut g, true)?;
    let y = model.logits(&mut g, &binding, &ids, sequence)?;
    let objective = g.cross_entropy(y, &target)?;
    g.backward(&[(objective, 1.)])?;
    let grads = binding
        .leaves
        .iter()
        .map(|v| g.gradient(*v).cloned())
        .collect::<Vec<_>>();
    drop(g);
    let mut rng = Rng::new(208);
    let mut originals = Vec::new();
    let mut directions = Vec::new();
    let mut analytical = 0f64;
    for (p, grad) in model.parameters.iter().zip(grads) {
        let values = e.to_f32(&p.value)?;
        let direction = (0..values.len())
            .map(|_| rng.normal() / (model.parameter_count() as f32).sqrt())
            .collect::<Vec<_>>();
        if let Some(grad) = grad {
            analytical += e
                .to_f32(&grad)?
                .iter()
                .zip(&direction)
                .map(|(g, d)| f64::from(*g) * f64::from(*d))
                .sum::<f64>();
        }
        originals.push(values);
        directions.push(direction);
    }
    let epsilon = 0.02f32;
    let mut objectives = Vec::new();
    for sign in [-1., 1.] {
        for ((p, w), direction) in model.parameters.iter().zip(&originals).zip(&directions) {
            let changed = w
                .iter()
                .zip(direction)
                .map(|(w, d)| w + sign * epsilon * d)
                .collect::<Vec<_>>();
            let changed = e.from_f32(&p.value.shape, &changed)?;
            e.copy_into(&changed, &p.value)?;
        }
        objectives.push(loss(&mut e, &model, &tokens, &targets, sequence)?);
    }
    for (p, w) in model.parameters.iter().zip(&originals) {
        let original = e.from_f32(&p.value.shape, w)?;
        e.copy_into(&original, &p.value)?;
    }
    let numerical = (objectives[1] - objectives[0]) / (2. * f64::from(epsilon));
    let derivative_error = (analytical - numerical).abs();
    ensure(
        derivative_error < 0.002 + 0.08 * analytical.abs(),
        format!("hierarchy derivative analytical={analytical}, numerical={numerical}"),
    )?;
    let mut half_config = config();
    half_config.precision = DType::F16;
    let half = Model::initialize(&mut e, half_config)?;
    let half_expected = logits(&mut e, &half, &tokens[..sequence], sequence)?;
    ensure(
        half_expected.iter().all(|v| v.is_finite()),
        "F16 hierarchy logits must be finite",
    )?;
    let half_weights = half.inference_weights(&mut e)?;
    let mut half_cache = half.new_cache(&mut e)?;
    let mut half_cache_error = 0f32;
    for chunks in &chunk_patterns {
        half_cache.reset();
        let mut half_cached = Vec::new();
        let mut start = 0;
        for &count in chunks {
            let hidden = half.hidden_cached(
                &mut e,
                &half_weights,
                &tokens[start..start + count],
                &mut half_cache,
                None,
            )?;
            let hidden = e.cast(&hidden, DType::F32)?;
            let output = e.matmul(&hidden, &half_weights["output"], false, false, DType::F32)?;
            half_cached.extend(e.to_f32(&output)?);
            start += count;
        }
        half_cache_error = half_cache_error.max(error(&half_expected, &half_cached));
    }
    ensure(
        half_cache_error < 0.03,
        format!("F16 hierarchy cache error {half_cache_error}"),
    )?;
    let mut g = Graph::new(&mut e, true);
    let binding = half.bind(&mut g, true)?;
    let y = half.logits(&mut g, &binding, &ids, sequence)?;
    let objective = g.cross_entropy(y, &target)?;
    g.backward(&[(objective, 1.)])?;
    let half_gradients = binding
        .leaves
        .iter()
        .filter_map(|v| g.gradient(*v).cloned())
        .collect::<Vec<_>>();
    drop(g);
    for gradient in &half_gradients {
        ensure(
            e.to_f32(gradient)?.iter().all(|v| v.is_finite()),
            "F16 hierarchy gradients must be finite",
        )?;
    }
    let root = std::env::temp_dir().join(format!(
        "sejong-hierarchy-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    model.save(&mut e, &root.join("model"), model.manifest(0), None)?;
    let (loaded, _) = Model::load(&mut e, &root.join("model"))?;
    let reload_error = error(&expected, &logits(&mut e, &loaded, &tokens, sequence)?);
    ensure(
        reload_error < 1e-6,
        format!("hierarchy artifact reload mismatch {reload_error}"),
    )?;
    let mut native = NativeValue::initialize(
        &mut e,
        ValueMetadata {
            format: FORMAT.into(),
            requests: vec!["first".into(), "second".into()],
            prompts: vec!["abcde".into(), "abcdefghi".into()],
            values: vec!["yes".into(), "no".into()],
            model: config(),
            steps: 0,
            source_hash: "hierarchy-native-contract".into(),
            loss: 0.,
        },
    )?;
    for _ in 0..2 {
        let target = e.from_i32(&[2], &[0, 1])?;
        let mut g = Graph::new(&mut e, true);
        let (y, leaves) = native.forward_batch(&mut g, &[0, 1], true)?;
        let objective = g.cross_entropy(y, &target)?;
        let losses = g.value(objective)?;
        native.metadata.loss = losses.iter().map(|x| f64::from(*x)).sum::<f64>() / 2.;
        g.backward(&[(objective, 1.)])?;
        let gradients = leaves
            .iter()
            .map(|v| g.gradient(*v).cloned())
            .collect::<Vec<_>>();
        drop(g);
        for (p, gradient) in native.parameters().iter().zip(gradients) {
            if let Some(gradient) = gradient {
                let update = e.add(&p.value, &gradient, -0.01, DType::F32)?;
                e.copy_into(&update, &p.value)?;
            }
        }
        native.metadata.steps += 1;
    }
    native.seal(&mut e)?;
    let law = native.row(&mut e, 0)?;
    ensure(
        law.iter().all(|p| p.is_finite() && *p >= 0.)
            && (law.iter().sum::<f64>() - 1.).abs() < 1e-12,
        "native probability normalization mismatch",
    )?;
    native.save(&mut e, &root.join("native"))?;
    let reloaded = NativeValue::load(&mut e, &root.join("native"))?;
    let restored = reloaded.row(&mut e, 0)?;
    let native_reload_error = law
        .iter()
        .zip(restored)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    ensure(
        native_reload_error < 1e-12,
        "native hierarchy reload law mismatch",
    )?;
    std::fs::remove_dir_all(root)?;
    println!(
        "{}",
        json::to_string_pretty(&json::json!({
            "scope":"GPU configured hierarchy/GQA/MoE model contract", "parameters":model.parameter_count(),
            "batch":2,"sequence":sequence,"patch":3,"window":5,"batch_max_error":batch_error,
            "causal_suffix_max_error":causal_error,"streaming_max_error":cache_error,
            "streaming_chunk_patterns":chunk_patterns,"streaming_launches":cache_launches,
            "gradient_analytical":analytical,"gradient_numerical":numerical,"gradient_error":derivative_error,
        "f16_streaming_max_error":half_cache_error,"f16_gradient_tensors":half_gradients.len(),
            "artifact_reload_max_error":reload_error,"native_steps":native.metadata.steps,
            "native_reload_law_error":native_reload_error,"native_law":law
        }))?
    );
    Ok(())
}
