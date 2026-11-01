//! 공유 인코더 관측, 제어 배치 보정, 지연 저랭크 SGD와 원자적 체크포인트.
use crate::{Corpus, TrainingSummary, save_execution_metrics};
use sejong_core::{
    Config, ControlSampling, DType, Error, Manifest, OptimizerKind, Result, Rng, TrainConfig,
    ensure, json,
};
use sejong_gpu::{Engine, Tensor, graph::Graph};
use sejong_model::{LowRankCorrection, Model, load_tensors};
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};

struct PredictorBlock {
    index: usize,
    left: Tensor,
    right: Tensor,
    left_gram: Tensor,
    right_gram: Tensor,
    deferred: Tensor,
    predictor: Tensor,
}

/// θ, 예측기와 basis는 한 step의 모든 관측과 제어 미분 동안 고정한다.
struct SgdState {
    step: usize,
    blocks: Vec<PredictorBlock>,
    sampling_rng: Rng,
    basis_rng: Rng,
}

fn predicted_parameter(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name != "embedding"
        && name != "global_bos"
        && !name.starts_with("encoder.")
        && name != "input_projection"
        && !name.ends_with(".router")
        && !name.contains(".experts.")
        && !name.contains(".expert.")
}

fn rng_tensor(engine: &mut Engine, rng: &Rng) -> Result<Tensor> {
    let words: Vec<f32> = (0..4)
        .map(|i| ((rng.state >> (i * 16)) & 0xffff) as f32)
        .collect();
    engine.from_f32(&[4], &words)
}

fn read_rng(engine: &mut Engine, state: &BTreeMap<String, Tensor>, key: &str) -> Result<Rng> {
    let tensor = state
        .get(key)
        .ok_or_else(|| Error::Message(format!("missing optimizer state {key}")))?;
    ensure(tensor.shape == [4], "RNG state requires four 16-bit words")?;
    let words = engine.to_f32(tensor)?;
    let mut value = 0u64;
    for (i, word) in words.into_iter().enumerate() {
        ensure(
            word.is_finite() && (0.0..=65535.0).contains(&word) && word.fract() == 0.,
            "invalid optimizer RNG word",
        )?;
        value |= (word as u64) << (i * 16);
    }
    Ok(Rng { state: value })
}

impl SgdState {
    fn new(engine: &mut Engine, model: &Model, config: &TrainConfig) -> Result<Self> {
        let mut state = Self {
            step: 0,
            blocks: vec![],
            sampling_rng: Rng::new(config.seed ^ 0x38b8_7c4d_9f0a_1265),
            basis_rng: Rng::new(config.seed ^ 0xa412_39c5_07ef_68bd),
        };
        if config.optimizer == OptimizerKind::ControlVariateSgd {
            let rank = config.control_variate.as_ref().unwrap().rank;
            for (index, parameter) in model.parameters.iter().enumerate() {
                if !predicted_parameter(&parameter.name, &parameter.value.shape) {
                    continue;
                }
                let [a, b] = parameter.value.shape[..] else {
                    unreachable!()
                };
                ensure(
                    rank <= a.min(b),
                    format!(
                        "predictor rank {rank} exceeds matrix dimensions of {}",
                        parameter.name
                    ),
                )?;
                let left_values: Vec<f32> =
                    (0..a * rank).map(|_| state.basis_rng.normal()).collect();
                let right_values: Vec<f32> =
                    (0..b * rank).map(|_| state.basis_rng.normal()).collect();
                let left = engine.from_f32(&[a, rank], &left_values)?;
                let right = engine.from_f32(&[b, rank], &right_values)?;
                let left = engine.orthonormalize_columns(&left)?;
                let right = engine.orthonormalize_columns(&right)?;
                let left_gram = engine.matmul(&left, &left, true, false, DType::F32)?;
                let right_gram = engine.matmul(&right, &right, true, false, DType::F32)?;
                state.blocks.push(PredictorBlock {
                    index,
                    left,
                    right,
                    left_gram,
                    right_gram,
                    deferred: engine.full(&[rank, rank], DType::F32, 0.)?,
                    predictor: engine.full(
                        &[3 * model.encoder_width() + 1, rank * rank],
                        DType::F32,
                        0.,
                    )?,
                });
            }
            ensure(
                !state.blocks.is_empty(),
                "control-variate training requires a predicted matrix",
            )?;
        }
        Ok(state)
    }

    fn load(
        engine: &mut Engine,
        model: &Model,
        config: &TrainConfig,
        directory: &Path,
        manifest: &Manifest,
    ) -> Result<Self> {
        let filename = manifest
            .optimizer
            .as_ref()
            .ok_or_else(|| Error::Message("resume requires SGD optimizer state".into()))?;
        let tensors = load_tensors(engine, &directory.join(filename))?;
        let mut state = Self {
            step: manifest.step,
            blocks: vec![],
            sampling_rng: read_rng(engine, &tensors, "sgd.sampling_rng")?,
            basis_rng: read_rng(engine, &tensors, "sgd.basis_rng")?,
        };
        let version = tensors
            .get("sgd.version")
            .ok_or_else(|| Error::Message("SGD state version missing".into()))?;
        ensure(
            version.shape == [1] && engine.to_f32(version)? == [1.],
            "unsupported SGD state version",
        )?;
        if config.optimizer == OptimizerKind::ControlVariateSgd {
            let rank = config.control_variate.as_ref().unwrap().rank;
            for (index, parameter) in model.parameters.iter().enumerate() {
                if !predicted_parameter(&parameter.name, &parameter.value.shape) {
                    continue;
                }
                let [a, b] = parameter.value.shape[..] else {
                    unreachable!()
                };
                let read = |suffix: &str, shape: &[usize]| -> Result<Tensor> {
                    let key = format!("cv.{}.{suffix}", parameter.name);
                    let tensor = tensors
                        .get(&key)
                        .ok_or_else(|| Error::Message(format!("missing optimizer tensor {key}")))?;
                    ensure(
                        tensor.shape == shape && tensor.dtype == DType::F32,
                        format!("optimizer tensor shape/dtype mismatch: {key}"),
                    )?;
                    Ok(tensor.clone())
                };
                let left = read("left", &[a, rank])?;
                let right = read("right", &[b, rank])?;
                let left_gram = engine.matmul(&left, &left, true, false, DType::F32)?;
                let right_gram = engine.matmul(&right, &right, true, false, DType::F32)?;
                state.blocks.push(PredictorBlock {
                    index,
                    left,
                    right,
                    left_gram,
                    right_gram,
                    deferred: read("deferred", &[rank, rank])?,
                    predictor: read("predictor", &[3 * model.encoder_width() + 1, rank * rank])?,
                });
            }
        }
        ensure(
            tensors.len() == 3 + 4 * state.blocks.len(),
            "SGD checkpoint tensor count differs from its configuration",
        )?;
        Ok(state)
    }

    fn corrections(&self) -> Vec<LowRankCorrection> {
        self.blocks
            .iter()
            .map(|b| LowRankCorrection {
                parameter_index: b.index,
                left: b.left.clone(),
                coefficient: b.deferred.clone(),
                right: b.right.clone(),
            })
            .collect()
    }

    fn tensors(&self, engine: &mut Engine, model: &Model) -> Result<BTreeMap<String, Tensor>> {
        let mut out = BTreeMap::new();
        out.insert("sgd.version".into(), engine.from_f32(&[1], &[1.])?);
        out.insert(
            "sgd.sampling_rng".into(),
            rng_tensor(engine, &self.sampling_rng)?,
        );
        out.insert("sgd.basis_rng".into(), rng_tensor(engine, &self.basis_rng)?);
        for block in &self.blocks {
            for (suffix, tensor) in [
                ("left", &block.left),
                ("right", &block.right),
                ("deferred", &block.deferred),
                ("predictor", &block.predictor),
            ] {
                out.insert(
                    format!("cv.{}.{suffix}", model.parameters[block.index].name),
                    tensor.clone(),
                );
            }
        }
        Ok(out)
    }

    /// 체크포인트 경계에서 θ_base += U A Vᵀ, A = 0을 함께 반영한다.
    fn flush(&mut self, engine: &mut Engine, model: &Model) -> Result<()> {
        for block in &mut self.blocks {
            let left = engine.matmul(&block.left, &block.deferred, false, false, DType::F32)?;
            let correction = engine.matmul(&left, &block.right, false, true, DType::F32)?;
            let base = &model.parameters[block.index].value;
            engine.add_into(base, &correction, base, 1.)?;
            let zero = engine.full(&block.deferred.shape, DType::F32, 0.)?;
            engine.copy_into(&zero, &block.deferred)?;
        }
        Ok(())
    }

    fn refresh_bases(
        &mut self,
        engine: &mut Engine,
        model: &Model,
        sketches: &[Option<(Tensor, Tensor)>],
    ) -> Result<()> {
        self.flush(engine, model)?;
        for (block, sketch) in self.blocks.iter_mut().zip(sketches) {
            let Some((left_sketch, right_sketch)) = sketch else {
                continue;
            };
            let magnitude = (engine.squared_norm(left_sketch)?
                + engine.squared_norm(right_sketch)?)
            .sqrt() as f32;
            if magnitude == 0. {
                continue;
            }
            let regularization = magnitude * f32::EPSILON.sqrt();
            let sketch = engine.add(left_sketch, &block.left, regularization, DType::F32)?;
            let left = engine.orthonormalize_columns(&sketch)?;
            let sketch = engine.add(right_sketch, &block.right, regularization, DType::F32)?;
            let right = engine.orthonormalize_columns(&sketch)?;
            // 각 feature의 C를 U_newᵀ U_old C V_oldᵀ V_new로 옮긴다.
            let left_overlap = engine.matmul(&block.left, &left, true, false, DType::F32)?;
            let right_overlap = engine.matmul(&block.right, &right, true, false, DType::F32)?;
            block.predictor =
                engine.transport_coefficients(&block.predictor, &left_overlap, &right_overlap)?;
            block.left_gram = engine.matmul(&left, &left, true, false, DType::F32)?;
            block.right_gram = engine.matmul(&right, &right, true, false, DType::F32)?;
            block.left = left;
            block.right = right;
        }
        Ok(())
    }
}

struct Observation {
    input: Vec<i32>,
    target: Vec<i32>,
}

fn entropy(input: &[i32]) -> f64 {
    let mut counts = [0usize; 257];
    for &value in input {
        counts[value as usize] += 1;
    }
    counts
        .into_iter()
        .filter(|&n| n > 0)
        .map(|n| {
            let p = n as f64 / input.len() as f64;
            -p * p.ln()
        })
        .sum()
}

/// 가중치는 층별 N_h / (M m_h)이며 선택 전 입력으로 strata를 고정한다.
fn control_weights(observations: &[Observation], config: &TrainConfig, rng: &mut Rng) -> Vec<f32> {
    let count = observations.len();
    if config.optimizer == OptimizerKind::Sgd {
        return vec![1. / count as f32; count];
    }
    let cv = config.control_variate.as_ref().unwrap();
    let strata = if cv.sampling == ControlSampling::Uniform {
        1
    } else {
        cv.strata
    };
    let mut order: Vec<usize> = (0..count).collect();
    if strata > 1 {
        order.sort_by(|&a, &b| {
            entropy(&observations[a].input)
                .total_cmp(&entropy(&observations[b].input))
                .then(a.cmp(&b))
        });
    }
    let mut groups: Vec<Vec<usize>> = (0..strata)
        .map(|h| order[h * count / strata..(h + 1) * count / strata].to_vec())
        .collect();
    let mut allocations = vec![1usize; strata];
    for _ in strata..cv.controls {
        let h = (0..strata)
            .filter(|&h| allocations[h] < groups[h].len())
            .max_by(|&a, &b| {
                (groups[a].len() as f64 / (allocations[a] + 1) as f64)
                    .total_cmp(&(groups[b].len() as f64 / (allocations[b] + 1) as f64))
                    .then(b.cmp(&a))
            })
            .expect("validated control count fits strata");
        allocations[h] += 1;
    }
    let mut weights = vec![0.; count];
    for (group, &selected) in groups.iter_mut().zip(&allocations) {
        let weight = group.len() as f32 / (count * selected) as f32;
        for i in 0..selected {
            let j = i + rng.index(group.len() - i);
            group.swap(i, j);
            weights[group[i]] = weight;
        }
    }
    weights
}

fn accumulate(
    engine: &mut Engine,
    destination: &mut Option<Tensor>,
    incoming: &Tensor,
    weight: f32,
) -> Result<()> {
    if let Some(current) = destination {
        engine.add_into(current, incoming, current, weight)
    } else {
        *destination = Some(engine.scale(incoming, weight)?);
        Ok(())
    }
}

fn correction_norm(
    engine: &mut Engine,
    block: &PredictorBlock,
    coefficient: &Tensor,
) -> Result<f64> {
    let product = engine.matmul(&block.left_gram, coefficient, false, false, DType::F32)?;
    let product = engine.matmul(&product, &block.right_gram, false, false, DType::F32)?;
    engine.dot(coefficient, &product)
}

fn augmented_feature(
    engine: &mut Engine,
    encoded: &Tensor,
    target_embedding: &Tensor,
) -> Result<Tensor> {
    let encoded = engine.cast(encoded, DType::F32)?;
    let feature = engine.row_mean(&encoded)?;
    let target = engine.row_mean(target_embedding)?;
    let interaction = engine.mul(&encoded, target_embedding)?;
    let interaction = engine.row_mean(&interaction)?;
    let one = engine.full(&[1, 1], DType::F32, 1.)?;
    engine.concat_columns(&[feature, target, interaction, one])
}

fn save(
    state: &mut SgdState,
    engine: &mut Engine,
    model: &Model,
    directory: &Path,
    config: &TrainConfig,
    corpus: &Corpus,
    rng: &Rng,
) -> Result<()> {
    state.flush(engine, model)?;
    engine.synchronize()?;
    let mut manifest = model.manifest(state.step);
    manifest.training = Some(config.clone());
    manifest.data_fingerprint = Some(corpus.fingerprint.clone());
    manifest.rng_state = Some(rng.state);
    let tensors = state.tensors(engine, model)?;
    model.save(engine, directory, manifest, Some(&tensors))
}

pub fn train(
    config: &Config,
    data: &Path,
    output: &Path,
    resume: Option<&Path>,
) -> Result<TrainingSummary> {
    let complete_started = Instant::now();
    config.model.validate()?;
    config.training.validate(&config.model)?;
    let tc = &config.training;
    ensure(
        matches!(
            tc.optimizer,
            OptimizerKind::Sgd | OptimizerKind::ControlVariateSgd
        ),
        "SGD route requires sgd or control_variate_sgd",
    )?;
    ensure(
        !output.exists(),
        format!("training destination already exists: {}", output.display()),
    )?;
    let corpus = Corpus::open(data)?;
    ensure(
        corpus.tokens > tc.sequence,
        "corpus is shorter than the training sequence",
    )?;
    let mut engine = Engine::new(config.gpu.clone())?;
    let (model, mut state, mut data_rng) = if let Some(directory) = resume {
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
        let mut after = json::to_value(tc)?;
        for field in ["checkpoint_every", "log_every"] {
            before.as_object_mut().unwrap().remove(field);
            after.as_object_mut().unwrap().remove(field);
        }
        ensure(
            before == after,
            "resume SGD, predictor, schedule or batch settings differ from checkpoint",
        )?;
        let state = SgdState::load(&mut engine, &model, tc, directory, &manifest)?;
        let rng = Rng {
            state: manifest
                .rng_state
                .ok_or_else(|| Error::Message("checkpoint corpus RNG state missing".into()))?,
        };
        (model, state, rng)
    } else {
        let model = Model::initialize(&mut engine, config.model.clone())?;
        let state = SgdState::new(&mut engine, &model, tc)?;
        (model, state, Rng::new(tc.seed))
    };
    ensure(
        state.step < tc.steps,
        "checkpoint already reached configured training steps",
    )?;
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("config.json"), json::to_vec_pretty(config)?)?;
    let mut log = std::fs::File::create(output.join("metrics.jsonl"))?;
    let started = Instant::now();
    let mut last_loss = 0.;
    while state.step < tc.steps {
        engine.synchronize()?;
        let before = engine.execution_counters();
        let step_started = Instant::now();
        let observations: Vec<Observation> = (0..tc.accumulation)
            .map(|_| {
                corpus
                    .batch(&mut data_rng, tc.batch, tc.sequence)
                    .map(|(input, target)| Observation { input, target })
            })
            .collect::<Result<_>>()?;
        let weights = control_weights(&observations, tc, &mut state.sampling_rng);
        let controls = weights.iter().filter(|&&w| w > 0.).count();
        let corrections = state.corrections();
        let mut gradients = vec![None; model.parameters.len()];
        let mut mean_predictions: Vec<Option<Tensor>> = vec![None; state.blocks.len()];
        let mut control_predictions: Vec<Option<Tensor>> = vec![None; state.blocks.len()];
        let mut mean_targets: Vec<Option<Tensor>> = vec![None; state.blocks.len()];
        let mut predictor_gradients: Vec<Option<Tensor>> = vec![None; state.blocks.len()];
        let mut loss = 0.;
        let mut gradient_second_moment = 0.;
        let mut residual_second_moment = 0.;
        let mut predictor_mse = 0.;
        let diagnostics = tc.gradient_diagnostics_every > 0
            && (state.step + 1).is_multiple_of(tc.gradient_diagnostics_every);
        let mut diagnostic_seconds = 0.;
        let refreshed = !state.blocks.is_empty()
            && (state.step + 1).is_multiple_of(tc.control_variate.as_ref().unwrap().basis_refresh);
        let mut refresh_sketches: Vec<Option<(Tensor, Tensor)>> = vec![None; state.blocks.len()];
        let mut control_number = 0;
        for (observation, &control_weight) in observations.iter().zip(&weights) {
            let control = control_weight > 0.;
            let ids = engine.from_i32(&[observation.input.len()], &observation.input)?;
            let mut graph = Graph::new(&mut engine, control);
            graph.checkpointing = tc.activation_checkpointing && control;
            let binding = if control {
                model.bind_with_corrections(&mut graph, true, &corrections)?
            } else {
                model.bind_encoder(&mut graph, false)?
            };
            let encoded = model.encode(&mut graph, &binding, &ids, tc.sequence, None)?;
            let feature_tensor = graph.tensor(encoded).clone();
            let mut current_gradients = vec![];
            let mut targets = vec![];
            if control {
                let labels = graph
                    .engine
                    .from_i32(&[observation.target.len()], &observation.target)?;
                let logits =
                    model.logits_from_encoded(&mut graph, &binding, encoded, tc.sequence)?;
                let objective = graph.cross_entropy(logits, &labels)?;
                let values = graph.value(objective)?;
                let value = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
                ensure(value.is_finite(), "nonfinite control loss")?;
                loss += control_weight as f64 * value;
                graph.backward(&[(objective, tc.loss_scale)])?;
                current_gradients = binding
                    .leaves
                    .iter()
                    .map(|&leaf| graph.gradient(leaf).cloned())
                    .collect();
                targets = state
                    .blocks
                    .iter()
                    .map(|block| {
                        graph
                            .projected_gradient(binding.leaves[block.index])
                            .cloned()
                    })
                    .collect();
            }
            drop(graph);
            let feature = if !state.blocks.is_empty() {
                let labels = engine.from_i32(&[observation.target.len()], &observation.target)?;
                let target_embedding =
                    engine.embedding(model.parameter("embedding"), &labels, DType::F32)?;
                Some(augmented_feature(
                    &mut engine,
                    &feature_tensor,
                    &target_embedding,
                )?)
            } else {
                None
            };
            let mut sample_norm = 0.;
            if control {
                for (destination, gradient) in gradients.iter_mut().zip(&current_gradients) {
                    if let Some(gradient) = gradient {
                        if diagnostics {
                            let diagnostic_started = Instant::now();
                            sample_norm +=
                                engine.squared_norm(gradient)? / (tc.loss_scale as f64).powi(2);
                            diagnostic_seconds += diagnostic_started.elapsed().as_secs_f64();
                        }
                        accumulate(
                            &mut engine,
                            destination,
                            gradient,
                            control_weight / tc.loss_scale,
                        )?;
                    }
                }
                gradient_second_moment += control_weight as f64 * sample_norm;
                residual_second_moment += control_weight as f64 * sample_norm;
            }
            for (j, block) in state.blocks.iter().enumerate() {
                let feature = feature.as_ref().unwrap();
                let rank = block.deferred.shape[0];
                let prediction = engine
                    .matmul(feature, &block.predictor, false, false, DType::F32)?
                    .reshape(&[rank, rank])?;
                accumulate(
                    &mut engine,
                    &mut mean_predictions[j],
                    &prediction,
                    1. / tc.accumulation as f32,
                )?;
                if control {
                    if refreshed && control_number < 2 {
                        if let Some(gradient) = &current_gradients[block.index] {
                            let left =
                                engine.matmul(gradient, &block.right, false, false, DType::F32)?;
                            let right =
                                engine.matmul(gradient, &block.left, true, false, DType::F32)?;
                            let left = engine.scale(&left, 1. / tc.loss_scale)?;
                            let right = engine.scale(&right, 1. / tc.loss_scale)?;
                            refresh_sketches[j] = Some(
                                if let Some((first_left, first_right)) = &refresh_sketches[j] {
                                    (
                                        engine.add(first_left, &left, -1., DType::F32)?,
                                        engine.add(first_right, &right, -1., DType::F32)?,
                                    )
                                } else {
                                    (left, right)
                                },
                            );
                        }
                    }
                    let target = targets[j].as_ref().ok_or_else(|| {
                        Error::Message(format!(
                            "projected gradient missing for {}",
                            model.parameters[block.index].name
                        ))
                    })?;
                    let target = engine.scale(target, 1. / tc.loss_scale)?;
                    accumulate(&mut engine, &mut mean_targets[j], &target, control_weight)?;
                    accumulate(
                        &mut engine,
                        &mut control_predictions[j],
                        &prediction,
                        control_weight,
                    )?;
                    let residual = engine.add(&prediction, &target, -1., DType::F32)?;
                    if diagnostics {
                        let diagnostic_started = Instant::now();
                        predictor_mse += control_weight as f64 * engine.squared_norm(&residual)?
                            / (rank * rank) as f64;
                        residual_second_moment += control_weight as f64
                            * (correction_norm(&mut engine, block, &prediction)?
                                - 2. * engine.dot(&prediction, &target)?);
                        diagnostic_seconds += diagnostic_started.elapsed().as_secs_f64();
                    }
                    let residual = residual.reshape(&[1, rank * rank])?;
                    let gradient = engine.matmul(feature, &residual, true, false, DType::F32)?;
                    accumulate(
                        &mut engine,
                        &mut predictor_gradients[j],
                        &gradient,
                        2. * control_weight / (rank * rank) as f32,
                    )?;
                }
            }
            if control {
                control_number += 1;
            }
        }
        drop(corrections);
        let mut base_norm = 0.;
        for gradient in gradients.iter().flatten() {
            base_norm += engine.squared_norm(gradient)?;
        }
        let mut norm_squared = base_norm;
        let mut residual_mean_norm = base_norm;
        let mut deltas = vec![];
        for (j, block) in state.blocks.iter().enumerate() {
            let control = control_predictions[j].as_ref().unwrap();
            let target = mean_targets[j].as_ref().unwrap();
            let delta = engine.add(
                mean_predictions[j].as_ref().unwrap(),
                control,
                -1.,
                DType::F32,
            )?;
            norm_squared +=
                correction_norm(&mut engine, block, &delta)? + 2. * engine.dot(target, &delta)?;
            if diagnostics {
                let diagnostic_started = Instant::now();
                residual_mean_norm += correction_norm(&mut engine, block, control)?
                    - 2. * engine.dot(target, control)?;
                diagnostic_seconds += diagnostic_started.elapsed().as_secs_f64();
            }
            deltas.push(delta);
        }
        ensure(
            norm_squared.is_finite() && (!diagnostics || predictor_mse.is_finite()),
            "nonfinite SGD gradient or predictor error; update aborted",
        )?;
        let norm = norm_squared.max(0.).sqrt();
        let clipping = if tc.clip_enabled && norm > tc.clip_norm as f64 {
            tc.clip_norm as f64 / norm
        } else {
            1.
        } as f32;
        let learning_rate = tc.lr(state.step);
        for (parameter, gradient) in model.parameters.iter().zip(&gradients) {
            let decay = if parameter.value.shape.len() > 1 {
                1. - learning_rate * tc.weight_decay
            } else {
                1.
            };
            if decay != 1. {
                engine.add_into(
                    &parameter.value,
                    &parameter.value,
                    &parameter.value,
                    decay - 1.,
                )?;
            }
            if let Some(gradient) = gradient {
                engine.add_into(
                    &parameter.value,
                    gradient,
                    &parameter.value,
                    -learning_rate * clipping,
                )?;
            }
        }
        for ((block, delta), predictor_gradient) in state
            .blocks
            .iter_mut()
            .zip(&deltas)
            .zip(&predictor_gradients)
        {
            engine.add_into(
                &block.deferred,
                &block.deferred,
                &block.deferred,
                -learning_rate * tc.weight_decay,
            )?;
            engine.add_into(
                &block.deferred,
                delta,
                &block.deferred,
                -learning_rate * clipping,
            )?;
            if let Some(gradient) = predictor_gradient {
                engine.add_into(
                    &block.predictor,
                    gradient,
                    &block.predictor,
                    -tc.control_variate.as_ref().unwrap().predictor_learning_rate,
                )?;
            }
        }
        state.step += 1;
        if refreshed {
            state.refresh_bases(&mut engine, &model, &refresh_sketches)?;
        }
        engine.synchronize()?;
        let step_seconds = step_started.elapsed().as_secs_f64();
        let counters = engine.execution_counters().difference(&before);
        let gradient_dispersion = (gradient_second_moment - base_norm).max(0.);
        let residual_dispersion = (residual_second_moment - residual_mean_norm).max(0.);
        let record = json::json!({
            "step":state.step,"optimizer":tc.optimizer,"loss":loss,"loss_estimator":"weighted_control_mean",
            "gradient_norm":norm,"learning_rate":learning_rate,"step_seconds":step_seconds,
            "elapsed_seconds":started.elapsed().as_secs_f64(),"observed_bytes":tc.accumulation*tc.batch*tc.sequence,
            "exact_bytes":controls*tc.batch*tc.sequence,"controls":controls,"basis_refreshed":refreshed,
            "basis_source":if controls>1 {"control_gradient_contrast"} else {"control_gradient"},
            "gradient_diagnostics":diagnostics,"diagnostic_seconds":diagnostics.then_some(diagnostic_seconds),
            "predictor_mse":diagnostics.then_some(predictor_mse),
            "control_gradient_dispersion":diagnostics.then_some(gradient_dispersion),
            "control_residual_dispersion":diagnostics.then_some(residual_dispersion),
            "residual_variance_ratio_diagnostic":if diagnostics && controls>1 && gradient_dispersion>0. {Some(residual_dispersion/gradient_dispersion)} else {None},
            "peak_gpu_mib":engine.peak_bytes()/1048576,"counters":{
                "launches":counters.launches,"logical_flops":counters.logical_flops,"issued_flops":counters.issued_flops,
                "gemm_launches":counters.gemm_launches,"attention_launches":counters.attention_launches,
                "compilations":counters.compilations,"compile_seconds":counters.compile_seconds
            }
        });
        writeln!(log, "{record}")?;
        log.flush()?;
        if state.step == 1 || state.step.is_multiple_of(tc.log_every) || state.step == tc.steps {
            eprintln!("{record}");
        }
        last_loss = loss;
        if tc.checkpoint_every > 0
            && state.step.is_multiple_of(tc.checkpoint_every)
            && state.step < tc.steps
        {
            let directory = output
                .join("checkpoints")
                .join(format!("step-{:08}", state.step));
            save(
                &mut state,
                &mut engine,
                &model,
                &directory,
                tc,
                &corpus,
                &data_rng,
            )?;
        }
    }
    let directory = output.join("model");
    save(
        &mut state,
        &mut engine,
        &model,
        &directory,
        tc,
        &corpus,
        &data_rng,
    )?;
    save_execution_metrics(&engine, output, complete_started.elapsed().as_secs_f64())?;
    Ok(TrainingSummary {
        steps: state.step,
        last_loss,
        model_directory: directory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stratified_weights_cover_the_population() {
        let observations: Vec<_> = (0..11)
            .map(|i| Observation {
                input: (0..16).map(|j| (j % (i + 1)) as i32).collect(),
                target: vec![],
            })
            .collect();
        let mut config = TrainConfig {
            optimizer: OptimizerKind::ControlVariateSgd,
            accumulation: 11,
            ..Default::default()
        };
        config.control_variate = Some(sejong_core::ControlVariateConfig {
            controls: 5,
            rank: 2,
            sampling: ControlSampling::Stratified,
            strata: 3,
            basis_refresh: 4,
            predictor_learning_rate: 0.01,
        });
        let mut totals = vec![0f64; 11];
        let mut rng = Rng::new(7);
        for _ in 0..2000 {
            let weights = control_weights(&observations, &config, &mut rng);
            assert_eq!(weights.iter().filter(|&&w| w > 0.).count(), 5);
            assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-6);
            for (total, weight) in totals.iter_mut().zip(weights) {
                *total += weight as f64 / 2000.;
            }
        }
        for total in totals {
            assert!((total - 1. / 11.).abs() < 0.01);
        }
    }
}
