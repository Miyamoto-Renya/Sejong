use sejong_core::{Config, DType, Result, Rng, ensure, json};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::Model;
use sejong_train::Corpus;
use std::{io::Write, path::PathBuf, time::Instant};

fn source_fingerprint() -> Result<String> {
    fn collect(path: &std::path::Path, paths: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if entry.file_name() != "__pycache__" {
                    collect(&entry.path(), paths)?;
                }
            } else if entry.file_type()?.is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "rs" || ext == "py" || ext == "toml")
            {
                paths.push(entry.path());
            }
        }
        Ok(())
    }
    let root = sejong_core::root();
    let mut paths = vec![root.join("Cargo.lock"), root.join("Cargo.toml")];
    collect(&root.join("crates"), &mut paths)?;
    collect(&root.join("python"), &mut paths)?;
    paths.sort();
    let mut hash = blake3::Hasher::new();
    for path in paths {
        let bytes = std::fs::read(&path)?;
        hash.update(
            path.strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hash.update(&(bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    Ok(hash.finalize().to_hex().to_string())
}

fn runtime_versions() -> Result<json::Value> {
    let python = std::env::var_os("SEJONG_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| sejong_core::root().join(".venv/bin/python"));
    let output = std::process::Command::new(python).args(["-c",
        "import importlib.metadata as m, json, platform; print(json.dumps({'python':platform.python_version(),'triton':m.version('triton'),'rocm_sdk_core':m.version('rocm-sdk-core')}))"
    ]).output()?;
    ensure(
        output.status.success(),
        "benchmark runtime version discovery failed",
    )?;
    Ok(json::from_slice(&output.stdout)?)
}

pub struct Request {
    pub configs: Vec<PathBuf>,
    pub data: PathBuf,
    pub validation: PathBuf,
    pub out: PathBuf,
    pub warmup_steps: usize,
    pub measure_steps: usize,
    pub evaluation_batches: usize,
    pub decode_bytes: usize,
}

fn evaluate(
    engine: &mut Engine,
    model: &Model,
    corpus: &Corpus,
    config: &Config,
    batches: usize,
) -> Result<f64> {
    let mut rng = Rng::new(config.training.seed);
    let mut sum = 0.;
    let mut count = 0usize;
    for _ in 0..batches {
        let (input, target) =
            corpus.batch(&mut rng, config.training.batch, config.training.sequence)?;
        let ids = engine.from_i32(&[input.len()], &input)?;
        let targets = engine.from_i32(&[target.len()], &target)?;
        let mut graph = Graph::new(engine, false);
        let binding = model.bind(&mut graph, false)?;
        let logits = model.logits(&mut graph, &binding, &ids, config.training.sequence)?;
        let loss = graph.cross_entropy(logits, &targets)?;
        let values = graph.value(loss)?;
        ensure(
            values.iter().all(|x| x.is_finite()),
            "evaluation produced nonfinite loss",
        )?;
        sum += values.iter().map(|x| *x as f64).sum::<f64>();
        count += values.len();
    }
    Ok(sum / count as f64 / std::f64::consts::LN_2)
}

fn distribution(values: &[f64]) -> json::Value {
    if values.is_empty() {
        return json::Value::Null;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let n = ordered.len();
    json::json!({"samples": n, "mean": values.iter().sum::<f64>() / n as f64,
        "median": (ordered[(n - 1) / 2] + ordered[n / 2]) / 2.,
        "minimum": ordered[0], "maximum": ordered[n - 1]})
}

fn cached_measurement(
    engine: &mut Engine,
    model: &Model,
    corpus: &Corpus,
    config: &Config,
    decode: usize,
) -> Result<json::Value> {
    let mut rng = Rng::new(config.training.seed);
    let (input, _) = corpus.batch(&mut rng, 1, config.training.sequence)?;
    let prefix = input.len() - decode;
    let ids = engine.from_i32(&[input.len()], &input)?;
    let mut graph = Graph::new(engine, false);
    let binding = model.bind(&mut graph, false)?;
    let logits = model.logits(&mut graph, &binding, &ids, input.len())?;
    let full = graph.value(logits)?;
    drop(graph);
    let weights = model.inference_weights(engine)?;
    // 첫 실행은 실제 cache 경로의 커널을 준비하고 두 번째 실행을 측정한다.
    let mut last = None;
    let mut prefill_seconds = 0.;
    let mut decode_seconds = Vec::new();
    for pass in 0..2 {
        let mut cache = model.new_cache(engine)?;
        engine.synchronize()?;
        let start = Instant::now();
        let _ = model.forward_cached(engine, &weights, &input[..prefix], &mut cache)?;
        engine.synchronize()?;
        if pass == 1 {
            prefill_seconds = start.elapsed().as_secs_f64();
        }
        for token in &input[prefix..] {
            let start = Instant::now();
            let output = model.forward_cached(engine, &weights, &[*token], &mut cache)?;
            engine.synchronize()?;
            if pass == 1 {
                decode_seconds.push(start.elapsed().as_secs_f64());
            }
            last = Some(output);
        }
    }
    let last = engine.to_f32(&last.unwrap())?;
    let expected = &full[full.len() - last.len()..];
    ensure(
        last.iter().all(|x| x.is_finite()),
        "cached inference produced nonfinite logits",
    )?;
    let max_error = last
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0., f64::max);
    Ok(
        json::json!({"prefill_tokens":prefix, "decode_tokens":decode,
        "prefill_seconds":prefill_seconds,"decode_seconds_per_token":distribution(&decode_seconds),
        "decode_tokens_per_second":decode as f64 / decode_seconds.iter().sum::<f64>(),
        "cached_vs_full_last_logits_max_abs_error":max_error,
        "measurement":"same held-out input, one cache warmup, synchronized teacher-forced decoding"}),
    )
}

pub fn run(request: Request) -> Result<()> {
    ensure(
        request.measure_steps > 0 && request.evaluation_batches > 0 && request.decode_bytes > 0,
        "measurement steps, evaluation batches and decode bytes must be positive",
    )?;
    ensure(
        !request.configs.is_empty(),
        "benchmark requires at least one configuration",
    )?;
    ensure(
        !request.out.exists(),
        "benchmark destination already exists",
    )?;
    let steps = request
        .warmup_steps
        .checked_add(request.measure_steps)
        .ok_or_else(|| sejong_core::Error::Message("benchmark step count overflow".into()))?;
    let validation = Corpus::open(&request.validation)?;
    let train_corpus = Corpus::open(&request.data)?;
    ensure(
        validation.fingerprint != train_corpus.fingerprint,
        "benchmark requires distinct training and validation corpora",
    )?;
    let mut configs = Vec::new();
    for path in &request.configs {
        let mut config = Config::load(path)?;
        ensure(
            request.decode_bytes < config.training.sequence,
            "decode bytes must be smaller than the training sequence",
        )?;
        ensure(
            config.training.sequence < validation.tokens,
            "validation corpus must exceed the sequence length",
        )?;
        config.training.steps = steps;
        config.training.log_every = 1;
        config.training.validate(&config.model)?;
        configs.push(config);
    }
    std::fs::create_dir_all(&request.out)?;
    let mut reports = Vec::new();
    let source_blake3 = source_fingerprint()?;
    let runtime_versions = runtime_versions()?;
    let executable_blake3 = blake3::hash(&std::fs::read(std::env::current_exe()?)?)
        .to_hex()
        .to_string();
    for (index, config) in configs.iter().enumerate() {
        let destination = request.out.join(format!("run-{index:03}"));
        let (initial_bpb, parameters, device) = {
            let mut engine = Engine::new(config.gpu.clone())?;
            let model = Model::initialize(&mut engine, config.model.clone())?;
            (
                evaluate(
                    &mut engine,
                    &model,
                    &validation,
                    config,
                    request.evaluation_batches,
                )?,
                model.parameter_count(),
                engine.name().to_owned(),
            )
        };
        let wall_start = Instant::now();
        let trained = sejong_train::train_measured(config, &train_corpus, &destination, None)?;
        let complete_training_seconds = wall_start.elapsed().as_secs_f64();
        let execution: json::Value =
            json::from_slice(&std::fs::read(destination.join("execution.json"))?)?;
        let records = std::fs::read_to_string(destination.join("metrics.jsonl"))?
            .lines()
            .map(json::from_str::<json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let measured: Vec<_> = records
            .iter()
            .filter(|r| {
                r["step"]
                    .as_u64()
                    .is_some_and(|s| s > request.warmup_steps as u64)
            })
            .collect();
        ensure(
            measured.len() == request.measure_steps,
            "training metrics must contain every measured step",
        )?;
        let times: Vec<_> = measured
            .iter()
            .map(|r| {
                r["step_seconds"].as_f64().ok_or_else(|| {
                    sejong_core::Error::Message("training metrics require step_seconds".into())
                })
            })
            .collect::<Result<_>>()?;
        ensure(
            times.iter().all(|t| t.is_finite() && *t > 0.),
            "measured step times must be finite and positive",
        )?;
        let read_bytes = |key: &str| -> Result<u64> {
            measured.iter().try_fold(0u64, |total, r| {
                let value = r[key].as_u64().ok_or_else(|| {
                    sejong_core::Error::Message(format!("training metrics require {key}"))
                })?;
                total.checked_add(value).ok_or_else(|| {
                    sejong_core::Error::Message("measured byte count overflow".into())
                })
            })
        };
        let bytes = read_bytes("observed_bytes")?;
        let exact_bytes = if measured.iter().all(|r| r["exact_bytes"].is_number()) {
            Some(read_bytes("exact_bytes")?)
        } else {
            None
        };
        let backward_bytes = measured.iter().try_fold(0u64, |total, r| {
            let value = r["backward_bytes"]
                .as_u64()
                .or_else(|| r["exact_bytes"].as_u64())
                .ok_or_else(|| {
                    sejong_core::Error::Message("training metrics require backward_bytes".into())
                })?;
            total
                .checked_add(value)
                .ok_or_else(|| sejong_core::Error::Message("backward byte count overflow".into()))
        })?;
        let steady_times: Vec<_> = measured
            .iter()
            .zip(&times)
            .filter(|(r, _)| r["counters"]["compilations"].as_u64() == Some(0))
            .map(|(_, t)| *t)
            .collect();
        let mut engine = Engine::new(config.gpu.clone())?;
        let (model, _) = Model::load(&mut engine, &trained.model_directory)?;
        let final_bpb = evaluate(
            &mut engine,
            &model,
            &validation,
            config,
            request.evaluation_batches,
        )?;
        let inference = cached_measurement(
            &mut engine,
            &model,
            &validation,
            config,
            request.decode_bytes,
        )?;
        let report = json::json!({"config_path":request.configs[index],"config":config,"device":device,
            "model_parameters":parameters,"training_fingerprint":train_corpus.fingerprint,
            "validation_fingerprint":validation.fingerprint,"evaluation_batches":request.evaluation_batches,
            "initial_validation_bits_per_byte_token":initial_bpb,"final_validation_bits_per_byte_token":final_bpb,
            "warmup_steps":request.warmup_steps,"measured_steps":request.measure_steps,
            "training_step_seconds":distribution(&times),"steps_without_compilation_seconds":distribution(&steady_times),
            "observed_bytes":bytes,"exact_bytes":exact_bytes,"backward_bytes":backward_bytes,
            "observed_bytes_per_second":bytes as f64 / times.iter().sum::<f64>(),
            "exact_bytes_per_second":exact_bytes.map(|bytes|bytes as f64 / times.iter().sum::<f64>()),
            "backward_bytes_per_second":backward_bytes as f64 / times.iter().sum::<f64>(),
            "complete_training_seconds_including_initialization_and_save":complete_training_seconds,
            "complete_execution":execution,"source_blake3":source_blake3,"executable_blake3":executable_blake3,
            "runtime_versions":runtime_versions,
            "build_profile":if cfg!(debug_assertions) {"debug"} else {"release"},
            "flops_scope":"executed GEMM, attention dot and parameter-basis QR arithmetic, FMA=2; pointwise and transfers included in wall time",
            "measured_step_records":measured,"inference":inference,
            "loss_unit":"cross entropy / ln(2), byte vocabulary with document EOS",
            "precision_unit_roundoff":match config.model.precision { DType::F16 => 2f64.powi(-11), _=>2f64.powi(-24)}});
        std::fs::write(
            destination.join("benchmark.json"),
            json::to_vec_pretty(&report)?,
        )?;
        reports.push(report);
        let report_path = request.out.join("benchmark.json");
        let temporary = request.out.join("benchmark.json.tmp");
        let mut output = std::fs::File::create(&temporary)?;
        output.write_all(&json::to_vec_pretty(
            &json::json!({"format":"sejong-measured-benchmark-v1",
            "training_data":request.data,"validation_data":request.validation,"runs":reports}),
        )?)?;
        output.sync_all()?;
        std::fs::rename(temporary, report_path)?;
    }
    println!("{}", request.out.join("benchmark.json").display());
    Ok(())
}
