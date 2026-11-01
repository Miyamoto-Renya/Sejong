//! 원본 학습·재개의 전체 실행 시간과 GPU 연산량을 같은 입력으로 기록한다.
use sejong_core::{
    GpuConfig, Result, ensure, json,
    native::{MemoryTraining, RevisionTraining, ValueTraining},
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::native::memory::Memory;
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() >= 3,
        "usage: native_training_benchmark memory|value|revision|query SPEC OUT [ORIGINAL]",
    )?;
    let out = Path::new(&args[2]);
    let bytes = std::fs::read(&args[1])?;
    let mut e = Engine::new(GpuConfig::default())?;
    let original = if args[0] == "revision" {
        Some(Memory::load(&mut e, Path::new(&args[3]))?)
    } else if args[0] == "query" {
        Some(Memory::load_training(&mut e, Path::new(&args[3]))?)
    } else {
        None
    };
    e.enable_execution_metrics(true);
    e.reset_execution_counters();
    e.synchronize()?;
    let started = Instant::now();
    let details = match args[0].as_str() {
        "memory" => {
            let spec: MemoryTraining = json::from_slice(&bytes)?;
            let model = sejong_train::memory::train_memory(&mut e, &spec, out)?;
            json::json!({"query_steps":model.metadata.query_steps,
                "fact_steps":model.metadata.fact_steps,"requests":spec.requests.len(),
                "parameters":model.parameters.iter().map(|p|p.value.len()).sum::<usize>()})
        }
        "value" => {
            let spec: ValueTraining = json::from_slice(&bytes)?;
            let model = sejong_train::memory::train_value(&mut e, &spec, out)?;
            json::json!({"steps":model.metadata.steps,"loss":model.metadata.loss,
                "parameters":model.parameters().iter().map(|p|p.value.len()).sum::<usize>(),
                "requests":spec.requests.len()})
        }
        "revision" => {
            let spec: RevisionTraining = json::from_slice(&bytes)?;
            let model = sejong_train::memory::revise_memory(
                &mut e,
                original.as_ref().unwrap(),
                &spec,
                out,
                None,
            )?;
            json::json!({"steps":spec.steps,"fact_steps":model.metadata.fact_steps})
        }
        "query" => {
            let spec: MemoryTraining = json::from_slice(&bytes)?;
            let model = original.as_ref().unwrap();
            let chunk = spec.batch.max_examples.unwrap_or(spec.requests.len());
            let mut checksum = 0.;
            for _ in 0..spec.fact_steps {
                for contexts in spec.contexts.chunks(chunk) {
                    let values: Vec<_> = contexts.iter().flatten().map(|x| *x as f32).collect();
                    let input = e.from_f32(
                        &[contexts.len() * spec.config.tokens, spec.config.dimension],
                        &values,
                    )?;
                    let mut graph = Graph::new(&mut e, false);
                    let weights = model.bind(&mut graph, false, false);
                    let query = model.query(&mut graph, &weights, &input)?;
                    checksum += graph
                        .value(query)?
                        .iter()
                        .map(|x| f64::from(*x))
                        .sum::<f64>();
                }
            }
            std::fs::create_dir_all(out)?;
            json::json!({"passes":spec.fact_steps,"requests":spec.requests.len(),"checksum":checksum})
        }
        _ => {
            return Err(sejong_core::Error::Message(
                "mode must be memory, value, revision or query".into(),
            ));
        }
    };
    e.synchronize()?;
    let seconds = started.elapsed().as_secs_f64();
    let c = e.execution_counters();
    let scope = if args[0] == "query" {
        "fixed query forward passes including context upload and output readback"
    } else {
        "complete training call including initialization, optimizer, checkpoint, calibration and model save"
    };
    let result = json::json!({"scope":scope,
        "mode":args[0],"seconds":seconds,"peak_gpu_bytes":e.peak_bytes(),"details":details,
        "counters":{"launches":c.launches,"logical_flops":c.logical_flops,"issued_flops":c.issued_flops,
        "gemm_launches":c.gemm_launches,"attention_launches":c.attention_launches,
        "compilations":c.compilations,"compile_seconds":c.compile_seconds}});
    std::fs::write(out.join("benchmark.json"), json::to_vec_pretty(&result)?)?;
    println!("{}", json::to_string_pretty(&result)?);
    Ok(())
}
