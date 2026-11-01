//! 실제 Sejong 모델의 선형 역전파 operand를 원래 tensor shape로 저장한다.
use sejong_core::{Config, Result, ensure, json};
use sejong_gpu::{Engine, Tensor, graph::Graph};
use sejong_model::Model;
use std::{collections::HashMap, io::Write, path::Path};

fn write_tensor(engine: &mut Engine, path: &Path, tensor: &Tensor) -> Result<json::Value> {
    let values = engine.to_f32(tensor)?;
    ensure(
        values.iter().all(|x| x.is_finite()),
        "trace tensor is nonfinite",
    )?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in &values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(
        json::json!({"file":path.file_name().unwrap().to_string_lossy(),
        "shape":tensor.shape,"storage":"little_endian_f32","original_dtype":tensor.dtype}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() == 5 || args.len() == 6,
        "usage: adjoint_trace CONFIG DATA OUT SEQUENCE OFFSET [MODEL]",
    )?;
    let config = Config::load(Path::new(&args[0]))?;
    let data = std::fs::read(&args[1])?;
    let out = Path::new(&args[2]);
    let sequence: usize = args[3]
        .parse()
        .map_err(|_| sejong_core::Error::Message("invalid sequence".into()))?;
    let offset: usize = args[4]
        .parse()
        .map_err(|_| sejong_core::Error::Message("invalid offset".into()))?;
    ensure(
        sequence > 0
            && offset
                .checked_add(sequence)
                .is_some_and(|end| end < data.len()),
        "trace input must contain sequence+1 bytes after offset",
    )?;
    std::fs::create_dir_all(out)?;
    let mut engine = Engine::new(config.gpu.clone())?;
    let model = if let Some(path) = args.get(5) {
        Model::load(&mut engine, Path::new(path))?.0
    } else {
        Model::initialize(&mut engine, config.model.clone())?
    };
    let input: Vec<_> = data[offset..offset + sequence]
        .iter()
        .map(|x| i32::from(*x))
        .collect();
    let target: Vec<_> = data[offset + 1..offset + sequence + 1]
        .iter()
        .map(|x| i32::from(*x))
        .collect();
    let ids = engine.from_i32(&[sequence], &input)?;
    let targets = engine.from_i32(&[sequence], &target)?;
    let mut graph = Graph::new(&mut engine, true);
    graph.checkpointing = false;
    graph.enable_linear_adjoint_trace()?;
    let binding = model.bind(&mut graph, true)?;
    let names: HashMap<_, _> = binding
        .compute
        .iter()
        .zip(&model.parameters)
        .chain(binding.leaves.iter().zip(&model.parameters))
        .map(|(v, p)| (v.0, p.name.clone()))
        .collect();
    let logits = model.logits(&mut graph, &binding, &ids, sequence)?;
    let loss = graph.cross_entropy(logits, &targets)?;
    let losses = graph.value(loss)?;
    graph.backward(&[(loss, 1.)])?;
    let traces = graph.linear_adjoint_trace().to_vec();
    drop(graph);
    let mut tensors = HashMap::new();
    let mut records = Vec::new();
    for trace in traces {
        let Some(name) = names.get(&trace.weight.0) else {
            continue;
        };
        for (id, tensor) in [
            (trace.input, &trace.x),
            (trace.weight, &trace.w),
            (trace.output, &trace.y),
        ] {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                tensors.entry(id.0.to_string())
            {
                entry.insert(write_tensor(
                    &mut engine,
                    &out.join(format!("v{}.f32", id.0)),
                    tensor,
                )?);
            }
        }
        let delta = write_tensor(
            &mut engine,
            &out.join(format!("d{}.f32", trace.output.0)),
            &trace.dy,
        )?;
        records.push(
            json::json!({"parameter":name,"input":trace.input.0.to_string(),
            "weight":trace.weight.0.to_string(),"output":trace.output.0.to_string(),"delta":delta}),
        );
    }
    let mean_loss = losses.iter().map(|x| f64::from(*x)).sum::<f64>() / losses.len() as f64;
    let report = json::json!({"format":"sejong-linear-adjoint-trace-v1","device":engine.name(),
        "model":model.config,"model_parameters":model.parameter_count(),"sequence":sequence,"offset":offset,
        "model_path":args.get(5),"data_path":args[1],"data_blake3":blake3::hash(&data).to_hex().to_string(),
        "executable_blake3":blake3::hash(&std::fs::read(std::env::current_exe()?)?).to_hex().to_string(),
        "mean_cross_entropy":mean_loss,"bits_per_byte_token":mean_loss/std::f64::consts::LN_2,
        "checkpointing":false,"derivative":"existing exact manual reverse mode",
        "tensors":tensors,"linear_operations":records});
    std::fs::write(out.join("trace.json"), json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json::json!({"out":out,"device":engine.name(),"linear_operations":records.len(),
        "bits_per_byte_token":mean_loss/std::f64::consts::LN_2})
    );
    Ok(())
}
