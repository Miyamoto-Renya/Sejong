//! 실제 FFN operand에서 정확 VJP와 채널 블록 VJP의 전체 입력 packing 비용을 측정한다.
use sejong_core::{DType, GpuConfig, Result, Rng, ensure, json};
use sejong_gpu::{Engine, ExecutionCounters, Tensor};
use std::{io::Write, path::Path, time::Instant};

fn read_tensor(e: &mut Engine, root: &Path, record: &json::Value) -> Result<Tensor> {
    let shape: Vec<usize> = json::from_value(record["shape"].clone())?;
    let bytes = std::fs::read(root.join(record["file"].as_str().unwrap()))?;
    let data: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure(
        bytes.len() == 4 * data.len() && data.len() == shape.iter().product::<usize>(),
        "trace tensor byte count mismatch",
    )?;
    let value = e.from_f32(&shape, &data)?;
    let dtype: DType = json::from_value(record["original_dtype"].clone())?;
    e.cast(&value, dtype)
}
fn value(e: &mut Engine, root: &Path, meta: &json::Value, id: &json::Value) -> Result<Tensor> {
    read_tensor(e, root, &meta["tensors"][id.as_str().unwrap()])
}
struct Ffn {
    x: Tensor,
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    g: Tensor,
    u: Tensor,
    h: Tensor,
    dy: Tensor,
}
struct Packed {
    wg: Tensor,
    wu: Tensor,
    wd: Tensor,
    g: Tensor,
    u: Tensor,
    h: Tensor,
}
fn columns(e: &mut Engine, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let transposed = e.transpose2d(x)?;
    let selected = e.gather(&transposed, ids)?;
    e.transpose2d(&selected)
}
fn pack(e: &mut Engine, f: &Ffn, ids: &Tensor) -> Result<Packed> {
    Ok(Packed {
        wg: columns(e, &f.wg, ids)?,
        wu: columns(e, &f.wu, ids)?,
        wd: e.gather(&f.wd, ids)?,
        g: columns(e, &f.g, ids)?,
        u: columns(e, &f.u, ids)?,
        h: columns(e, &f.h, ids)?,
    })
}
fn derivative(
    e: &mut Engine,
    x: &Tensor,
    dy: &Tensor,
    p: &Packed,
    probability: f32,
) -> Result<Vec<Tensor>> {
    let dy = if probability == 1. {
        dy.clone()
    } else {
        e.scale(dy, 1. / probability)?
    };
    let dh = e.matmul(&dy, &p.wd, false, true, DType::F32)?;
    let dwd = e.matmul(&p.h, &dy, true, false, DType::F32)?;
    let (dg, du) = e.swiglu_backward(&p.g, &p.u, &dh)?;
    let dwg = e.matmul(x, &dg, true, false, DType::F32)?;
    let dwu = e.matmul(x, &du, true, false, DType::F32)?;
    let dxg = e.matmul(&dg, &p.wg, false, true, DType::F32)?;
    let dxu = e.matmul(&du, &p.wu, false, true, DType::F32)?;
    let dx = e.add(&dxg, &dxu, 1., DType::F32)?;
    Ok(vec![dx, dwg, dwu, dwd])
}
fn materialize(e: &mut Engine, compact: Vec<Tensor>, f: &Ffn, ids: &Tensor) -> Result<Vec<Tensor>> {
    let mut output = vec![compact[0].clone()];
    for (gradient, shape) in [(&compact[1], &f.wg.shape), (&compact[2], &f.wu.shape)] {
        let transposed = e.transpose2d(gradient)?;
        let full = e.gather_backward(&transposed, ids, &[shape[1], shape[0]])?;
        output.push(e.transpose2d(&full)?);
    }
    output.push(e.gather_backward(&compact[3], ids, &f.wd.shape)?);
    Ok(output)
}
fn counters(c: &ExecutionCounters) -> json::Value {
    json::json!({"launches":c.launches,"logical_flops":c.logical_flops,"issued_flops":c.issued_flops,
        "gemm_launches":c.gemm_launches,"compilations":c.compilations})
}
fn save(e: &mut Engine, out: &Path, prefix: &str, tensors: &[Tensor]) -> Result<()> {
    for (name, tensor) in ["dx", "dwg", "dwu", "dwd"].into_iter().zip(tensors) {
        let mut file = std::io::BufWriter::new(std::fs::File::create(
            out.join(format!("{prefix}-{name}.f32")),
        )?);
        for value in e.to_f32(tensor)? {
            ensure(value.is_finite(), "replayed VJP produced nonfinite values")?;
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() == 7,
        "usage: adjoint_replay TRACE FFN_PREFIX OUT CHANNELS BLOCK WARMUP REPEATS",
    )?;
    let root = Path::new(&args[0]);
    let prefix = &args[1];
    let out = Path::new(&args[2]);
    let parse = |s: &str| {
        s.parse::<usize>()
            .map_err(|_| sejong_core::Error::Message(format!("invalid integer {s}")))
    };
    let channels = parse(&args[3])?;
    let block = parse(&args[4])?;
    let warmup = parse(&args[5])?;
    let repeats = parse(&args[6])?;
    ensure(
        repeats > 0 && block > 0 && channels > 0 && channels.is_multiple_of(block),
        "positive repeats and whole selected channel blocks required",
    )?;
    let bytes = std::fs::read(root.join("trace.json"))?;
    let meta: json::Value = json::from_slice(&bytes)?;
    let find = |suffix: &str| {
        meta["linear_operations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["parameter"].as_str() == Some(&format!("{prefix}.{suffix}")))
            .cloned()
            .ok_or_else(|| sejong_core::Error::Message(format!("missing FFN {prefix}.{suffix}")))
    };
    let gate = find("gate")?;
    let up = find("up")?;
    let down = find("down")?;
    let mut e = Engine::new(GpuConfig::default())?;
    let f = Ffn {
        x: value(&mut e, root, &meta, &gate["input"])?,
        wg: value(&mut e, root, &meta, &gate["weight"])?,
        wu: value(&mut e, root, &meta, &up["weight"])?,
        wd: value(&mut e, root, &meta, &down["weight"])?,
        g: value(&mut e, root, &meta, &gate["output"])?,
        u: value(&mut e, root, &meta, &up["output"])?,
        h: value(&mut e, root, &meta, &down["input"])?,
        dy: read_tensor(&mut e, root, &down["delta"])?,
    };
    let width = f.wg.shape[1];
    ensure(
        width.is_multiple_of(block) && channels <= width,
        "channel budget exceeds FFN or block divides hidden width incorrectly",
    )?;
    let mut rng = Rng::new(meta["model"]["seed"].as_u64().unwrap());
    let mut blocks: Vec<_> = (0..width / block).collect();
    for i in (1..blocks.len()).rev() {
        let j = rng.index(i + 1);
        blocks.swap(i, j);
    }
    blocks.truncate(channels / block);
    blocks.sort_unstable();
    let selected: Vec<i32> = blocks
        .iter()
        .flat_map(|b| b * block..(b + 1) * block)
        .map(|i| i as i32)
        .collect();
    let ids = e.from_i32(&[selected.len()], &selected)?;
    let probability = channels as f32 / width as f32;
    let full = Packed {
        wg: f.wg.clone(),
        wu: f.wu.clone(),
        wd: f.wd.clone(),
        g: f.g.clone(),
        u: f.u.clone(),
        h: f.h.clone(),
    };
    let packed = pack(&mut e, &f, &ids)?;
    std::fs::create_dir_all(out)?;
    let mut measurements = Vec::new();
    for mode in [
        "exact",
        "sampled_compact_prepacked",
        "sampled_dense_with_packing",
    ] {
        let mut times = Vec::new();
        let mut operation_counts = Vec::new();
        for repeat in 0..warmup + repeats {
            e.synchronize()?;
            let before = e.execution_counters();
            let start = Instant::now();
            let result = match mode {
                "exact" => derivative(&mut e, &f.x, &f.dy, &full, 1.)?,
                "sampled_compact_prepacked" => {
                    derivative(&mut e, &f.x, &f.dy, &packed, probability)?
                }
                _ => {
                    let packed = pack(&mut e, &f, &ids)?;
                    let compact = derivative(&mut e, &f.x, &f.dy, &packed, probability)?;
                    materialize(&mut e, compact, &f, &ids)?
                }
            };
            e.synchronize()?;
            let seconds = start.elapsed().as_secs_f64();
            let count = e.execution_counters().difference(&before);
            if repeat >= warmup {
                times.push(seconds);
                operation_counts.push(counters(&count));
            }
            if repeat == warmup + repeats - 1 && mode != "sampled_compact_prepacked" {
                save(&mut e, out, mode, &result)?;
            }
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        measurements.push(
            json::json!({"mode":mode,"mean_seconds":times.iter().sum::<f64>()/times.len() as f64,
            "median_seconds":(sorted[(sorted.len()-1)/2]+sorted[sorted.len()/2])/2.,
            "seconds":times,"counters":operation_counts}),
        );
    }
    let report = json::json!({"format":"sejong-ffn-adjoint-replay-v1","trace_sha256_note":"trace_blake3 identifies input metadata",
        "trace_blake3":blake3::hash(&bytes).to_hex().to_string(),"device":e.name(),"prefix":prefix,
        "shape":[f.x.shape[0],f.x.shape[1],width],"channels":channels,"block":block,"selected_channels":selected,
        "sampling_probability":probability,"warmup":warmup,"repeats":repeats,"measurements":measurements,
        "scope":"fixed captured FFN operands; exact forward already executed; per-VJP wall time excludes optimizer and fresh selection",
        "sampled_compact_prepacked_scope":"cached selected operands and compact parameter gradients",
        "sampled_dense_with_packing_scope":"each VJP gathers selected operands and scatters full-sized parameter gradients"});
    std::fs::write(out.join("replay.json"), json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json::json!({"out":out,"prefix":prefix,"shape":report["shape"],"measurements":measurements.iter().map(|m|json::json!({"mode":m["mode"],"seconds":m["mean_seconds"]})).collect::<Vec<_>>()})
    );
    Ok(())
}
