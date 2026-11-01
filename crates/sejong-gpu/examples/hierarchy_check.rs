//! 고정 행렬에서 계층 GPU 연산과 수동 기울기를 비교한다.
use sejong_core::{DType, GpuConfig, Result, ensure, json};
use sejong_gpu::{Engine, graph::Graph};

fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}
fn values(n: usize, offset: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + offset) % 41) as f32 - 20.) / 31.)
        .collect()
}
fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
    t: usize,
    h: usize,
    kh: usize,
    d: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.; q.len()];
    let mut dq = out.clone();
    let mut dk = vec![0.; k.len()];
    let mut dv = dk.clone();
    for pos in 0..t {
        for head in 0..h {
            let qr = (pos * h + head) * d;
            let kvh = head / (h / kh);
            let first = if window == 0 {
                0
            } else {
                (pos + 1).saturating_sub(window)
            };
            let scores = (first..=pos)
                .map(|p| {
                    let kr = (p * kh + kvh) * d;
                    (0..d)
                        .map(|j| q[qr + j] as f64 * k[kr + j] as f64)
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z = scores.iter().map(|x| (x - maximum).exp()).sum::<f64>();
            let probs = scores
                .iter()
                .map(|x| (x - maximum).exp() / z)
                .collect::<Vec<_>>();
            let dp = (first..=pos)
                .map(|p| {
                    let kr = (p * kh + kvh) * d;
                    (0..d)
                        .map(|j| dy[qr + j] as f64 * v[kr + j] as f64)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let delta = probs.iter().zip(&dp).map(|(p, g)| p * g).sum::<f64>();
            for (j, p) in (first..=pos).enumerate() {
                let kr = (p * kh + kvh) * d;
                let ds = probs[j] * (dp[j] - delta) / (d as f64).sqrt();
                for col in 0..d {
                    out[qr + col] += (probs[j] * v[kr + col] as f64) as f32;
                    dq[qr + col] += (ds * k[kr + col] as f64) as f32;
                    dk[kr + col] += (ds * q[qr + col] as f64) as f32;
                    dv[kr + col] += (probs[j] * dy[qr + col] as f64) as f32;
                }
            }
        }
    }
    (out, dq, dk, dv)
}

fn main() -> Result<()> {
    let mut e = Engine::new(GpuConfig::default())?;
    let (t, h, kh, d) = (5, 4, 2, 4);
    let qv = values(t * h * d, 0);
    let kv = values(t * kh * d, 3);
    let vv = values(t * kh * d, 9);
    let dyv = values(t * h * d, 12);
    let q = e.from_f32(&[t, h * d], &qv)?;
    let k = e.from_f32(&[t, kh * d], &kv)?;
    let v = e.from_f32(&[t, kh * d], &vv)?;
    let dy = e.from_f32(&[t, h * d], &dyv)?;
    let mut attention_error = 0f32;
    let mut cache_error = 0f32;
    for window in [0, 3] {
        let setting = if window == 0 { None } else { Some(window) };
        let (out, lse) = e.attention_grouped(&q, &k, &v, t, t, 0, h, kh, setting)?;
        let (dq, dk, dv) =
            e.attention_grouped_backward(&q, &k, &v, &out, &lse, &dy, t, h, kh, setting)?;
        let reference = attention_reference(&qv, &kv, &vv, &dyv, t, h, kh, d, window);
        for (actual, expected) in [
            (&out, &reference.0),
            (&dq, &reference.1),
            (&dk, &reference.2),
            (&dv, &reference.3),
        ] {
            attention_error = attention_error.max(max_error(&e.to_f32(actual)?, expected));
        }
        let (cached, _) =
            e.attention_grouped(&q.rows(t - 1, 1)?, &k, &v, 1, t, t - 1, h, kh, setting)?;
        cache_error = cache_error.max(max_error(
            &e.to_f32(&cached)?,
            &reference.0[(t - 1) * h * d..],
        ));
    }
    ensure(
        attention_error < 2e-5 && cache_error < 2e-5,
        "grouped/window attention comparison",
    )?;

    let input_values = (0..42).map(|v| v as f32 / 10.).collect::<Vec<_>>();
    let x = e.from_f32(&[14, 3], &input_values)?;
    let bos = e.from_f32(&[1, 3], &[-1., -2., -3.])?;
    let mut g = Graph::new(&mut e, true);
    g.checkpointing = false;
    let xv = g.leaf(x, true);
    let bv = g.leaf(bos, true);
    let pooled = g.patch_pool(xv, 7, 3)?;
    let broadcast = g.patch_broadcast(pooled, bv, 7, 3)?;
    let output = g.value(broadcast)?;
    let seed = g.engine.full(&[14, 3], DType::F32, 1.)?;
    g.backward_seeded(&[(broadcast, seed)])?;
    let dx = g.gradient(xv).unwrap().clone();
    let db = g.gradient(bv).unwrap().clone();
    drop(g);
    let mut expected = vec![];
    let mut expected_dx = vec![0.; 42];
    for batch in 0..2 {
        for pos in 0..7 {
            for col in 0..3 {
                if pos < 3 {
                    expected.push(-(col as f32 + 1.));
                } else {
                    let previous = pos / 3 - 1;
                    expected.push(
                        (0..3)
                            .map(|j| input_values[(batch * 7 + previous * 3 + j) * 3 + col])
                            .sum::<f32>()
                            / 3.,
                    );
                }
                expected_dx[(batch * 7 + pos) * 3 + col] = if pos < 3 {
                    1.
                } else if pos < 6 {
                    1. / 3.
                } else {
                    0.
                };
            }
        }
    }
    let patch_error = max_error(&output, &expected)
        .max(max_error(&e.to_f32(&dx)?, &expected_dx))
        .max(max_error(&e.to_f32(&db)?, &[6., 6., 6.]));
    ensure(patch_error < 2e-6, "patch shift/partial pooling gradients")?;

    let logits_values = [
        0.2, 1.1, -0.7, 0.6, -0.4, 1.4, -0.1, 0.8, 1.7, 2.1, 0.2, -1.,
    ];
    let logits = e.from_f32(&[4, 3], &logits_values)?;
    let input_values = values(20, 8);
    let x = e.from_f32(&[4, 5], &input_values)?;
    let mut g = Graph::new(&mut e, true);
    let lv = g.leaf(logits, true);
    let xv = g.leaf(x, true);
    let (weights, ids) = g.route_topk(lv, 2)?;
    let routes = g.engine.expert_routes(&ids, 3)?;
    let mut outputs = vec![];
    for (expert, route) in routes.into_iter().enumerate() {
        if let Some((tokens, slots)) = route {
            let gathered = g.gather(xv, &tokens)?;
            let scaled = g.scale(gathered, (expert + 1) as f32)?;
            outputs.push((scaled, slots));
        }
    }
    let combined = g.expert_combine(&outputs, weights)?;
    let output = g.value(combined)?;
    let seed = g.engine.full(&[4, 5], DType::F32, 1.)?;
    g.backward_seeded(&[(combined, seed)])?;
    let dx = g.gradient(xv).unwrap().clone();
    let dl = g.gradient(lv).unwrap().clone();
    drop(g);
    let mut expected = vec![0.; 20];
    let mut expected_dx = expected.clone();
    let mut expected_dl = vec![0.; 12];
    for row in 0..4 {
        let mut order = (0..3).collect::<Vec<_>>();
        order.sort_by(|a, b| logits_values[row * 3 + *b].total_cmp(&logits_values[row * 3 + *a]));
        let selected = &order[..2];
        let norm = (0..3)
            .map(|j| logits_values[row * 3 + j].exp())
            .sum::<f32>();
        let factor = selected
            .iter()
            .map(|j| logits_values[row * 3 + j].exp() / norm * (*j + 1) as f32)
            .sum::<f32>();
        let sum = input_values[row * 5..row * 5 + 5].iter().sum::<f32>();
        for j in 0..3 {
            let upstream = if selected.contains(&j) {
                (j + 1) as f32
            } else {
                0.
            };
            expected_dl[row * 3 + j] =
                logits_values[row * 3 + j].exp() / norm * sum * (upstream - factor);
        }
        for col in 0..5 {
            expected[row * 5 + col] = input_values[row * 5 + col] * factor;
            expected_dx[row * 5 + col] = factor;
        }
    }
    let router_error = max_error(&output, &expected)
        .max(max_error(&e.to_f32(&dx)?, &expected_dx))
        .max(max_error(&e.to_f32(&dl)?, &expected_dl));
    ensure(router_error < 3e-6, "packed expert/router gradients")?;

    let logits = e.from_f32(&[4, 3], &logits_values)?;
    let mut g = Graph::new(&mut e, true);
    let logits_var = g.leaf(logits, true);
    let (gate, _) = g.route_topk(logits_var, 1)?;
    let seed = g.engine.full(&[4, 1], DType::F32, 1.)?;
    g.backward_seeded(&[(gate, seed)])?;
    let gradient = g.gradient(logits_var).unwrap().clone();
    drop(g);
    let mut top1_reference = vec![0.; 12];
    for row in 0..4 {
        let best = (0..3)
            .max_by(|a, b| logits_values[row * 3 + *a].total_cmp(&logits_values[row * 3 + *b]))
            .unwrap();
        let norm = (0..3)
            .map(|j| logits_values[row * 3 + j].exp())
            .sum::<f32>();
        let selected = logits_values[row * 3 + best].exp() / norm;
        for j in 0..3 {
            top1_reference[row * 3 + j] =
                logits_values[row * 3 + j].exp() / norm * ((j == best) as u8 as f32 - selected);
        }
    }
    let top1_gradient = e.to_f32(&gradient)?;
    let top1_error = max_error(&top1_gradient, &top1_reference);
    let top1_gradient_norm = top1_gradient.iter().map(|x| x * x).sum::<f32>().sqrt();
    ensure(
        top1_error < 2e-6 && top1_gradient_norm > 0.,
        "top-1 router derivative",
    )?;

    let u0 = e.from_f32(&[7, 3], &values(21, 3))?;
    let u = e.orthonormalize_columns(&u0)?;
    let v0 = e.from_f32(&[9, 3], &values(27, 7))?;
    let v = e.orthonormalize_columns(&v0)?;
    let gram = e.matmul(&u, &u, true, false, DType::F32)?;
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let orthogonal_error = max_error(&e.to_f32(&gram)?, &identity);
    ensure(orthogonal_error < 2e-5, "orthogonal basis comparison")?;
    let a = e.from_f32(&[3, 3], &values(9, 11))?;
    let w = e.from_f32(&[7, 9], &values(63, 4))?;
    let x = e.from_f32(&[5, 7], &values(35, 5))?;
    let dy = e.from_f32(&[5, 9], &values(45, 2))?;
    let target_left = e.matmul(&x, &u, false, false, DType::F32)?;
    let target_right = e.matmul(&dy, &v, false, false, DType::F32)?;
    let target = e.matmul(&target_left, &target_right, true, false, DType::F32)?;
    let correction = e.matmul(&u, &a, false, false, DType::F32)?;
    let correction = e.matmul(&correction, &v, false, true, DType::F32)?;
    let effective = e.add(&w, &correction, 1., DType::F32)?;
    let reference = e.matmul(&x, &effective, false, false, DType::F32)?;
    let reference_dx = e.matmul(&dy, &effective, false, true, DType::F32)?;
    let mut g = Graph::new(&mut e, true);
    let xv = g.leaf(x, true);
    let wv = g.leaf(w, true);
    let output = g.checkpoint(&[xv, wv], move |nested, inputs| {
        nested.low_rank_matmul(inputs[0], inputs[1], &u, &a, &v)
    })?;
    let actual = g.value(output)?;
    g.backward_seeded(&[(output, dy)])?;
    let actual_dx = g.gradient(xv).unwrap().clone();
    let actual_target = g.projected_gradient(wv).unwrap().clone();
    drop(g);
    let low_rank_error = max_error(&actual, &e.to_f32(&reference)?)
        .max(max_error(&e.to_f32(&actual_dx)?, &e.to_f32(&reference_dx)?))
        .max(max_error(&e.to_f32(&actual_target)?, &e.to_f32(&target)?));
    ensure(
        low_rank_error < 2e-5,
        "checkpoint low-rank forward/dX/projected target",
    )?;
    let transpose = e.transpose2d(&target)?;
    let kronecker = e.kronecker(&target, &transpose)?;
    ensure(
        e.dot(&kronecker, &kronecker)?.is_finite(),
        "basis coordinate transform",
    )?;
    let mean = e.row_mean(&effective)?;
    ensure(
        e.to_f32(&mean)?.iter().all(|v| v.is_finite()),
        "feature mean",
    )?;
    let coefficient_values = values(18, 13);
    let left_values = values(9, 10);
    let right_values = values(9, 19);
    let coefficients = e.from_f32(&[2, 9], &coefficient_values)?;
    let left = e.from_f32(&[3, 3], &left_values)?;
    let right = e.from_f32(&[3, 3], &right_values)?;
    let transported = e.transport_coefficients(&coefficients, &left, &right)?;
    let mut transported_reference = vec![0.; 18];
    for feature in 0..2 {
        for row in 0..3 {
            for column in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        transported_reference[feature * 9 + row * 3 + column] += left_values
                            [i * 3 + row]
                            * coefficient_values[feature * 9 + i * 3 + j]
                            * right_values[j * 3 + column];
                    }
                }
            }
        }
    }
    let transport_error = max_error(&e.to_f32(&transported)?, &transported_reference);
    ensure(transport_error < 2e-6, "coefficient coordinate transport")?;
    let c = e.execution_counters();
    println!(
        "{}",
        json::to_string_pretty(
            &json::json!({"device":e.name(),"attention_error":attention_error,
        "cache_error":cache_error,"patch_error":patch_error,"router_error":router_error,
        "top1_gradient_error":top1_error,"top1_gradient_norm":top1_gradient_norm,
        "orthogonal_error":orthogonal_error,"low_rank_error":low_rank_error,"transport_error":transport_error,
        "launches":c.launches,"logical_flops":c.logical_flops,"issued_flops":c.issued_flops,
        "compilations":c.compilations,"compile_seconds":c.compile_seconds})
        )?
    );
    Ok(())
}
