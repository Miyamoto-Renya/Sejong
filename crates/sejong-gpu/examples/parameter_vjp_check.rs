//! 비정렬 행렬에서 실제 기저·선택·잔차·인수 SGD를 FP64 수식과 대조한다.
use sejong_core::{DType, GpuConfig, ParameterVjpConfig, Result, Rng, ensure, json};
use sejong_gpu::{
    Engine,
    parameter_vjp::{GradientPlan, ParameterGradient},
};

fn product(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            for z in 0..k {
                y[i * n + j] += a[i * k + z] * b[z * n + j];
            }
        }
    }
    y
}
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..n)
        .flat_map(|j| (0..m).map(move |i| a[i * n + j]))
        .collect()
}
fn errors(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut max = 0f64;
    let mut sum = 0.;
    let mut norm = 0.;
    for (a, b) in a.iter().zip(b) {
        max = max.max((a - b).abs());
        sum += (a - b).powi(2);
        norm += b * b;
    }
    (max, sum / norm)
}
fn main() -> Result<()> {
    let mut e = Engine::new(GpuConfig::default())?;
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.is_empty() || args.len() == 4,
        "usage: parameter_vjp_check [N A B RANK]",
    )?;
    let (n, a, b, rank) = if args.is_empty() {
        (67, 71, 97, 33)
    } else {
        let values = args
            .iter()
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|_| sejong_core::Error::Message("invalid matrix dimension".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        (values[0], values[1], values[2], values[3])
    };
    ensure(
        rank > 0 && rank <= n.min(a).min(b),
        "rank must fit all matrix dimensions",
    )?;
    let policy = ParameterVjpConfig {
        rank,
        residual_fraction: 0.4,
        block_rows: 16,
        seed: 42,
    };
    let mut data = Rng::new(42);
    let xv: Vec<_> = (0..n * a).map(|_| data.normal()).collect();
    let wv: Vec<_> = (0..a * b)
        .map(|_| data.normal() / (a as f32).sqrt())
        .collect();
    let dv: Vec<_> = (0..n * b).map(|_| data.normal() / (n as f32)).collect();
    let x = e.from_f32(&[n, a], &xv)?;
    let w = e.from_f32(&[a, b], &wv)?;
    let d = e.from_f32(&[n, b], &dv)?;
    let y = e.matmul(&x, &w, false, false, DType::F32)?;
    let mut rng = Rng::new(policy.seed);
    let mut expected_rng = rng.clone();
    let mut selection_rng = Rng::new(expected_rng.next_u64());
    let mut columns: Vec<_> = (0..b).collect();
    for i in 0..rank {
        let j = i + selection_rng.index(b - i);
        columns.swap(i, j);
    }
    let mut blocks: Vec<_> = (0..a.div_ceil(policy.block_rows)).collect();
    let count = (blocks.len() as f64 * policy.residual_fraction).ceil() as usize;
    for i in 0..count {
        let j = i + selection_rng.index(blocks.len() - i);
        blocks.swap(i, j);
    }
    let expected_rows: Vec<_> = blocks[..count]
        .iter()
        .flat_map(|&i| {
            (0..policy.block_rows).map(move |j| {
                let row = i * policy.block_rows + j;
                if row < a { row as i32 } else { -1 }
            })
        })
        .collect();
    let prepared = e.prepare_parameter_vjp(&x, &y, &policy, &mut rng)?;
    ensure(
        e.to_i32(&prepared.selected_columns)?
            == columns[..rank]
                .iter()
                .map(|&v| v as i32)
                .collect::<Vec<_>>(),
        "GPU basis selection differs from exact uniform CPU generator",
    )?;
    let selected = e.to_i32(&prepared.selected_rows)?;
    ensure(
        selected == expected_rows,
        "GPU residual block selection differs from CPU generator",
    )?;
    let q: Vec<f64> = e.to_f32(&prepared.q)?.into_iter().map(f64::from).collect();
    let x64: Vec<_> = xv.iter().map(|&v| f64::from(v)).collect();
    let d64: Vec<_> = dv.iter().map(|&v| f64::from(v)).collect();
    let xt = transpose(&x64, n, a);
    let qt = transpose(&q, n, rank);
    let left = product(&xt, &q, a, rank, n);
    let right = product(&qt, &d64, rank, b, n);
    let projected = product(&q, &transpose(&left, a, rank), n, a, rank);
    let mut expected = product(&left, &right, a, b, rank);
    let inverse = blocks.len() as f64 / count as f64;
    for &row in &selected {
        if row >= 0 {
            let row = row as usize;
            for j in 0..b {
                for i in 0..n {
                    expected[row * b + j] +=
                        (x64[i * a + row] - projected[i * a + row]) * d64[i * b + j] * inverse;
                }
            }
        }
    }
    let term = e.parameter_vjp(&prepared, &d)?;
    let mut gradient = ParameterGradient::default();
    gradient.terms.push(term);
    let materialized = GradientPlan::materialize(&mut e, &gradient, &[a, b])?;
    let actual: Vec<f64> = e
        .to_f32(&materialized)?
        .into_iter()
        .map(f64::from)
        .collect();
    let (maximum, relative) = errors(&actual, &expected);
    let operations = (2 * n * rank + 2 * n + rank) as f64;
    let tolerance = operations * f32::EPSILON as f64 / (1. - operations * f32::EPSILON as f64);
    ensure(
        relative < tolerance * tolerance,
        "factored GPU VJP exceeds its shape-derived roundoff budget",
    )?;
    let mut second_policy = policy.clone();
    second_policy.rank = rank.div_ceil(2);
    let second = e.prepare_parameter_vjp(&x, &y, &second_policy, &mut rng)?;
    let r2 = second.q.shape[1];
    let q2: Vec<f64> = e.to_f32(&second.q)?.into_iter().map(f64::from).collect();
    let a2 = product(&xt, &q2, a, r2, n);
    let b2 = product(&transpose(&q2, n, r2), &d64, r2, b, n);
    let projected2 = product(&q2, &transpose(&a2, a, r2), n, a, r2);
    let mut expected2 = product(&a2, &b2, a, b, r2);
    for row in e.to_i32(&second.selected_rows)? {
        if row >= 0 {
            let row = row as usize;
            for j in 0..b {
                for i in 0..n {
                    expected2[row * b + j] +=
                        (x64[i * a + row] - projected2[i * a + row]) * d64[i * b + j] * inverse;
                }
            }
        }
    }
    let second_term = e.parameter_vjp(&second, &d)?;
    let mut combined = ParameterGradient {
        dense: Some(e.scale(&w, 1.)?),
        terms: vec![],
    };
    combined.accumulate(&mut e, &gradient, 0.5)?;
    combined.accumulate(
        &mut e,
        &ParameterGradient {
            dense: None,
            terms: vec![second_term],
        },
        -0.25,
    )?;
    let expected: Vec<_> = wv
        .iter()
        .zip(&expected)
        .zip(&expected2)
        .map(|((&w, &first), &second)| f64::from(w) + 0.5 * first - 0.25 * second)
        .collect();
    let plan = GradientPlan::new(&mut e, std::slice::from_ref(&w))?;
    let packed = plan.pack(&mut e, std::slice::from_ref(&combined))?;
    let norm = plan.norm(&mut e, &packed)?;
    let reference_norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure(
        (norm - reference_norm).abs() <= tolerance * reference_norm,
        "compact gradient norm differs from reference",
    )?;
    let q64 = e.from_f64(&[n, rank], &q)?;
    let x16 = e.cast(&x, DType::F16)?;
    let x16_host = e.to_f32(&x16)?;
    let mixed_reference = (wv.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
        + q.iter().map(|v| v * v).sum::<f64>()
        + x16_host.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>())
    .sqrt();
    let mixed_norm = e.gradient_norm(&[Some(w.clone()), Some(q64), None, Some(x16)])?;
    ensure(
        (mixed_norm - mixed_reference).abs() <= tolerance * mixed_reference,
        "mixed precision global gradient norm differs",
    )?;
    let lr = 0.01;
    let clipping = 0.7;
    let decay = 0.02;
    plan.update(&mut e, &packed, lr, clipping, decay)?;
    let updated: Vec<f64> = e.to_f32(&w)?.into_iter().map(f64::from).collect();
    let expected_weight: Vec<_> = wv
        .iter()
        .zip(&expected)
        .map(|(&w, &g)| {
            f64::from(w) * (1. - f64::from(lr) * f64::from(decay))
                - f64::from(lr) * f64::from(clipping) * g
        })
        .collect();
    let (weight_error, weight_relative) = errors(&updated, &expected_weight);
    ensure(
        weight_relative < tolerance * tolerance,
        "compact SGD update exceeds roundoff budget",
    )?;
    let gram = product(&qt, &q, rank, rank, n);
    let mut orthogonality = 0f64;
    for i in 0..rank {
        for j in 0..rank {
            orthogonality = orthogonality.max((gram[i * rank + j] - f64::from(i == j)).abs());
        }
    }
    ensure(
        orthogonality < tolerance,
        "streaming CGS2 orthogonality exceeds roundoff budget",
    )?;
    let large = e.from_f32(&[4, 4], &[f32::MAX / 2.; 16])?;
    let zero = e.full(&[4, 4], DType::F32, 0.)?;
    let large_plan = GradientPlan::new(&mut e, std::slice::from_ref(&zero))?;
    let large_packed = large_plan.pack(
        &mut e,
        &[ParameterGradient {
            dense: Some(large),
            terms: vec![],
        }],
    )?;
    let large_norm = large_plan.norm(&mut e, &large_packed)?;
    large_plan.update(&mut e, &large_packed, 2f32.powi(-10), 1., 0.)?;
    let large_result = e.to_f32(&zero)?;
    ensure(
        large_norm > f32::MAX as f64 && large_result.iter().all(|v| *v == -f32::MAX / 2048.),
        "finite large-gradient update differs",
    )?;
    println!(
        "{}",
        json::json!({"device":e.name(),"shape":[n,a,b],"rank":rank,"second_rank":r2,"factored_terms":2,"dense_term":true,
        "vjp_max_abs_error":maximum,"vjp_relative_squared_error":relative,"sgd_max_abs_error":weight_error,
        "sgd_relative_squared_error":weight_relative,"basis_orthogonality_max_abs_error":orthogonality,
        "gradient_norm":norm,"mixed_precision_norm":mixed_norm,"large_gradient_norm":large_norm,"large_gradient_update_finite":true,"roundoff_budget":tolerance})
    );
    Ok(())
}
