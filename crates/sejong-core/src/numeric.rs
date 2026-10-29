//! IEEE-754 scale- and operation-count-dependent rounding allowances.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0. {
        return f64::from_bits(1);
    }
    f64::from_bits(if x > 0. {
        x.to_bits() + 1
    } else {
        x.to_bits() - 1
    })
}
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}
pub fn gamma(operations: usize) -> f64 {
    if operations == 0 {
        return 0.;
    }
    let product = (operations as f64) * (f64::EPSILON / 2.);
    if product >= 1. {
        f64::INFINITY
    } else {
        next_up(product / (1. - product))
    }
}
pub fn sum_error(abs_sum: f64, operations: usize) -> f64 {
    if operations == 0 {
        return 0.;
    }
    next_up(
        next_up(gamma(operations) * abs_sum.abs())
            + next_up(f64::from_bits(1) * (operations as f64)),
    )
}
pub fn normalized_tolerance(values: &[f64], operations: usize) -> f64 {
    sum_error(
        values.iter().map(|x| x.abs()).sum::<f64>(),
        operations.saturating_add(values.len()),
    )
}
