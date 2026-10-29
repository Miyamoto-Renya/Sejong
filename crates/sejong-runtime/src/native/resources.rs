//! Storage accounting in bytes for input-dependent exact inference.
use sejong_core::{Error, Result};
pub fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| Error::Message("inference storage size overflow".into()))
    })
}
pub fn bytes<T>(entries: usize) -> Result<usize> {
    product(&[entries, std::mem::size_of::<T>()])
}
pub fn power(size: usize, arity: usize) -> Result<usize> {
    let mut n = 1usize;
    for _ in 0..arity {
        n = n
            .checked_mul(size)
            .ok_or_else(|| Error::Message("joint tensor shape overflows address space".into()))?;
    }
    Ok(n)
}
pub fn require(limit: usize, required: usize, operation: &str) -> Result<()> {
    if required > limit {
        Err(Error::Capacity {
            operation: operation.into(),
            required,
            available: limit,
        })
    } else {
        Ok(())
    }
}
pub fn sum_error(values: &[f64], extra_ops: usize) -> f64 {
    sejong_core::numeric::normalized_tolerance(values, extra_ops)
}
pub fn normalized(values: &[f64]) -> bool {
    let mass = values.iter().sum::<f64>();
    values.iter().all(|v| v.is_finite() && *v >= 0.)
        && (mass - 1.).abs() <= sum_error(values, values.len())
}
pub fn vector<T: Clone>(length: usize, value: T, limit: usize, operation: &str) -> Result<Vec<T>> {
    require(limit, bytes::<T>(length)?, operation)?;
    let mut out = Vec::new();
    out.try_reserve_exact(length)
        .map_err(|e| Error::Message(format!("{operation} allocation: {e}")))?;
    out.resize(length, value);
    Ok(out)
}
