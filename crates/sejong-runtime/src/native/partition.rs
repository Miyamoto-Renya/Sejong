//! Equality-partition tensor contractions with input-sized rank and byte accounting.
use super::resources;
use sejong_core::{Result, ensure, numeric};
fn length(size: usize, arity: usize, limit: usize) -> Result<usize> {
    ensure(
        size > 0 && arity > 0,
        "joint requires nonempty dimensions and rank",
    )?;
    let entries = resources::power(size, arity)?;
    resources::require(
        limit,
        resources::bytes::<f64>(entries)?.saturating_add(resources::bytes::<usize>(arity)?),
        "joint tensor",
    )?;
    Ok(entries)
}
fn coordinates(mut index: usize, size: usize, arity: usize) -> Vec<usize> {
    let mut result = vec![0; arity];
    for i in (0..arity).rev() {
        result[i] = index % size;
        index /= size;
    }
    result
}
fn flatten(xs: impl IntoIterator<Item = usize>, size: usize) -> usize {
    xs.into_iter().fold(0, |n, x| n * size + x)
}
/// Restricted-growth strings generated on demand; impossible block counts are pruned.
pub struct Partitions {
    current: Vec<usize>,
    first: bool,
    done: bool,
    max_blocks: usize,
}
impl Iterator for Partitions {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.first {
            self.first = false;
            return Some(self.current.clone());
        }
        for i in (1..self.current.len()).rev() {
            let max = self.current[..i].iter().copied().max().unwrap_or(0);
            let allowed = (max + 1).min(self.max_blocks.saturating_sub(1));
            if self.current[i] < allowed {
                self.current[i] += 1;
                self.current[i + 1..].fill(0);
                return Some(self.current.clone());
            }
        }
        self.done = true;
        None
    }
}
pub fn partitions(arity: usize) -> Partitions {
    partitions_bounded(arity, arity)
}
fn partitions_bounded(arity: usize, max_blocks: usize) -> Partitions {
    Partitions {
        current: vec![0; arity],
        first: arity > 0 && max_blocks > 0,
        done: arity == 0 || max_blocks == 0,
        max_blocks,
    }
}
fn distinct(coords: &[usize]) -> bool {
    coords
        .iter()
        .enumerate()
        .all(|(i, x)| coords[..i].iter().all(|y| x != y))
}
fn tensor_bytes(entries: usize) -> Result<usize> {
    resources::bytes::<f64>(entries)
}
fn contract(
    data: &[f64],
    dimensions: &mut [usize],
    axis: usize,
    matrix: &[f64],
    output: usize,
    limit: usize,
    base: usize,
) -> Result<Vec<f64>> {
    let input = dimensions[axis];
    let before = resources::product(&dimensions[..axis])?;
    let after = resources::product(&dimensions[axis + 1..])?;
    let entries = resources::product(&[before, output, after])?;
    resources::require(
        limit,
        base.saturating_add(tensor_bytes(data.len())?)
            .saturating_add(tensor_bytes(entries)?),
        "joint axis contraction workspace",
    )?;
    let mut result = resources::vector(entries, 0., limit, "joint axis contraction")?;
    for b in 0..before {
        for i in 0..input {
            for j in 0..output {
                let factor = matrix[i * output + j];
                if factor == 0. {
                    continue;
                }
                let source = (b * input + i) * after;
                let target = (b * output + j) * after;
                for a in 0..after {
                    result[target + a] += data[source + a] * factor;
                }
            }
        }
    }
    dimensions[axis] = output;
    Ok(result)
}
fn check(
    joint: &[f64],
    input: usize,
    output: usize,
    arity: usize,
    kernel: &[f64],
    limit: usize,
) -> Result<()> {
    ensure(
        joint.len() == length(input, arity, limit)?
            && kernel.len() == resources::product(&[input, output])?,
        "joint/kernel dimensions differ",
    )?;
    length(output, arity, limit)?;
    ensure(
        joint.iter().all(|p| p.is_finite() && *p >= 0.)
            && kernel.chunks_exact(output).all(resources::normalized),
        "finite nonnegative normalized tensor kernel required",
    )?;
    Ok(())
}
pub fn pushforward(
    joint: &[f64],
    input: usize,
    kernel: &[f64],
    output: usize,
    arity: usize,
    limit: usize,
) -> Result<Vec<f64>> {
    check(joint, input, output, arity, kernel, limit)?;
    let count = length(output, arity, limit)?;
    let base = tensor_bytes(joint.len())?
        .saturating_add(tensor_bytes(kernel.len())?)
        .saturating_add(tensor_bytes(count)?)
        .saturating_add(resources::bytes::<usize>(arity.saturating_mul(5))?);
    resources::require(limit, base, "joint transport live storage")?;
    let mut result = resources::vector(count, 0., limit, "joint output")?;
    for pattern in partitions_bounded(arity, input) {
        let blocks = pattern.iter().max().copied().unwrap() + 1;
        let entries = resources::power(input, blocks)?;
        resources::require(
            limit,
            base.saturating_add(tensor_bytes(entries)?),
            "joint equality selection",
        )?;
        let mut selected = resources::vector(entries, 0., limit, "joint equality selection")?;
        for (index, weight) in selected.iter_mut().enumerate() {
            let x = coordinates(index, input, blocks);
            if distinct(&x) {
                *weight = joint[flatten(pattern.iter().map(|b| x[*b]), input)];
            }
        }
        let mut dimensions = vec![input; blocks];
        for axis in 0..blocks {
            selected = contract(
                &selected,
                &mut dimensions,
                axis,
                kernel,
                output,
                limit,
                base,
            )?;
        }
        for (index, p) in selected.into_iter().enumerate() {
            let y = coordinates(index, output, blocks);
            let target = flatten(pattern.iter().map(|b| y[*b]), output);
            result[target] += p;
        }
    }
    Ok(result)
}
/// Pull a terminal observable backward without building the terminal joint law.
pub fn pullback(
    observable: &[f64],
    output: usize,
    kernel: &[f64],
    input: usize,
    arity: usize,
    limit: usize,
) -> Result<Vec<f64>> {
    let count = length(input, arity, limit)?;
    ensure(
        observable.len() == length(output, arity, limit)?
            && observable.iter().all(|v| v.is_finite())
            && kernel.len() == resources::product(&[input, output])?
            && kernel.chunks_exact(output).all(resources::normalized),
        "observable/kernel contract differs",
    )?;
    let base = tensor_bytes(observable.len())?
        .saturating_add(tensor_bytes(kernel.len())?.saturating_mul(2))
        .saturating_add(tensor_bytes(count)?)
        .saturating_add(resources::bytes::<usize>(arity.saturating_mul(5))?);
    resources::require(limit, base, "joint pullback live storage")?;
    let mut transpose = resources::vector(kernel.len(), 0., limit, "joint transposed kernel")?;
    for i in 0..input {
        for j in 0..output {
            transpose[j * input + i] = kernel[i * output + j];
        }
    }
    let mut result = resources::vector(count, 0., limit, "joint pullback output")?;
    for pattern in partitions_bounded(arity, input) {
        let blocks = pattern.iter().max().copied().unwrap() + 1;
        let entries = resources::power(output, blocks)?;
        resources::require(
            limit,
            base.saturating_add(tensor_bytes(entries)?),
            "joint observable selection",
        )?;
        let mut selected = resources::vector(entries, 0., limit, "joint observable selection")?;
        for (index, v) in selected.iter_mut().enumerate() {
            let y = coordinates(index, output, blocks);
            *v = observable[flatten(pattern.iter().map(|b| y[*b]), output)];
        }
        let mut dimensions = vec![output; blocks];
        for axis in 0..blocks {
            selected = contract(
                &selected,
                &mut dimensions,
                axis,
                &transpose,
                input,
                limit,
                base,
            )?;
        }
        for (index, p) in selected.into_iter().enumerate() {
            let x = coordinates(index, input, blocks);
            if distinct(&x) {
                result[flatten(pattern.iter().map(|b| x[*b]), input)] += p;
            }
        }
    }
    Ok(result)
}
pub fn bounded_pushforward(
    joint: &[f64],
    input: usize,
    kernel: &[f64],
    output: usize,
    arity: usize,
    radii: &[f64],
    previous_error: f64,
    limit: usize,
) -> Result<(Vec<f64>, f64)> {
    ensure(
        radii.len() == input
            && radii
                .iter()
                .all(|r| r.is_finite() && (0. ..=1.).contains(r))
            && previous_error.is_finite()
            && (0. ..=1.).contains(&previous_error),
        "valid joint read-error radii required",
    )?;
    let transported = pushforward(joint, input, kernel, output, arity, limit)?;
    let mut increment = 0.;
    for (index, mass) in joint.iter().copied().enumerate() {
        if mass == 0. {
            continue;
        }
        let mut rows = coordinates(index, input, arity);
        rows.sort_unstable();
        rows.dedup();
        let mut survival = 1.;
        for r in rows {
            if radii[r] > 0. {
                survival = numeric::next_down(survival * numeric::next_down(1. - radii[r]).max(0.))
                    .max(0.);
            }
        }
        let error = if survival == 1. {
            0.
        } else {
            numeric::next_up(1. - survival).min(1.)
        };
        if error > 0. {
            increment = numeric::next_up(increment + numeric::next_up(mass * error));
        }
    }
    let total = if increment == 0. {
        previous_error
    } else {
        numeric::next_up(previous_error + increment).min(1.)
    };
    Ok((transported, total))
}

/// Floating operations performed by the equality-partition contractions and scatter.
pub fn operation_bound(input: usize, output: usize, arity: usize) -> usize {
    let mut count = 0usize;
    for pattern in partitions_bounded(arity, input) {
        let blocks = pattern.iter().max().copied().unwrap_or(0) + 1;
        for axis in 0..blocks {
            let before = (0..axis).fold(1usize, |n, _| n.saturating_mul(output));
            let after = (axis + 1..blocks).fold(1usize, |n, _| n.saturating_mul(input));
            count = count.saturating_add(
                before
                    .saturating_mul(input)
                    .saturating_mul(output)
                    .saturating_mul(after)
                    .saturating_mul(2),
            );
        }
        count = count.saturating_add((0..blocks).fold(1usize, |n, _| n.saturating_mul(output)));
    }
    count
}
