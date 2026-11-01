//! 순전파 기저의 parameter VJP와 인수·선택 행 gradient를 소비하는 SGD.
use crate::{Arg, Engine, Tensor, n, p};
use sejong_core::{DType, Error, ParameterVjpConfig, Result, Rng, ensure, json};

pub const BM: usize = 16;
pub const BN: usize = 32;
pub const BK: usize = 32;

pub fn issued_gemm(m: usize, n: usize, k: usize) -> u128 {
    2 * (m.div_ceil(BM) as u128 * BM as u128)
        * (n.div_ceil(BN) as u128 * BN as u128)
        * (k.div_ceil(BK) as u128 * BK as u128)
}

/// G = scale * (left right + scatterRows(tail)).
#[derive(Clone)]
pub struct FactoredRows {
    pub left: Tensor,
    pub right: Tensor,
    pub tail: Tensor,
    pub row_map: Tensor,
    pub scale: f32,
}

#[derive(Clone, Default)]
pub struct ParameterGradient {
    pub dense: Option<Tensor>,
    pub terms: Vec<FactoredRows>,
}
impl ParameterGradient {
    pub fn accumulate(&mut self, e: &mut Engine, incoming: &Self, scale: f32) -> Result<()> {
        if let Some(value) = &incoming.dense {
            if let Some(current) = &self.dense {
                e.add_into(current, value, current, scale)?;
            } else {
                self.dense = Some(e.scale(value, scale)?);
            }
        }
        for term in &incoming.terms {
            let mut term = term.clone();
            term.scale *= scale;
            self.terms.push(term);
        }
        Ok(())
    }
}

pub struct PreparedVjp {
    pub q: Tensor,
    pub left: Tensor,
    pub projection: Tensor,
    pub row_map: Tensor,
    pub selected_rows: Tensor,
    pub selected_columns: Tensor,
    pub basis_donor_width: usize,
}
#[derive(Clone)]
pub(crate) struct TokenBasis {
    pub q: Tensor,
    pub columns: Tensor,
    pub donor_width: usize,
}
#[derive(Clone)]
pub(crate) struct FeatureSelection {
    pub rows: Tensor,
    pub map: Tensor,
}

#[derive(Clone, Default)]
pub struct VjpStatistics {
    pub compressed_matrices: usize,
    pub dense_matrices: usize,
    pub basis_groups: usize,
    pub projection_groups: usize,
}
impl VjpStatistics {
    pub fn add(&mut self, other: &Self) {
        self.compressed_matrices += other.compressed_matrices;
        self.dense_matrices += other.dense_matrices;
        self.basis_groups += other.basis_groups;
        self.projection_groups += other.projection_groups;
    }
}

pub fn budget(
    config: &ParameterVjpConfig,
    n: usize,
    a: usize,
    donor: usize,
) -> (usize, usize, usize) {
    let rank = config.rank.min(n).min(a).min(donor);
    let blocks = a.div_ceil(config.block_rows);
    let selected = ((blocks as f64 * config.residual_fraction).ceil() as usize)
        .min(blocks)
        .max(1);
    (rank, blocks, selected)
}

/// 같은 입력을 쓰는 후보의 공유 비용을 실제 타일 형상에서 계산한다.
pub fn selects_compression(
    config: &ParameterVjpConfig,
    n: usize,
    a: usize,
    b: usize,
    donor: usize,
    uses: usize,
) -> bool {
    // 기저를 공유하는 긴 입력에서 준비 비용을 분산한다.
    if n < a || uses <= 1 {
        return false;
    }
    let (rank, blocks, selected) = budget(config, n, a, donor);
    if selected == blocks {
        return false;
    }
    let k = selected * config.block_rows;
    let padded = n.div_ceil(BM) as u128 * BM as u128;
    let last = rank.saturating_sub(1) as u128;
    let full = last / BN as u128;
    let tail = last % BN as u128;
    let blocks = BN as u128 * full * (full + 1) / 2 + tail * (full + 1);
    let streaming = 2 * (4 * padded * BN as u128 + padded) * blocks + 5 * padded * rank as u128;
    let resident =
        8 * n.next_power_of_two() as u128 * rank.next_power_of_two() as u128 * rank as u128
            + 5 * n.next_power_of_two() as u128 * rank as u128;
    let qr = streaming.max(resident);
    let shared = qr + issued_gemm(a, rank, n) + issued_gemm(k, n, rank);
    let own = issued_gemm(rank + k, b, n) + issued_gemm(a, b, rank);
    own * uses as u128 + shared < issued_gemm(a, b, n) * uses as u128
}

impl Engine {
    pub fn prepare_parameter_vjp(
        &mut self,
        x: &Tensor,
        donor: &Tensor,
        config: &ParameterVjpConfig,
        rng: &mut Rng,
    ) -> Result<PreparedVjp> {
        config.validate()?;
        ensure(
            x.shape.len() == 2 && donor.shape.len() == 2 && x.shape[0] == donor.shape[0],
            "forward-basis input and donor rows must match",
        )?;
        let (_, blocks, selected) = budget(config, x.shape[0], x.shape[1], donor.shape[1]);
        ensure(
            selected < blocks,
            "compact residual requires a proper block subset",
        )?;
        self.prepare_parameter_vjp_reusing(x, donor, config, rng, None, None)
    }
    pub(crate) fn prepare_parameter_vjp_reusing(
        &mut self,
        x: &Tensor,
        donor: &Tensor,
        config: &ParameterVjpConfig,
        rng: &mut Rng,
        basis: Option<&TokenBasis>,
        selection: Option<&FeatureSelection>,
    ) -> Result<PreparedVjp> {
        let (rows, a, b) = (x.shape[0], x.shape[1], donor.shape[1]);
        let (rank, blocks, selected) = budget(config, rows, a, b);
        let k = selected
            .checked_mul(config.block_rows)
            .ok_or_else(|| Error::Message("selected row size overflow".into()))?;
        let columns = if let Some(basis) = basis {
            basis.columns.clone()
        } else {
            self.empty(&[rank], DType::I32)?
        };
        let selected_rows = if let Some(selection) = selection {
            selection.rows.clone()
        } else {
            self.empty(&[k], DType::I32)?
        };
        let row_map = if let Some(selection) = selection {
            selection.map.clone()
        } else {
            self.empty(&[a], DType::I32)?
        };
        if basis.is_none() || selection.is_none() {
            let basis_width = if basis.is_none() { b } else { 0 };
            let feature_blocks = if selection.is_none() { blocks } else { 0 };
            let scratch = self.empty(
                &[basis_width
                    .checked_add(feature_blocks)
                    .ok_or_else(|| Error::Message("selection scratch overflow".into()))?],
                DType::I32,
            )?;
            let seed = rng.next_u64();
            self.launch(
                "parameter_indices",
                &[
                    ("Scratch", p(&scratch)),
                    ("Columns", p(&columns)),
                    ("Rows", p(&selected_rows)),
                    ("Map", p(&row_map)),
                    ("A", n(if selection.is_none() { a } else { 0 })?),
                    ("B", n(basis_width)?),
                    ("NB", n(feature_blocks)?),
                    ("S", n(if selection.is_none() { selected } else { 0 })?),
                    ("R", n(if basis.is_none() { rank } else { 0 })?),
                    ("SEED_LO", Arg::I32(seed as u32 as i32)),
                    ("SEED_HI", Arg::I32((seed >> 32) as u32 as i32)),
                ],
                json::json!({"GROUP":config.block_rows,"BLOCK":BM*BN}),
                [1, 1, 1],
            )?;
        }
        let q = if let Some(basis) = basis {
            basis.q.clone()
        } else {
            let q = self.empty(&[rows, rank], DType::F32)?;
            let operations = rows as f64 * (4. * rank as f64 + 1.);
            let roundoff = operations * (f32::EPSILON as f64 / 2.);
            let tolerance = if roundoff < 1. {
                roundoff / (1. - roundoff)
            } else {
                f64::INFINITY
            };
            let nr = rows.next_power_of_two();
            let nc = rank.next_power_of_two();
            // basis·곱·선택의 세 register 행렬과 세 row vector, coefficient vector의 작업량.
            let resident = nr
                .checked_mul(nc)
                .and_then(|v| v.checked_mul(3))
                .and_then(|v| v.checked_add(3 * nr + nc))
                .is_some_and(|v| v <= self.device.registers_per_block);
            let (operation, constants) = if resident {
                (
                    "parameter_basis_register",
                    json::json!({"R":rank,"NR":nr,"NC":nc}),
                )
            } else {
                ("parameter_basis", json::json!({"R":rank,"BR":BM,"BC":BN}))
            };
            let arguments = [
                ("Y", p(donor)),
                ("Columns", p(&columns)),
                ("Q", p(&q)),
                ("N", n(rows)?),
                ("B", n(b)?),
                ("EPS", Arg::F32(tolerance.min(f32::MAX as f64) as f32)),
            ];
            match self.launch(operation, &arguments, constants, [1, 1, 1]) {
                Err(Error::Capacity { operation, .. })
                    if operation == "parameter_basis_register shared memory" =>
                {
                    self.launch(
                        "parameter_basis",
                        &arguments,
                        json::json!({"R":rank,"BR":BM,"BC":BN}),
                        [1, 1, 1],
                    )?;
                }
                result => result?,
            }
            q
        };
        let projection = self.empty(&[rank + k, rows], DType::F32)?;
        let fused = rank <= BN
            && BN == BK
            && config.block_rows.is_multiple_of(BM)
            && a.is_multiple_of(config.block_rows);
        let left = if fused {
            self.empty(&[a, rank], DType::F32)?
        } else {
            self.matmul(x, &q, true, false, DType::F32)?
        };
        let mut arguments = vec![("X", p(x)), ("Q", p(&q)), ("A", p(&left))];
        if fused {
            arguments.push(("Map", p(&row_map)));
        } else {
            arguments.push(("Rows", p(&selected_rows)));
        }
        arguments.extend([
            ("E", p(&projection)),
            ("M", n(a)?),
            ("N", n(rows)?),
            ("K", n(k)?),
            ("R", n(rank)?),
            ("INVERSE", Arg::F32(blocks as f32 / selected as f32)),
        ]);
        self.launch(
            if fused {
                "parameter_prepare"
            } else {
                "parameter_residual"
            },
            &arguments,
            json::json!({"BM":BM,"BN":BN,"BK":BK}),
            if fused {
                [a.div_ceil(BM), 1, 1]
            } else {
                [rank.div_ceil(BM) + k.div_ceil(BM), rows.div_ceil(BN), 1]
            },
        )?;
        Ok(PreparedVjp {
            q,
            left,
            projection,
            row_map,
            selected_rows,
            selected_columns: columns,
            basis_donor_width: basis.map_or(b, |basis| basis.donor_width),
        })
    }
    pub fn parameter_vjp(&mut self, prepared: &PreparedVjp, dy: &Tensor) -> Result<FactoredRows> {
        let rank = prepared.q.shape[1];
        let coefficients = self.matmul(&prepared.projection, dy, false, false, DType::F32)?;
        let right = coefficients.rows(0, rank)?;
        let tail = coefficients.rows(rank, coefficients.shape[0] - rank)?;
        Ok(FactoredRows {
            left: prepared.left.clone(),
            right,
            tail,
            row_map: prepared.row_map.clone(),
            scale: 1.,
        })
    }
}

fn push_pointer(values: &mut Vec<i32>, t: Option<&Tensor>) {
    let address = t.map_or(0, Tensor::pointer);
    values.push(address as u32 as i32);
    values.push((address >> 32) as u32 as i32);
}

pub struct GradientPlan {
    weights: Vec<Tensor>,
    plan: Tensor,
    owners: Tensor,
    tiles: usize,
}
pub struct PackedGradients {
    metadata: Tensor,
    term_offset: usize,
    rank_offset: usize,
    gradients: Vec<ParameterGradient>,
}
enum GradientAction<'a> {
    Materialize,
    Norm(&'a Tensor),
    Update { learning_rate: f32, decay: f32 },
}

/// Dense moment optimizer에 전달할 gradient buffer를 별도로 소유한다.
pub struct GradientMaterializer {
    plan: GradientPlan,
}
impl GradientMaterializer {
    pub fn new(e: &mut Engine, parameters: &[Tensor]) -> Result<Self> {
        let buffers = parameters
            .iter()
            .map(|p| e.empty(&p.shape, DType::F32))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            plan: GradientPlan::new(e, &buffers)?,
        })
    }
    pub fn materialize(
        &self,
        e: &mut Engine,
        gradients: &[ParameterGradient],
    ) -> Result<Vec<Option<Tensor>>> {
        let packed = self.plan.pack(e, gradients)?;
        self.plan.tiles(e, &packed, GradientAction::Materialize)?;
        Ok(gradients
            .iter()
            .zip(&self.plan.weights)
            .map(|(g, t)| {
                if g.dense.is_none() && g.terms.is_empty() {
                    None
                } else {
                    Some(t.clone())
                }
            })
            .collect())
    }
}

impl GradientPlan {
    pub fn new(e: &mut Engine, weights: &[Tensor]) -> Result<Self> {
        ensure(!weights.is_empty(), "gradient plan requires parameters")?;
        let mut plan = Vec::new();
        let mut owners = Vec::new();
        for (index, w) in weights.iter().enumerate() {
            ensure(
                w.dtype == DType::F32 && !w.shape.is_empty(),
                "compact SGD requires FP32 parameters",
            )?;
            let m = w.shape[0];
            let n = w.len() / m;
            push_pointer(&mut plan, Some(w));
            plan.extend([
                i32::try_from(m).map_err(|_| Error::Message("parameter rows overflow".into()))?,
                i32::try_from(n)
                    .map_err(|_| Error::Message("parameter columns overflow".into()))?,
                i32::try_from(owners.len())
                    .map_err(|_| Error::Message("gradient grid overflow".into()))?,
                i32::from(w.shape.len() > 1),
            ]);
            let count = m
                .div_ceil(BM)
                .checked_mul(n.div_ceil(BN))
                .ok_or_else(|| Error::Message("gradient grid overflow".into()))?;
            let length = owners
                .len()
                .checked_add(count)
                .filter(|&v| v <= i32::MAX as usize)
                .ok_or_else(|| Error::Message("gradient grid exceeds signed index range".into()))?;
            owners.resize(length, index as i32);
        }
        let tiles = owners.len();
        Ok(Self {
            weights: weights.to_vec(),
            plan: e.from_i32(&[plan.len()], &plan)?,
            owners: e.from_i32(&[tiles], &owners)?,
            tiles,
        })
    }
    pub fn pack(&self, e: &mut Engine, gradients: &[ParameterGradient]) -> Result<PackedGradients> {
        let mut headers = Vec::with_capacity(gradients.len() * 6);
        let mut terms = Vec::new();
        let mut rank_map = Vec::new();
        for g in gradients {
            push_pointer(&mut headers, g.dense.as_ref());
            let total_rank: usize = g.terms.iter().map(|t| t.left.shape[1]).sum();
            headers.extend(
                [terms.len() / 11, g.terms.len(), rank_map.len(), total_rank].map(|v| v as i32),
            );
            let mut rank_start = 0usize;
            for term in &g.terms {
                let term_index = (terms.len() / 11) as i32;
                rank_map.resize(rank_map.len() + term.left.shape[1], term_index);
                for t in [&term.left, &term.right, &term.tail, &term.row_map] {
                    push_pointer(&mut terms, Some(t));
                }
                terms.extend([
                    term.left.shape[1] as i32,
                    term.scale.to_bits() as i32,
                    rank_start as i32,
                ]);
                rank_start += term.left.shape[1];
            }
        }
        let term_offset = headers.len();
        headers.extend(terms);
        let rank_offset = headers.len();
        headers.extend(rank_map);
        Ok(PackedGradients {
            metadata: e.from_i32(&[headers.len()], &headers)?,
            term_offset,
            rank_offset,
            gradients: gradients.to_vec(),
        })
    }
    fn tiles(&self, e: &mut Engine, g: &PackedGradients, action: GradientAction<'_>) -> Result<()> {
        let mut args = vec![
            ("Plan", p(&self.plan)),
            ("Owners", p(&self.owners)),
            ("Meta", p(&g.metadata)),
            ("RANK_OFFSET", n(g.rank_offset)?),
        ];
        let constants = json::json!({"TERM_OFFSET":g.term_offset,"BM":BM,"BN":BN,"BK":BK});
        let operation = match action {
            GradientAction::Update {
                learning_rate,
                decay,
            } => {
                args.extend([("LR", Arg::F32(learning_rate)), ("DECAY", Arg::F32(decay))]);
                "parameter_gradient_update"
            }
            GradientAction::Materialize => "parameter_gradient_materialize",
            GradientAction::Norm(output) => {
                args.push(("Output", p(output)));
                "parameter_gradient_norm"
            }
        };
        e.launch(operation, &args, constants, [self.tiles, 1, 1])?;
        if e.metrics_enabled() {
            let mut has_gemm = false;
            for (w, gradient) in self.weights.iter().zip(&g.gradients) {
                let m = w.shape[0];
                let b = w.len() / m;
                let rank: usize = gradient.terms.iter().map(|t| t.left.shape[1]).sum();
                if rank > 0 {
                    e.counters.logical_flops += (2 * m as u128 * b as u128 * rank as u128) as u64;
                    e.counters.issued_flops += issued_gemm(m, b, rank) as u64;
                    has_gemm = true;
                }
            }
            e.counters.gemm_launches += u64::from(has_gemm);
        }
        Ok(())
    }
    pub fn norm(&self, e: &mut Engine, g: &PackedGradients) -> Result<f64> {
        let output = e.empty(&[self.tiles], DType::F64)?;
        self.tiles(e, g, GradientAction::Norm(&output))?;
        Ok(e.to_f64(&output)?.iter().sum::<f64>().sqrt())
    }
    pub fn update(
        &self,
        e: &mut Engine,
        g: &PackedGradients,
        learning_rate: f32,
        clipping: f32,
        weight_decay: f32,
    ) -> Result<()> {
        self.tiles(
            e,
            g,
            GradientAction::Update {
                learning_rate: learning_rate * clipping,
                decay: learning_rate * weight_decay,
            },
        )
    }
    pub fn materialize(
        e: &mut Engine,
        gradient: &ParameterGradient,
        shape: &[usize],
    ) -> Result<Tensor> {
        let value = e.empty(shape, DType::F32)?;
        let plan = Self::new(e, std::slice::from_ref(&value))?;
        let packed = plan.pack(e, std::slice::from_ref(gradient))?;
        plan.tiles(e, &packed, GradientAction::Materialize)?;
        Ok(value)
    }
}
