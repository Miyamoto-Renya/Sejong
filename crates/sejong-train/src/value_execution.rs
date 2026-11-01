//! NativeValue의 계층별 활성값 수명과 optimizer별 gradient 저장량으로 실행을 계획한다.
use super::{allocation, workspace_sum};
use sejong_core::{DType, ExpertConfig, ModelConfig, Result, ensure, native::ValueTraining};
use sejong_gpu::{Engine, parameter_vjp::budget};
use sejong_model::{Parameter, native::value::NativeValue};

pub(super) struct Plan {
    pub batch: usize,
    pub checkpointing: bool,
    pub workspace_bytes: usize,
    pub fixed_bytes: usize,
}

fn floats(elements: usize) -> Result<usize> {
    allocation(elements as u128, DType::F32)
}
fn pair(elements: usize, forward_dtype: DType) -> Result<usize> {
    // 순전파 값과 FP32 adjoint를 각각 allocator 단위로 계산한다.
    workspace_sum([
        (1, allocation(elements as u128, forward_dtype)?),
        (1, floats(elements)?),
    ])
}

struct Activations {
    precision: DType,
    retained: usize,
    replay: usize,
    largest: usize,
}
impl Activations {
    fn value(&mut self, elements: usize) -> Result<()> {
        self.value_as(elements, self.precision)
    }
    fn value_as(&mut self, elements: usize, forward_dtype: DType) -> Result<()> {
        let bytes = pair(elements, forward_dtype)?;
        self.retained = workspace_sum([(1, self.retained), (1, bytes)])?;
        self.largest = self.largest.max(floats(elements)?);
        Ok(())
    }
    fn stage(
        &mut self,
        rows: usize,
        width: usize,
        layers: usize,
        heads: usize,
        kv_heads: usize,
        ffn: usize,
        experts: Option<&ExpertConfig>,
        checkpoint: bool,
    ) -> Result<()> {
        let kv = width / heads * kv_heads;
        // norm/Q/RoPE/attention/projection/residual/norm/FFN-output/final-residual.
        let mut block = workspace_sum([
            (9, pair(rows * width, self.precision)?),
            (3, pair(rows * kv, self.precision)?),
            (1, pair(rows * heads, DType::F32)?), // LSE와 backward delta.
        ])?;
        let mut largest = floats(rows * width.max(experts.map_or(ffn, |e| e.ffn_width)))?;
        if let Some(expert) = experts {
            let routes = rows * expert.top_k;
            // Expert별 실제 행 분할의 power-of-two 할당 합은 2*총바이트 이하다.
            // 입력 gather, gate/up/activation, down을 순전파와 adjoint별로 집계한다.
            let expert_tensor = |columns: usize| -> Result<usize> {
                let elements = routes as u128 * columns as u128;
                // Expert별 forward/adjoint의 독립 할당 반올림 상한을 합한다.
                usize::try_from(
                    2 * elements * (self.precision.bytes() + DType::F32.bytes()) as u128,
                )
                .map_err(|_| sejong_core::Error::Message("expert activation size overflow".into()))
            };
            block = workspace_sum([
                (1, block),
                (1, pair(rows * expert.count, self.precision)?), // Router logits.
                (1, pair(routes, DType::F32)?),                  // Route weights.
                (1, allocation(routes as u128, DType::I32)?),    // Route ids.
                (2, allocation((expert.count * routes) as u128, DType::I32)?),
                (1, allocation(expert.count as u128, DType::I32)?),
                (2, expert_tensor(width)?), // Gather and expert output.
                (3, expert_tensor(expert.ffn_width)?),
            ])?;
            largest = largest.max(floats(rows * expert.ffn_width)?);
        } else {
            block = workspace_sum([(1, block), (3, pair(rows * ffn, self.precision)?)])?;
        }
        if checkpoint {
            // 각 block의 출력과 adjoint를 보존하고 재실행 중인 한 block을 더한다.
            self.retained = workspace_sum([
                (1, self.retained),
                (layers as u128, pair(rows * width, self.precision)?),
            ])?;
            self.replay = self.replay.max(block);
        } else {
            self.retained = workspace_sum([(1, self.retained), (layers as u128, block)])?;
        }
        self.largest = self.largest.max(largest);
        Ok(())
    }
}

pub(super) fn workspace_bytes(
    c: &ModelConfig,
    sequence: usize,
    values: usize,
    batch: usize,
    checkpointing: bool,
) -> Result<usize> {
    let rows = batch * sequence;
    let mut a = Activations {
        precision: c.precision,
        retained: 0,
        replay: 0,
        largest: 0,
    };
    a.value(rows * c.width)?; // Native byte embedding.
    let global_rows = if let Some(h) = &c.hierarchy {
        let globals = batch * sequence.div_ceil(h.patch_size);
        a.value(rows * h.encoder_width)?; // Input projection.
        a.stage(
            rows,
            h.encoder_width,
            h.encoder_layers,
            h.encoder_heads,
            h.encoder_heads,
            h.encoder_ffn_width,
            None,
            checkpointing && h.checkpoint_encoder,
        )?;
        a.value(rows * h.encoder_width)?; // Encoder norm.
        a.value(globals * h.encoder_width)?; // Patch pool.
        a.value(globals * c.width)?; // Encoder-to-global.
        globals
    } else {
        rows
    };
    a.stage(
        global_rows,
        c.width,
        c.layers,
        c.heads,
        c.kv_heads.unwrap_or(c.heads),
        c.ffn_width,
        c.experts.as_ref(),
        checkpointing,
    )?;
    if let Some(h) = &c.hierarchy {
        a.value(global_rows * c.width)?; // Global norm.
        a.value(global_rows * h.decoder_width)?; // Global-to-decoder.
        a.value(h.decoder_width)?; // Global BOS projection.
        a.value(rows * h.decoder_width)?; // Patch broadcast.
        a.value(rows * h.decoder_width)?; // Encoder-to-decoder.
        a.value(rows * h.decoder_width)?; // Decoder input add.
        a.stage(
            rows,
            h.decoder_width,
            h.decoder_layers,
            h.decoder_heads,
            h.decoder_heads,
            h.decoder_ffn_width,
            None,
            checkpointing && h.checkpoint_decoder,
        )?;
        a.value(rows * h.decoder_width)?; // Decoder norm.
        a.value(rows * c.width)?; // Output projection.
    }
    a.value(rows * c.width)?; // Final norm.
    a.value(batch * c.width)?; // Selected request states.
    a.value_as(batch * c.width, DType::F32)?; // FP32 selection cast.
    a.value_as(batch * values, DType::F32)?; // Value projection.
    a.value_as(batch * values, DType::F32)?; // Bias add.
    a.value_as(batch, DType::F32)?; // Cross-entropy losses.
    workspace_sum([
        (1, a.retained),
        (1, a.replay),
        // Incoming/summed FP32 adjoints, RMS reduction, and expert-combine FP32 scratch.
        (3, a.largest),
        (1, allocation(rows as u128, DType::I32)?),
        (2, allocation(batch as u128, DType::I32)?),
    ])
}

fn factor_bytes(
    parameters: &[Parameter],
    spec: &ValueTraining,
    batch: usize,
    sequence: usize,
) -> Result<usize> {
    let Some(policy) = &spec.parameter_vjp else {
        return Ok(0);
    };
    let mut groups = std::collections::BTreeMap::<(&str, bool), Vec<&Parameter>>::new();
    for p in parameters {
        if let Some((prefix, suffix)) = p.name.rsplit_once('.') {
            match suffix {
                "query" | "key" | "value" => groups.entry((prefix, false)).or_default().push(p),
                "gate" | "up" => groups.entry((prefix, true)).or_default().push(p),
                _ => {}
            }
        }
    }
    let global_rows = batch
        * spec
            .model
            .hierarchy
            .as_ref()
            .map_or(sequence, |h| sequence.div_ceil(h.patch_size));
    let mut retained = 0;
    let mut scratch = 0;
    for ((prefix, _), members) in groups {
        let rows = if prefix.starts_with("encoder.") || prefix.starts_with("decoder.") {
            batch * sequence
        } else {
            // 한 expert가 받는 행 수의 상한은 global 행 수다.
            global_rows
        };
        let a = members[0].value.shape[0];
        if rows < a || members.len() <= 1 {
            continue;
        }
        let donor = members.iter().map(|p| p.value.shape[1]).max().unwrap();
        let (rank, blocks, selected) = budget(policy, rows, a, donor);
        if selected == blocks {
            continue;
        }
        let k = selected * policy.block_rows;
        // QKV 및 gate/up은 입력별 PreparedVjp를 공유한다. Graph가 projection도 보존한다.
        retained = workspace_sum([
            (1, retained),
            (1, floats(a * rank)?),
            (1, floats((rank + k) * rows)?),
            (1, floats(rows * rank)?),
            (1, allocation(a as u128, DType::I32)?),
            (1, allocation(k as u128, DType::I32)?),
            (1, allocation(rank as u128, DType::I32)?),
        ])?;
        for p in members {
            let b = p.value.shape[1];
            // right와 tail은 합쳐 할당한 coefficient 행렬의 view다.
            retained = workspace_sum([(1, retained), (1, floats((rank + k) * b)?)])?;
        }
        scratch = scratch.max(allocation((donor + blocks) as u128, DType::I32)?);
    }
    workspace_sum([(1, retained), (1, scratch)])
}

pub(super) fn prepare(
    e: &Engine,
    model: &NativeValue,
    parameters: &[Parameter],
    spec: &ValueTraining,
    saved: Option<(usize, bool)>,
) -> Result<Plan> {
    let sequence = spec.prompts.iter().map(|p| p.len() + 2).max().unwrap();
    let mut fixed = 0;
    let mut largest = 0;
    for p in parameters {
        let bytes = floats(p.value.len())?;
        fixed = workspace_sum([(1, fixed), (1, bytes)])?;
        largest = largest.max(bytes);
    }
    // Master/cast adjoints share storage. Reused weights can add incoming+summed temporaries.
    fixed = workspace_sum([(1, fixed), (2, largest)])?;
    if model.body.config.precision != DType::F32 {
        for p in &model.body.parameters {
            fixed = workspace_sum([
                (1, fixed),
                (
                    1,
                    allocation(p.value.len() as u128, model.body.config.precision)?,
                ),
            ])?;
        }
        fixed = workspace_sum([
            (1, fixed),
            (
                1,
                allocation(model.embedding.len() as u128, model.body.config.precision)?,
            ),
        ])?;
    }
    let available = e.available_memory_bytes()?;
    // BatchPolicy의 memory_bytes는 gradient/cast 예약 뒤 활성값·인수 작업 공간을 제한한다.
    let available = spec
        .batch
        .memory_bytes
        .map_or(available, |b| fixed.saturating_add(b).min(available));
    let cap = spec
        .batch
        .max_examples
        .unwrap_or(spec.requests.len())
        .min(spec.requests.len());
    let required = |batch, checkpoint| -> Result<usize> {
        workspace_sum([
            (1, fixed),
            (
                1,
                workspace_bytes(&spec.model, sequence, spec.values.len(), batch, checkpoint)?,
            ),
            (1, factor_bytes(parameters, spec, batch, sequence)?),
        ])
    };
    if let Some((batch, checkpoint)) = saved {
        ensure(
            batch > 0 && batch <= cap,
            "saved native batch exceeds requested batch limit",
        )?;
        let bytes = required(batch, checkpoint)?;
        if bytes > available {
            return Err(sejong_core::Error::Capacity {
                operation: "native resumed batch".into(),
                required: bytes,
                available,
            });
        }
        return Ok(Plan {
            batch,
            checkpointing: checkpoint,
            workspace_bytes: bytes - fixed,
            fixed_bytes: fixed,
        });
    }
    let modes = spec
        .activation_checkpointing
        .map_or(vec![false, true], |x| vec![x]);
    for checkpoint in modes {
        if cap == 0 || required(1, checkpoint)? > available {
            continue;
        }
        let (mut low, mut high) = (1, cap);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if required(middle, checkpoint)? <= available {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let bytes = required(low, checkpoint)?;
        return Ok(Plan {
            batch: low,
            checkpointing: checkpoint,
            workspace_bytes: bytes - fixed,
            fixed_bytes: fixed,
        });
    }
    let checkpoint = spec.activation_checkpointing.unwrap_or(true);
    Err(sejong_core::Error::Capacity {
        operation: "native training workspace".into(),
        required: required(1, checkpoint)?,
        available,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_replay_keeps_one_block_and_hierarchy_shortens_global_sequence() {
        let mut c = ModelConfig::default();
        c.layers = 8;
        let retained = workspace_bytes(&c, 1024, 16, 1, false).unwrap();
        let replay = workspace_bytes(&c, 1024, 16, 1, true).unwrap();
        assert!(replay < retained);
        let bigger = workspace_bytes(&c, 1024, 16, 2, false).unwrap();
        assert!(bigger >= retained);
        c.hierarchy = Some(sejong_core::HierarchyConfig {
            patch_size: 8,
            encoder_width: 128,
            encoder_layers: 2,
            encoder_heads: 2,
            encoder_ffn_width: 384,
            decoder_width: 128,
            decoder_layers: 2,
            decoder_heads: 2,
            decoder_ffn_width: 384,
            local_window: 128,
            checkpoint_encoder: false,
            checkpoint_decoder: false,
        });
        let patched = workspace_bytes(&c, 1024, 16, 1, false).unwrap();
        c.hierarchy.as_mut().unwrap().patch_size = 1;
        assert!(workspace_bytes(&c, 1024, 16, 1, false).unwrap() > patched);
    }
}
