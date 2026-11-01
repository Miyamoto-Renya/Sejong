use super::{Binding, Cache, LowRankCorrection, Model};
use sejong_core::{DType, ExpertConfig, ModelConfig, Result, VOCAB, ensure};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use std::collections::HashMap;

#[derive(Clone)]
struct Stage {
    prefix: String,
    width: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    ffn: usize,
    window: Option<usize>,
    experts: Option<ExpertConfig>,
    checkpoint: bool,
}

fn global(c: &ModelConfig) -> Stage {
    Stage {
        prefix: "layers".into(),
        width: c.width,
        layers: c.layers,
        heads: c.heads,
        kv_heads: c.kv_heads.unwrap_or(c.heads),
        ffn: c.ffn_width,
        window: None,
        experts: c.experts.clone(),
        checkpoint: true,
    }
}
fn local(c: &ModelConfig, encoder: bool) -> Stage {
    let h = c.hierarchy.as_ref().unwrap();
    if encoder {
        Stage {
            prefix: "encoder".into(),
            width: h.encoder_width,
            layers: h.encoder_layers,
            heads: h.encoder_heads,
            kv_heads: h.encoder_heads,
            ffn: h.encoder_ffn_width,
            window: Some(h.local_window),
            experts: None,
            checkpoint: h.checkpoint_encoder,
        }
    } else {
        Stage {
            prefix: "decoder".into(),
            width: h.decoder_width,
            layers: h.decoder_layers,
            heads: h.decoder_heads,
            kv_heads: h.decoder_heads,
            ffn: h.decoder_ffn_width,
            window: Some(h.local_window),
            experts: None,
            checkpoint: h.checkpoint_decoder,
        }
    }
}
fn stage_layout(s: &Stage, out: &mut Vec<(String, Vec<usize>, bool)>) {
    let d = s.width;
    let kv = d / s.heads * s.kv_heads;
    for i in 0..s.layers {
        let prefix = format!("{}.{i}", s.prefix);
        for name in ["attention_norm", "ffn_norm"] {
            out.push((format!("{prefix}.{name}"), vec![d], true));
        }
        for (name, width) in [
            ("query", d),
            ("key", kv),
            ("value", kv),
            ("attention_output", d),
        ] {
            out.push((format!("{prefix}.{name}"), vec![d, width], false));
        }
        let mut ffn = |prefix: String, f: usize| {
            for name in ["gate", "up"] {
                out.push((format!("{prefix}.{name}"), vec![d, f], false));
            }
            out.push((format!("{prefix}.down"), vec![f, d], false));
        };
        if let Some(experts) = &s.experts {
            for e in 0..experts.count {
                ffn(format!("{prefix}.experts.{e}"), experts.ffn_width);
            }
            out.push((format!("{prefix}.router"), vec![d, experts.count], false));
        } else {
            ffn(prefix, s.ffn);
        }
    }
}
pub(super) fn layout(c: &ModelConfig) -> Vec<(String, Vec<usize>, bool)> {
    let embedding_width = c.hierarchy.as_ref().map_or(c.width, |h| h.encoder_width);
    let mut out = vec![("embedding".into(), vec![VOCAB, embedding_width], false)];
    if let Some(h) = &c.hierarchy {
        stage_layout(&local(c, true), &mut out);
        out.push(("encoder_norm".into(), vec![h.encoder_width], true));
        out.push((
            "input_projection".into(),
            vec![c.width, h.encoder_width],
            false,
        ));
        out.push((
            "encoder_to_global".into(),
            vec![h.encoder_width, c.width],
            false,
        ));
        out.push((
            "encoder_to_decoder".into(),
            vec![h.encoder_width, h.decoder_width],
            false,
        ));
        out.push((
            "global_to_decoder".into(),
            vec![c.width, h.decoder_width],
            false,
        ));
        out.push(("global_bos".into(), vec![1, c.width], false));
        out.push(("global_norm".into(), vec![c.width], true));
        stage_layout(&global(c), &mut out);
        stage_layout(&local(c, false), &mut out);
        out.push(("decoder_norm".into(), vec![h.decoder_width], true));
        out.push((
            "output_projection".into(),
            vec![h.decoder_width, c.width],
            false,
        ));
    } else {
        stage_layout(&global(c), &mut out);
    }
    out.push(("final_norm".into(), vec![c.width], true));
    out.push(("output".into(), vec![c.width, VOCAB], false));
    out
}

#[derive(Clone)]
struct Linear {
    slot: usize,
    correction: Option<LowRankCorrection>,
}
impl Linear {
    fn apply(&self, g: &mut Graph, inputs: &[Var], x: Var) -> Result<Var> {
        match &self.correction {
            Some(c) => g.low_rank_matmul(x, inputs[self.slot], &c.left, &c.coefficient, &c.right),
            None => g.matmul(x, inputs[self.slot]),
        }
    }
}
fn linear(m: &Model, g: &mut Graph, b: &Binding, x: Var, name: &str) -> Result<Var> {
    let i = m.index[name];
    Linear {
        slot: 0,
        correction: b.corrections[i].clone(),
    }
    .apply(g, &[b.compute[i]], x)
}
pub(super) fn output(m: &Model, g: &mut Graph, b: &Binding, x: Var) -> Result<Var> {
    let i = m.index["output"];
    Linear {
        slot: 0,
        correction: b.corrections[i].clone(),
    }
    .apply(g, &[b.leaves[i]], x)
}
#[derive(Clone)]
struct Ffn {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Ffn {
    fn apply(&self, g: &mut Graph, v: &[Var], x: Var) -> Result<Var> {
        let gate = self.gate.apply(g, v, x)?;
        let up = self.up.apply(g, v, x)?;
        let active = g.swiglu(gate, up)?;
        self.down.apply(g, v, active)
    }
}
struct Block {
    attention_norm: usize,
    ffn_norm: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    ffn: Vec<Ffn>,
    router: Option<Linear>,
    stage: Stage,
    sequence: usize,
    theta: f32,
    epsilon: f32,
}
impl Block {
    fn apply(&self, g: &mut Graph, v: &[Var]) -> Result<Var> {
        let z = g.norm(v[0], v[self.attention_norm], self.epsilon)?;
        let q = self.query.apply(g, v, z)?;
        let k = self.key.apply(g, v, z)?;
        let value = self.value.apply(g, v, z)?;
        let q = g.rope(q, self.sequence, 0, self.stage.heads, self.theta)?;
        let k = g.rope(k, self.sequence, 0, self.stage.kv_heads, self.theta)?;
        let a = g.attention_grouped(
            q,
            k,
            value,
            self.sequence,
            self.stage.heads,
            self.stage.kv_heads,
            self.stage.window,
        )?;
        let a = self.attention_output.apply(g, v, a)?;
        let x = g.add(v[0], a)?;
        let z = g.norm(x, v[self.ffn_norm], self.epsilon)?;
        let update = if let Some(router) = &self.router {
            let logits = router.apply(g, v, z)?;
            let expert = self.stage.experts.as_ref().unwrap();
            let (weights, ids) = g.route_topk(logits, expert.top_k)?;
            let routes = g.engine.expert_routes(&ids, expert.count)?;
            let mut outputs = Vec::new();
            for (index, route) in routes.into_iter().enumerate() {
                if let Some((tokens, slots)) = route {
                    let input = g.gather(z, &tokens)?;
                    outputs.push((self.ffn[index].apply(g, v, input)?, slots));
                }
            }
            g.expert_combine(&outputs, weights)?
        } else {
            self.ffn[0].apply(g, v, z)?
        };
        g.add(x, update)
    }
}
fn block(
    m: &Model,
    g: &mut Graph,
    b: &Binding,
    x: Var,
    sequence: usize,
    stage: &Stage,
    layer: usize,
) -> Result<Var> {
    let prefix = format!("{}.{layer}", stage.prefix);
    let mut inputs = vec![x];
    let mut add = |name: &str| {
        let i = m.index[&format!("{prefix}.{name}")];
        let slot = inputs.len();
        inputs.push(b.compute[i]);
        Linear {
            slot,
            correction: b.corrections[i].clone(),
        }
    };
    let attention_norm = add("attention_norm").slot;
    let ffn_norm = add("ffn_norm").slot;
    let query = add("query");
    let key = add("key");
    let value = add("value");
    let attention_output = add("attention_output");
    let mut ffn = Vec::new();
    let router;
    if let Some(e) = &stage.experts {
        router = Some(add("router"));
        for index in 0..e.count {
            ffn.push(Ffn {
                gate: add(&format!("experts.{index}.gate")),
                up: add(&format!("experts.{index}.up")),
                down: add(&format!("experts.{index}.down")),
            });
        }
    } else {
        router = None;
        ffn.push(Ffn {
            gate: add("gate"),
            up: add("up"),
            down: add("down"),
        });
    }
    let op = Block {
        attention_norm,
        ffn_norm,
        query,
        key,
        value,
        attention_output,
        ffn,
        router,
        stage: stage.clone(),
        sequence,
        theta: m.config.rope_theta,
        epsilon: m.config.norm_epsilon,
    };
    if stage.checkpoint {
        g.checkpoint(&inputs, move |g, v| op.apply(g, v))
    } else {
        op.apply(g, &inputs)
    }
}
fn blocks(
    m: &Model,
    g: &mut Graph,
    b: &Binding,
    mut x: Var,
    sequence: usize,
    stage: Stage,
) -> Result<Var> {
    for i in 0..stage.layers {
        x = block(m, g, b, x, sequence, &stage, i)?;
    }
    Ok(x)
}
fn encoder(m: &Model, g: &mut Graph, b: &Binding, x: Var, sequence: usize) -> Result<Var> {
    let x = blocks(m, g, b, x, sequence, local(&m.config, true))?;
    g.norm(x, m.w(b, "encoder_norm"), m.config.norm_epsilon)
}
pub(super) fn encode(
    m: &Model,
    g: &mut Graph,
    b: &Binding,
    ids: &Tensor,
    sequence: usize,
    injection: Option<Var>,
) -> Result<Var> {
    let mut x = g.embedding(m.w(b, "embedding"), ids, m.config.precision)?;
    if let Some(input) = injection {
        let input = if m.config.hierarchy.is_some() {
            linear(m, g, b, input, "input_projection")?
        } else {
            input
        };
        x = g.add(x, input)?;
    }
    if m.config.hierarchy.is_some() {
        encoder(m, g, b, x, sequence)
    } else {
        Ok(x)
    }
}
pub(super) fn hidden_from(
    m: &Model,
    g: &mut Graph,
    b: &Binding,
    x: Var,
    sequence: usize,
) -> Result<Var> {
    let x = if m.config.hierarchy.is_some() {
        let x = linear(m, g, b, x, "input_projection")?;
        encoder(m, g, b, x, sequence)?
    } else {
        x
    };
    hidden_from_encoded(m, g, b, x, sequence)
}
pub(super) fn hidden_from_encoded(
    m: &Model,
    g: &mut Graph,
    b: &Binding,
    x: Var,
    sequence: usize,
) -> Result<Var> {
    let c = &m.config;
    let x = if let Some(h) = &c.hierarchy {
        let pooled = g.patch_pool(x, sequence, h.patch_size)?;
        let z = linear(m, g, b, pooled, "encoder_to_global")?;
        let z = blocks(m, g, b, z, sequence.div_ceil(h.patch_size), global(c))?;
        let z = g.norm(z, m.w(b, "global_norm"), c.norm_epsilon)?;
        let z = linear(m, g, b, z, "global_to_decoder")?;
        let bos = linear(m, g, b, m.w(b, "global_bos"), "global_to_decoder")?;
        let z = g.patch_broadcast(z, bos, sequence, h.patch_size)?;
        let bytes = linear(m, g, b, x, "encoder_to_decoder")?;
        let x = g.add(bytes, z)?;
        let x = blocks(m, g, b, x, sequence, local(c, false))?;
        let x = g.norm(x, m.w(b, "decoder_norm"), c.norm_epsilon)?;
        linear(m, g, b, x, "output_projection")?
    } else {
        blocks(m, g, b, x, sequence, global(c))?
    };
    g.norm(x, m.w(b, "final_norm"), c.norm_epsilon)
}

struct StageCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}
pub(super) struct HierarchyCache {
    encoder: StageCache,
    decoder: StageCache,
    patch_sum: Option<Tensor>,
    patch_count: usize,
    global_context: Option<Tensor>,
}
impl HierarchyCache {
    pub(super) fn reset(&mut self) {
        self.patch_sum = None;
        self.patch_count = 0;
        self.global_context = None;
    }
}
fn allocate_cache(
    engine: &mut Engine,
    stage: &Stage,
    capacity: usize,
    dtype: DType,
) -> Result<StageCache> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let width = stage.width / stage.heads * stage.kv_heads;
    for _ in 0..stage.layers {
        keys.push(engine.empty(&[capacity, width], dtype)?);
        values.push(engine.empty(&[capacity, width], dtype)?);
    }
    Ok(StageCache { keys, values })
}
pub(super) fn new_cache(m: &Model, engine: &mut Engine) -> Result<Cache> {
    let c = &m.config;
    let capacity = c
        .hierarchy
        .as_ref()
        .map_or(c.context, |h| c.context.div_ceil(h.patch_size));
    let global = allocate_cache(engine, &global(c), capacity, c.precision)?;
    let hierarchy = if c.hierarchy.is_some() {
        Some(HierarchyCache {
            encoder: allocate_cache(engine, &local(c, true), c.context, c.precision)?,
            decoder: allocate_cache(engine, &local(c, false), c.context, c.precision)?,
            patch_sum: None,
            patch_count: 0,
            global_context: None,
        })
    } else {
        None
    };
    Ok(Cache {
        keys: global.keys,
        values: global.values,
        length: 0,
        capacity: c.context,
        hierarchy,
    })
}
fn cached_ffn(
    engine: &mut Engine,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    let gate = engine.matmul(x, &weights[&format!("{prefix}.gate")], false, false, dtype)?;
    let up = engine.matmul(x, &weights[&format!("{prefix}.up")], false, false, dtype)?;
    let active = engine.swiglu(&gate, &up)?;
    engine.matmul(
        &active,
        &weights[&format!("{prefix}.down")],
        false,
        false,
        dtype,
    )
}
fn cached_blocks(
    engine: &mut Engine,
    weights: &HashMap<String, Tensor>,
    mut x: Tensor,
    stage: &Stage,
    keys: &[Tensor],
    values: &[Tensor],
    offset: usize,
    c: &ModelConfig,
) -> Result<Tensor> {
    let rows = x.shape[0];
    for layer in 0..stage.layers {
        let prefix = format!("{}.{layer}", stage.prefix);
        let w = |name: &str| &weights[&format!("{prefix}.{name}")];
        let z = engine.norm(&x, w("attention_norm"), c.norm_epsilon)?;
        let q = engine.matmul(&z, w("query"), false, false, c.precision)?;
        let k = engine.matmul(&z, w("key"), false, false, c.precision)?;
        let v = engine.matmul(&z, w("value"), false, false, c.precision)?;
        let q = engine.rope(&q, rows, offset, stage.heads, c.rope_theta, false)?;
        let k = engine.rope(&k, rows, offset, stage.kv_heads, c.rope_theta, false)?;
        engine.copy_into(&k, &keys[layer].rows(offset, rows)?)?;
        engine.copy_into(&v, &values[layer].rows(offset, rows)?)?;
        let (a, _) = engine.attention_grouped(
            &q,
            &keys[layer].rows(0, offset + rows)?,
            &values[layer].rows(0, offset + rows)?,
            rows,
            offset + rows,
            offset,
            stage.heads,
            stage.kv_heads,
            stage.window,
        )?;
        let a = engine.matmul(&a, w("attention_output"), false, false, c.precision)?;
        x = engine.add(&x, &a, 1., c.precision)?;
        let z = engine.norm(&x, w("ffn_norm"), c.norm_epsilon)?;
        let update = if let Some(expert) = &stage.experts {
            let logits = engine.matmul(&z, w("router"), false, false, c.precision)?;
            let (gates, ids) = engine.route_topk(&logits, expert.top_k)?;
            let routes = engine.expert_routes(&ids, expert.count)?;
            let mut outputs = Vec::new();
            for (index, route) in routes.into_iter().enumerate() {
                if let Some((tokens, slots)) = route {
                    let input = engine.gather(&z, &tokens)?;
                    let output = cached_ffn(
                        engine,
                        weights,
                        &format!("{prefix}.experts.{index}"),
                        &input,
                        c.precision,
                    )?;
                    outputs.push((output, slots));
                }
            }
            engine.expert_combine(&outputs, &gates)?
        } else {
            cached_ffn(engine, weights, &prefix, &z, c.precision)?
        };
        x = engine.add(&x, &update, 1., c.precision)?;
    }
    Ok(x)
}
fn hierarchical_prefill(
    m: &Model,
    engine: &mut Engine,
    weights: &HashMap<String, Tensor>,
    tokens: &[i32],
    cache: &mut Cache,
    injection: Option<&Tensor>,
) -> Result<Tensor> {
    let c = &m.config;
    let h = c.hierarchy.as_ref().unwrap();
    let state = cache.hierarchy.as_mut().unwrap();
    let ids = engine.from_i32(&[tokens.len()], tokens)?;
    let mut encoded = engine.embedding(&weights["embedding"], &ids, c.precision)?;
    if let Some(input) = injection {
        let projected = engine.matmul(
            input,
            &weights["input_projection"],
            false,
            false,
            c.precision,
        )?;
        encoded = engine.add(&encoded, &projected, 1., c.precision)?;
    }
    let encoded = cached_blocks(
        engine,
        weights,
        encoded,
        &local(c, true),
        &state.encoder.keys,
        &state.encoder.values,
        0,
        c,
    )?;
    let encoded = engine.norm(&encoded, &weights["encoder_norm"], c.norm_epsilon)?;
    let pooled = engine.patch_pool(&encoded, tokens.len(), h.patch_size)?;
    let latent = engine.matmul(
        &pooled,
        &weights["encoder_to_global"],
        false,
        false,
        c.precision,
    )?;
    let latent = cached_blocks(
        engine,
        weights,
        latent,
        &global(c),
        &cache.keys,
        &cache.values,
        0,
        c,
    )?;
    let latent = engine.norm(&latent, &weights["global_norm"], c.norm_epsilon)?;
    let contexts = engine.matmul(
        &latent,
        &weights["global_to_decoder"],
        false,
        false,
        c.precision,
    )?;
    let bos = engine.matmul(
        &weights["global_bos"],
        &weights["global_to_decoder"],
        false,
        false,
        c.precision,
    )?;
    let broadcast = engine.patch_broadcast(&contexts, &bos, tokens.len(), h.patch_size)?;
    let bytes = engine.matmul(
        &encoded,
        &weights["encoder_to_decoder"],
        false,
        false,
        c.precision,
    )?;
    let decoded = engine.add(&bytes, &broadcast, 1., c.precision)?;
    let decoded = cached_blocks(
        engine,
        weights,
        decoded,
        &local(c, false),
        &state.decoder.keys,
        &state.decoder.values,
        0,
        c,
    )?;
    let decoded = engine.norm(&decoded, &weights["decoder_norm"], c.norm_epsilon)?;
    let decoded = engine.matmul(
        &decoded,
        &weights["output_projection"],
        false,
        false,
        c.precision,
    )?;
    let output = engine.norm(&decoded, &weights["final_norm"], c.norm_epsilon)?;
    let completed = tokens.len() / h.patch_size;
    state.global_context = Some(if completed == 0 {
        bos
    } else {
        contexts.rows(completed - 1, 1)?
    });
    state.patch_count = tokens.len() % h.patch_size;
    state.patch_sum = if state.patch_count == 0 {
        None
    } else {
        let tail = engine.cast(
            &encoded.rows(tokens.len() - state.patch_count, state.patch_count)?,
            DType::F32,
        )?;
        let mean = engine.row_mean(&tail)?;
        Some(engine.scale(&mean, state.patch_count as f32)?)
    };
    // 다음 완성 patch는 같은 slot에 K/V를 기록하여 마지막 partial slot을 갱신한다.
    cache.length = tokens.len();
    Ok(output)
}

pub(super) fn hidden_cached(
    m: &Model,
    engine: &mut Engine,
    weights: &HashMap<String, Tensor>,
    tokens: &[i32],
    cache: &mut Cache,
    injection: Option<&Tensor>,
) -> Result<Tensor> {
    let c = &m.config;
    if let Some(input) = injection {
        ensure(
            input.shape == [tokens.len(), c.width],
            "cached injection shape differs from public hidden width",
        )?;
    }
    if let Some(h) = &c.hierarchy {
        if cache.length == 0 && tokens.len() >= h.patch_size {
            return hierarchical_prefill(m, engine, weights, tokens, cache, injection);
        }
        let state = cache.hierarchy.as_mut().unwrap();
        let mut outputs = Vec::with_capacity(tokens.len().div_ceil(h.patch_size) + 1);
        let mut i = 0;
        while i < tokens.len() {
            let count = (h.patch_size - state.patch_count).min(tokens.len() - i);
            let position = cache.length + i;
            let ids = engine.from_i32(&[count], &tokens[i..i + count])?;
            let mut encoded = engine.embedding(&weights["embedding"], &ids, c.precision)?;
            if let Some(input) = injection {
                let projected = engine.matmul(
                    &input.rows(i, count)?,
                    &weights["input_projection"],
                    false,
                    false,
                    c.precision,
                )?;
                encoded = engine.add(&encoded, &projected, 1., c.precision)?;
            }
            let encoded = cached_blocks(
                engine,
                weights,
                encoded,
                &local(c, true),
                &state.encoder.keys,
                &state.encoder.values,
                position,
                c,
            )?;
            let encoded = engine.norm(&encoded, &weights["encoder_norm"], c.norm_epsilon)?;
            if state.global_context.is_none() {
                state.global_context = Some(engine.matmul(
                    &weights["global_bos"],
                    &weights["global_to_decoder"],
                    false,
                    false,
                    c.precision,
                )?);
            }
            let bytes = engine.matmul(
                &encoded,
                &weights["encoder_to_decoder"],
                false,
                false,
                c.precision,
            )?;
            let decoded = engine.bias_add(&bytes, state.global_context.as_ref().unwrap())?;
            let decoded = cached_blocks(
                engine,
                weights,
                decoded,
                &local(c, false),
                &state.decoder.keys,
                &state.decoder.values,
                position,
                c,
            )?;
            let decoded = engine.norm(&decoded, &weights["decoder_norm"], c.norm_epsilon)?;
            let decoded = engine.matmul(
                &decoded,
                &weights["output_projection"],
                false,
                false,
                c.precision,
            )?;
            outputs.push(engine.norm(&decoded, &weights["final_norm"], c.norm_epsilon)?);
            let encoded = engine.cast(&encoded, DType::F32)?;
            let average = engine.row_mean(&encoded)?;
            let encoded = engine.scale(&average, count as f32)?;
            state.patch_sum = Some(match state.patch_sum.take() {
                Some(sum) => engine.add(&sum, &encoded, 1., DType::F32)?,
                None => encoded,
            });
            state.patch_count += count;
            if state.patch_count == h.patch_size {
                let pooled =
                    engine.scale(state.patch_sum.as_ref().unwrap(), 1. / h.patch_size as f32)?;
                let pooled = engine.cast(&pooled, c.precision)?;
                let latent = engine.matmul(
                    &pooled,
                    &weights["encoder_to_global"],
                    false,
                    false,
                    c.precision,
                )?;
                let latent = cached_blocks(
                    engine,
                    weights,
                    latent,
                    &global(c),
                    &cache.keys,
                    &cache.values,
                    position / h.patch_size,
                    c,
                )?;
                let latent = engine.norm(&latent, &weights["global_norm"], c.norm_epsilon)?;
                state.global_context = Some(engine.matmul(
                    &latent,
                    &weights["global_to_decoder"],
                    false,
                    false,
                    c.precision,
                )?);
                state.patch_sum = None;
                state.patch_count = 0;
            }
            i += count;
        }
        cache.length += tokens.len();
        engine.concat_rows(&outputs)
    } else {
        let ids = engine.from_i32(&[tokens.len()], tokens)?;
        let mut x = engine.embedding(&weights["embedding"], &ids, c.precision)?;
        if let Some(input) = injection {
            x = engine.add(&x, input, 1., c.precision)?;
        }
        let x = cached_blocks(
            engine,
            weights,
            x,
            &global(c),
            &cache.keys,
            &cache.values,
            cache.length,
            c,
        )?;
        let h = engine.norm(&x, &weights["final_norm"], c.norm_epsilon)?;
        cache.length += tokens.len();
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sejong_core::HierarchyConfig;

    #[test]
    fn grouped_parameter_budget_matches_named_layout() {
        let dense = ModelConfig::default();
        let mut gqa = dense.clone();
        gqa.kv_heads = Some(2);
        let mut hierarchy = gqa.clone();
        hierarchy.hierarchy = Some(HierarchyConfig {
            patch_size: 7,
            encoder_width: 48,
            encoder_layers: 2,
            encoder_heads: 3,
            encoder_ffn_width: 112,
            decoder_width: 80,
            decoder_layers: 3,
            decoder_heads: 5,
            decoder_ffn_width: 176,
            local_window: 23,
            checkpoint_encoder: false,
            checkpoint_decoder: true,
        });
        let mut sparse = hierarchy.clone();
        sparse.experts = Some(ExpertConfig {
            count: 5,
            top_k: 2,
            ffn_width: 96,
        });
        for config in [dense, gqa, hierarchy, sparse] {
            config.validate().unwrap();
            let expected = layout(&config)
                .iter()
                .map(|(_, shape, _)| {
                    Engine::allocation_bytes(shape.iter().product::<usize>() * 4).unwrap()
                })
                .sum::<usize>();
            let grouped = config
                .parameter_tensor_groups()
                .unwrap()
                .into_iter()
                .map(|(elements, count)| Engine::allocation_bytes(elements * 4).unwrap() * count)
                .sum::<usize>();
            assert_eq!(expected, grouped);
        }
    }

    #[test]
    fn dense_metadata_preserves_legacy_serialized_fields() {
        let config = ModelConfig::default();
        let serialized = sejong_core::json::to_value(&config).unwrap();
        assert!(serialized.get("hierarchy").is_none());
        assert!(serialized.get("kv_heads").is_none());
        assert!(serialized.get("experts").is_none());
        let restored: ModelConfig = sejong_core::json::from_value(serialized.clone()).unwrap();
        assert_eq!(sejong_core::json::to_value(restored).unwrap(), serialized);
    }

    #[test]
    fn allocation_dimensions_are_checked_before_layout() {
        let mut config = ModelConfig::default();
        config.width = usize::MAX - 1;
        config.heads = 1;
        assert!(config.validate().is_err());
        config.width = 2;
        config.ffn_width = 2;
        config.layers = usize::MAX;
        assert!(config.validate().is_err());
    }
}
