//! Cached autoregressive generation and byte-safe text output.
pub mod native;
use sejong_core::{EOS, Result, Rng, ensure};
use sejong_gpu::Engine;
use sejong_model::Model;
use std::io::Write;

#[derive(Clone, Debug)]
pub struct Sampling {
    pub max_bytes: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            max_bytes: 512,
            temperature: 0.8,
            top_k: 0,
            top_p: 0.95,
            seed: 42,
        }
    }
}
pub fn sample(logits: &[f32], options: &Sampling, rng: &mut Rng) -> Result<i32> {
    ensure(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "sampling requires finite logits",
    )?;
    ensure(
        options.temperature.is_finite()
            && options.temperature >= 0.
            && options.top_p > 0.
            && options.top_p <= 1.,
        "sampling requires temperature >= 0 and top_p in (0, 1]",
    )?;
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    if options.temperature == 0. {
        return Ok(ranked[0].0 as i32);
    }
    if options.top_k > 0 {
        ranked.truncate(options.top_k.min(ranked.len()));
    }
    let maximum = ranked[0].1;
    for (_, value) in &mut ranked {
        *value = ((*value - maximum) / options.temperature).exp();
    }
    let total: f32 = ranked.iter().map(|(_, v)| v).sum();
    let mut cumulative = 0.;
    let mut keep = ranked.len();
    for (i, (_, value)) in ranked.iter().enumerate() {
        cumulative += value / total;
        if cumulative >= options.top_p {
            keep = i + 1;
            break;
        }
    }
    ranked.truncate(keep);
    let total: f32 = ranked.iter().map(|(_, v)| v).sum();
    let threshold = rng.uniform() * total;
    let mut acc = 0.;
    for (index, value) in &ranked {
        acc += value;
        if acc >= threshold {
            return Ok(*index as i32);
        }
    }
    Ok(ranked.last().unwrap().0 as i32)
}
pub struct Generation {
    pub bytes: Vec<u8>,
    pub ended: bool,
}
pub fn generate(
    engine: &mut Engine,
    model: &Model,
    prompt: &[u8],
    options: &Sampling,
    mut emit: impl FnMut(u8) -> Result<()>,
) -> Result<Generation> {
    let weights = model.inference_weights(engine)?;
    let mut cache = model.new_cache(engine)?;
    let mut tokens = vec![EOS];
    tokens.extend(prompt.iter().map(|b| *b as i32));
    if tokens.len() > model.config.context {
        tokens.drain(..tokens.len() - model.config.context);
    }
    let mut logits = model.forward_cached(engine, &weights, &tokens, &mut cache)?;
    let mut rng = Rng::new(options.seed);
    let mut bytes = vec![];
    for index in 0..options.max_bytes {
        let scores = engine.to_f32(&logits)?;
        let token = sample(&scores, options, &mut rng)?;
        if token == EOS {
            return Ok(Generation { bytes, ended: true });
        }
        let byte = token as u8;
        bytes.push(byte);
        emit(byte)?;
        tokens.push(token);
        if index + 1 < options.max_bytes {
            if cache.length == cache.capacity {
                cache.length = 0;
                tokens.drain(..tokens.len() - model.config.context);
                logits = model.forward_cached(engine, &weights, &tokens, &mut cache)?;
            } else {
                logits = model.forward_cached(engine, &weights, &[token], &mut cache)?;
            }
        }
    }
    Ok(Generation {
        bytes,
        ended: false,
    })
}
/// Emits complete UTF-8 sequences while retaining a partial final code point.
pub struct TextStream<W: Write> {
    writer: W,
    pending: Vec<u8>,
}
impl<W: Write> TextStream<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            pending: vec![],
        }
    }
    pub fn push(&mut self, byte: u8) -> Result<()> {
        self.pending.push(byte);
        self.flush_complete(false)
    }
    pub fn finish(&mut self) -> Result<()> {
        self.flush_complete(true)
    }
    fn flush_complete(&mut self, finish: bool) -> Result<()> {
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(_) => {
                    self.writer.write_all(&self.pending)?;
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    if valid > 0 {
                        self.writer.write_all(&self.pending[..valid])?;
                        self.pending.drain(..valid);
                    }
                    if let Some(length) = error.error_len() {
                        self.writer.write_all("\u{fffd}".as_bytes())?;
                        self.pending.drain(..length);
                    } else {
                        if finish {
                            self.writer.write_all("\u{fffd}".as_bytes())?;
                            self.pending.clear();
                        }
                        break;
                    }
                }
            }
        }
        self.writer.flush()?;
        Ok(())
    }
}

pub fn generate_artifact(
    engine: &mut Engine,
    path: &std::path::Path,
    prompt: &[u8],
    options: &Sampling,
    mut emit: impl FnMut(u8) -> Result<()>,
) -> Result<Generation> {
    if path.join("native.json").exists() {
        let composite = native::Composite::load(engine, path)?;
        let text =
            std::str::from_utf8(prompt).map_err(|e| sejong_core::Error::Message(e.to_string()))?;
        let query = sejong_core::native::Query {
            question: text.into(),
            full: true,
            ..Default::default()
        };
        let answer = composite.query(engine, &query)?;
        let positive: Vec<_> = answer
            .probabilities
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.)
            .collect();
        let logits: Vec<_> = positive.iter().map(|(_, p)| p.ln() as f32).collect();
        let mut rng = Rng::new(options.seed);
        let selected = positive[sample(&logits, options, &mut rng)? as usize].0;
        let bytes = if selected < answer.values.len() {
            answer.values[selected].as_bytes().to_vec()
        } else {
            vec![]
        };
        ensure(
            bytes.len() <= options.max_bytes,
            "selected native value exceeds byte budget",
        )?;
        for &b in &bytes {
            emit(b)?;
        }
        return Ok(Generation { bytes, ended: true });
    }
    let (model, _) = Model::load(engine, path)?;
    generate(engine, &model, prompt, options, emit)
}
