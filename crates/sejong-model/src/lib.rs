pub mod native;
// Independent byte Transformers, named weights and atomic model artifacts.
use sejong_core::{DType, EOS, Error, Manifest, ModelConfig, Result, Rng, VOCAB, ensure, json};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

pub struct Parameter {
    pub name: String,
    pub value: Tensor,
}
pub struct Model {
    pub source_manifest: Option<Manifest>,
    pub config: ModelConfig,
    pub parameters: Vec<Parameter>,
    index: HashMap<String, usize>,
}
pub struct Binding {
    pub leaves: Vec<Var>,
    pub compute: Vec<Var>,
}
pub struct Cache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub length: usize,
    pub capacity: usize,
}

fn layout(c: &ModelConfig) -> Vec<(String, Vec<usize>, bool)> {
    let d = c.width;
    let f = c.ffn_width;
    let mut out = vec![("embedding".into(), vec![VOCAB, d], false)];
    for i in 0..c.layers {
        for name in ["attention_norm", "ffn_norm"] {
            out.push((format!("layers.{i}.{name}"), vec![d], true));
        }
        for name in ["query", "key", "value", "attention_output"] {
            out.push((format!("layers.{i}.{name}"), vec![d, d], false));
        }
        for name in ["gate", "up"] {
            out.push((format!("layers.{i}.{name}"), vec![d, f], false));
        }
        out.push((format!("layers.{i}.down"), vec![f, d], false));
    }
    out.push(("final_norm".into(), vec![d], true));
    out.push(("output".into(), vec![d, VOCAB], false));
    out
}
impl Model {
    pub fn initialize(engine: &mut Engine, config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let mut rng = Rng::new(config.seed);
        let mut parameters = vec![];
        for (name, shape, norm) in layout(&config) {
            let std = if name == "embedding" {
                1.0 / (config.width as f32).sqrt()
            } else {
                1.0 / (shape[0] as f32).sqrt()
            };
            let std = if name.ends_with(".down") || name.ends_with(".attention_output") {
                std / (2. * config.layers as f32).sqrt()
            } else {
                std
            };
            let data: Vec<f32> = (0..shape.iter().product())
                .map(|_| if norm { 1. } else { rng.normal() * std })
                .collect();
            parameters.push(Parameter {
                name,
                value: engine.from_f32(&shape, &data)?,
            });
        }
        let index = parameters
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();
        Ok(Self {
            source_manifest: None,
            config,
            parameters,
            index,
        })
    }
    pub fn load(engine: &mut Engine, directory: &Path) -> Result<(Self, Manifest)> {
        let manifest = Manifest::load(directory)?;
        let tensors = load_tensors(engine, &directory.join(&manifest.weights))?;
        let mut parameters = vec![];
        let expected = layout(&manifest.model);
        ensure(
            tensors.len() == expected.len(),
            "model tensor count differs from configuration",
        )?;
        for (name, shape, _) in expected {
            let value = tensors
                .get(&name)
                .ok_or_else(|| Error::Message(format!("missing tensor {name}")))?;
            ensure(
                value.shape == shape && value.dtype == DType::F32,
                format!("tensor {name} shape or dtype differs from configuration"),
            )?;
            parameters.push(Parameter {
                name,
                value: value.clone(),
            });
        }
        let index = parameters
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name.clone(), i))
            .collect();
        Ok((
            Self {
                source_manifest: Some(manifest.clone()),
                config: manifest.model.clone(),
                parameters,
                index,
            },
            manifest,
        ))
    }
    pub fn parameter_count(&self) -> usize {
        self.parameters.iter().map(|p| p.value.len()).sum()
    }
    pub fn parameter(&self, name: &str) -> &Tensor {
        &self.parameters[self.index[name]].value
    }
    pub fn bind(&self, g: &mut Graph, trainable: bool) -> Result<Binding> {
        let mut leaves = vec![];
        let mut compute = vec![];
        for p in &self.parameters {
            let leaf = g.leaf(p.value.clone(), trainable);
            let cast = g.cast(leaf, self.config.precision)?;
            leaves.push(leaf);
            compute.push(cast);
        }
        Ok(Binding { leaves, compute })
    }
    fn w(&self, b: &Binding, name: &str) -> Var {
        b.compute[self.index[name]]
    }
    pub fn hidden(
        &self,
        g: &mut Graph,
        b: &Binding,
        ids: &Tensor,
        sequence: usize,
        injection: Option<Var>,
    ) -> Result<Var> {
        ensure(
            sequence > 0 && sequence <= self.config.context && ids.len().is_multiple_of(sequence),
            "input sequence exceeds context or has partial batch",
        )?;
        let c = &self.config;
        let mut x = g.embedding(self.w(b, "embedding"), ids, c.precision)?;
        if let Some(input) = injection {
            x = g.add(x, input)?;
        }
        self.hidden_from(g, b, x, sequence)
    }
    pub fn hidden_from(
        &self,
        g: &mut Graph,
        b: &Binding,
        mut x: Var,
        sequence: usize,
    ) -> Result<Var> {
        ensure(
            sequence > 0
                && sequence <= self.config.context
                && g.tensor(x).shape[0].is_multiple_of(sequence)
                && g.tensor(x).shape[1] == self.config.width,
            "embedded input shape/context mismatch",
        )?;
        let c = &self.config;
        for layer in 0..c.layers {
            let key = |name: &str| format!("layers.{layer}.{name}");
            let mut inputs = vec![x];
            for name in [
                "attention_norm",
                "query",
                "key",
                "value",
                "attention_output",
                "ffn_norm",
                "gate",
                "up",
                "down",
            ] {
                inputs.push(self.w(b, &key(name)));
            }
            let (heads, theta, eps) = (c.heads, c.rope_theta, c.norm_epsilon);
            x = g.checkpoint(&inputs, move |g, v| {
                transformer_block(g, v, sequence, heads, theta, eps)
            })?;
        }
        g.norm(x, self.w(b, "final_norm"), c.norm_epsilon)
    }
    pub fn logits(&self, g: &mut Graph, b: &Binding, ids: &Tensor, sequence: usize) -> Result<Var> {
        let h = self.hidden(g, b, ids, sequence, None)?;
        let h = g.cast(h, DType::F32)?;
        let output = b.leaves[self.index["output"]];
        g.matmul(h, output)
    }
    pub fn inference_weights(&self, engine: &mut Engine) -> Result<HashMap<String, Tensor>> {
        self.parameters
            .iter()
            .map(|p| {
                Ok((
                    p.name.clone(),
                    engine.cast(
                        &p.value,
                        if p.name == "output" {
                            DType::F32
                        } else {
                            self.config.precision
                        },
                    )?,
                ))
            })
            .collect()
    }
    pub fn new_cache(&self, engine: &mut Engine) -> Result<Cache> {
        let mut keys = vec![];
        let mut values = vec![];
        for _ in 0..self.config.layers {
            keys.push(engine.empty(
                &[self.config.context, self.config.width],
                self.config.precision,
            )?);
            values.push(engine.empty(
                &[self.config.context, self.config.width],
                self.config.precision,
            )?);
        }
        Ok(Cache {
            keys,
            values,
            length: 0,
            capacity: self.config.context,
        })
    }
    pub fn forward_cached(
        &self,
        engine: &mut Engine,
        weights: &HashMap<String, Tensor>,
        tokens: &[i32],
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let h = self.hidden_cached(engine, weights, tokens, cache, None)?;
        let h = engine.cast(&h.rows(tokens.len() - 1, 1)?, DType::F32)?;
        engine.matmul(&h, &weights["output"], false, false, DType::F32)
    }
    pub fn hidden_cached(
        &self,
        engine: &mut Engine,
        weights: &HashMap<String, Tensor>,
        tokens: &[i32],
        cache: &mut Cache,
        injection: Option<&Tensor>,
    ) -> Result<Tensor> {
        ensure(
            !tokens.is_empty() && tokens.iter().all(|v| (0..=EOS).contains(v)),
            "input tokens must be bytes or EOS",
        )?;
        ensure(
            cache.length + tokens.len() <= cache.capacity,
            "KV cache context limit reached",
        )?;
        let c = &self.config;
        let ids = engine.from_i32(&[tokens.len()], tokens)?;
        let mut x = engine.embedding(&weights["embedding"], &ids, c.precision)?;
        if let Some(input) = injection {
            x = engine.add(&x, input, 1., c.precision)?;
        }
        for layer in 0..c.layers {
            let w = |name: &str| &weights[&format!("layers.{layer}.{name}")];
            let z = engine.norm(&x, w("attention_norm"), c.norm_epsilon)?;
            let q = engine.matmul(&z, w("query"), false, false, c.precision)?;
            let k = engine.matmul(&z, w("key"), false, false, c.precision)?;
            let v = engine.matmul(&z, w("value"), false, false, c.precision)?;
            let q = engine.rope(&q, tokens.len(), cache.length, c.heads, c.rope_theta, false)?;
            let k = engine.rope(&k, tokens.len(), cache.length, c.heads, c.rope_theta, false)?;
            engine.copy_into(&k, &cache.keys[layer].rows(cache.length, tokens.len())?)?;
            engine.copy_into(&v, &cache.values[layer].rows(cache.length, tokens.len())?)?;
            let all_k = cache.keys[layer].rows(0, cache.length + tokens.len())?;
            let all_v = cache.values[layer].rows(0, cache.length + tokens.len())?;
            let (attn, _) = engine.attention(
                &q,
                &all_k,
                &all_v,
                tokens.len(),
                cache.length + tokens.len(),
                cache.length,
                c.heads,
            )?;
            let projected =
                engine.matmul(&attn, w("attention_output"), false, false, c.precision)?;
            x = engine.add(&x, &projected, 1., c.precision)?;
            let z = engine.norm(&x, w("ffn_norm"), c.norm_epsilon)?;
            let gate = engine.matmul(&z, w("gate"), false, false, c.precision)?;
            let up = engine.matmul(&z, w("up"), false, false, c.precision)?;
            let activated = engine.swiglu(&gate, &up)?;
            let projected = engine.matmul(&activated, w("down"), false, false, c.precision)?;
            x = engine.add(&x, &projected, 1., c.precision)?;
        }
        let h = engine.norm(&x, &weights["final_norm"], c.norm_epsilon)?;
        cache.length += tokens.len();
        Ok(h)
    }
    pub fn save(
        &self,
        engine: &mut Engine,
        directory: &Path,
        mut manifest: Manifest,
        optimizer: Option<&BTreeMap<String, Tensor>>,
    ) -> Result<()> {
        ensure(
            !directory.exists(),
            format!(
                "artifact destination already exists: {}",
                directory.display()
            ),
        )?;
        let parent = directory.parent().unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = directory
            .file_name()
            .ok_or_else(|| Error::Message("artifact destination requires a directory name".into()))?
            .to_string_lossy();
        let staging = parent.join(format!(".{name}.{}.tmp", std::process::id()));
        std::fs::create_dir(&staging)?;
        let result = (|| {
            manifest.weights = "weights.safetensors".into();
            save_tensors(
                engine,
                &staging.join(&manifest.weights),
                self.parameters.iter().map(|p| (p.name.as_str(), &p.value)),
            )?;
            if let Some(state) = optimizer {
                manifest.optimizer = Some("optimizer.safetensors".into());
                save_tensors(
                    engine,
                    &staging.join("optimizer.safetensors"),
                    state.iter().map(|(k, v)| (k.as_str(), v)),
                )?;
            }
            std::fs::write(
                staging.join("manifest.json"),
                json::to_vec_pretty(&manifest)?,
            )?;
            std::fs::rename(&staging, directory)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result
    }
    pub fn manifest(&self, step: usize) -> Manifest {
        Manifest {
            format: 1,
            model: self.config.clone(),
            step,
            weights: "weights.safetensors".into(),
            optimizer: None,
            training: None,
            data_fingerprint: None,
            rng_state: None,
        }
    }
}

pub fn save_tensors<'a>(
    engine: &mut Engine,
    path: &Path,
    tensors: impl IntoIterator<Item = (&'a str, &'a Tensor)>,
) -> Result<()> {
    let mut owned = vec![];
    for (name, tensor) in tensors {
        let (dtype, bytes) = if tensor.dtype == DType::F64 {
            (
                safetensors::Dtype::F64,
                engine
                    .to_f64(tensor)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        } else {
            (
                safetensors::Dtype::F32,
                engine
                    .to_f32(tensor)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        owned.push((name.to_string(), tensor.shape.clone(), dtype, bytes));
    }
    let views: Vec<_> = owned
        .iter()
        .map(|(name, shape, dtype, data)| {
            Ok((
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                    .map_err(|e| Error::Message(e.to_string()))?,
            ))
        })
        .collect::<Result<_>>()?;
    safetensors::tensor::serialize_to_file(views, None, path)
        .map_err(|e| Error::Message(e.to_string()))
}
pub fn load_tensors(engine: &mut Engine, path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let storage =
        safetensors::SafeTensors::deserialize(&mmap).map_err(|e| Error::Message(e.to_string()))?;
    let mut out = BTreeMap::new();
    for (name, tensor) in storage.tensors() {
        ensure(
            matches!(
                tensor.dtype(),
                safetensors::Dtype::F32 | safetensors::Dtype::F64
            ),
            format!("tensor {name} requires F32 storage"),
        )?;
        if tensor.dtype() == safetensors::Dtype::F64 {
            let values: Vec<f64> = tensor
                .data()
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            ensure(
                values.iter().all(|x| x.is_finite()),
                "nonfinite f64 weights",
            )?;
            out.insert(name, engine.from_f64(tensor.shape(), &values)?);
            continue;
        }
        let values: Vec<f32> = tensor
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        ensure(
            values.iter().all(|v| v.is_finite()),
            format!("tensor {name} contains nonfinite values"),
        )?;
        out.insert(name, engine.from_f32(tensor.shape(), &values)?);
    }
    Ok(out)
}

fn transformer_block(
    g: &mut Graph,
    v: &[Var],
    sequence: usize,
    heads: usize,
    theta: f32,
    eps: f32,
) -> Result<Var> {
    let mut x = v[0];
    let z = g.norm(x, v[1], eps)?;
    let q = g.matmul(z, v[2])?;
    let k = g.matmul(z, v[3])?;
    let value = g.matmul(z, v[4])?;
    let q = g.rope(q, sequence, 0, heads, theta)?;
    let k = g.rope(k, sequence, 0, heads, theta)?;
    let attention = g.attention(q, k, value, sequence, heads)?;
    let projected = g.matmul(attention, v[5])?;
    x = g.add(x, projected)?;
    let z = g.norm(x, v[6], eps)?;
    let gate = g.matmul(z, v[7])?;
    let up = g.matmul(z, v[8])?;
    let activated = g.swiglu(gate, up)?;
    let projected = g.matmul(activated, v[9])?;
    g.add(x, projected)
}
