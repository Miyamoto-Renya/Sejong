mod architecture;
pub mod native;
// Independent byte Transformers, named weights and atomic model artifacts.
use sejong_core::{DType, EOS, Error, Manifest, ModelConfig, Result, Rng, ensure, json};
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
    corrections: Vec<Option<LowRankCorrection>>,
}
#[derive(Clone)]
pub struct LowRankCorrection {
    pub parameter_index: usize,
    pub left: Tensor,
    pub coefficient: Tensor,
    pub right: Tensor,
}
pub struct Cache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub length: usize,
    pub capacity: usize,
    hierarchy: Option<architecture::HierarchyCache>,
}
impl Cache {
    pub fn reset(&mut self) {
        self.length = 0;
        if let Some(state) = &mut self.hierarchy {
            state.reset();
        }
    }
}

fn layout(c: &ModelConfig) -> Vec<(String, Vec<usize>, bool)> {
    architecture::layout(c)
}
impl Model {
    pub fn initialize(engine: &mut Engine, config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let required = config.parameter_tensor_groups()?.into_iter().try_fold(
            0usize,
            |total, (elements, count)| {
                let bytes = elements
                    .checked_mul(DType::F32.bytes())
                    .ok_or_else(|| Error::Message("model parameter byte size overflow".into()))?;
                Engine::allocation_bytes(bytes)?
                    .checked_mul(count)
                    .and_then(|n| total.checked_add(n))
                    .ok_or_else(|| {
                        Error::Message("model parameter allocation size overflow".into())
                    })
            },
        )?;
        let available = engine.available_memory_bytes()?;
        if required > available {
            return Err(Error::Capacity {
                operation: "model initialization".into(),
                required,
                available,
            });
        }
        let mut rng = Rng::new(config.seed);
        let mut parameters = vec![];
        for (name, shape, norm) in layout(&config) {
            let std = if name == "embedding" {
                1.0 / (shape[1] as f32).sqrt()
            } else {
                1.0 / (shape[0] as f32).sqrt()
            };
            let depth = config.hierarchy.as_ref().map_or(config.layers, |h| {
                if name.starts_with("encoder.") {
                    h.encoder_layers
                } else if name.starts_with("decoder.") {
                    h.decoder_layers
                } else {
                    config.layers
                }
            });
            let std = if name.ends_with(".down") || name.ends_with(".attention_output") {
                std / (2. * depth as f32).sqrt()
            } else {
                std
            };
            let elements = shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| Error::Message("model tensor shape overflow".into()))?;
            let data: Vec<f32> = (0..elements)
                .map(|_| {
                    if norm {
                        1.
                    } else if name == "global_bos" {
                        0.
                    } else {
                        rng.normal() * std
                    }
                })
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
                format_args!("tensor {name} shape or dtype differs from configuration"),
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
        self.bind_with_corrections(g, trainable, &[])
    }
    pub fn is_encoder_parameter(&self, index: usize) -> bool {
        let name = &self.parameters[index].name;
        name == "embedding"
            || name.starts_with("encoder.")
            || name == "encoder_norm"
            || name == "input_projection"
    }
    pub fn bind_encoder(&self, g: &mut Graph, trainable: bool) -> Result<Binding> {
        let mut leaves = Vec::with_capacity(self.parameters.len());
        let mut compute = Vec::with_capacity(self.parameters.len());
        for (i, p) in self.parameters.iter().enumerate() {
            let selected = self.is_encoder_parameter(i);
            let leaf = g.parameter(p.value.clone(), trainable && selected);
            leaves.push(leaf);
            compute.push(if selected {
                g.cast(leaf, self.config.precision)?
            } else {
                leaf
            });
        }
        Ok(Binding {
            leaves,
            compute,
            corrections: vec![None; self.parameters.len()],
        })
    }
    pub fn bind_with_corrections(
        &self,
        g: &mut Graph,
        trainable: bool,
        corrections: &[LowRankCorrection],
    ) -> Result<Binding> {
        let mut leaves = vec![];
        let mut compute = vec![];
        for p in &self.parameters {
            let leaf = g.parameter(p.value.clone(), trainable);
            let cast = g.cast(leaf, self.config.precision)?;
            leaves.push(leaf);
            compute.push(cast);
        }
        let mut bound = vec![None; self.parameters.len()];
        for correction in corrections {
            ensure(
                correction.parameter_index < self.parameters.len(),
                "correction parameter index exceeds model",
            )?;
            let shape = &self.parameters[correction.parameter_index].value.shape;
            ensure(
                shape.len() == 2
                    && correction.left.shape.len() == 2
                    && correction.right.shape.len() == 2
                    && correction.coefficient.shape.len() == 2,
                "low-rank correction requires matrices",
            )?;
            let r = correction.left.shape[1];
            ensure(
                r > 0
                    && correction.left.shape[0] == shape[0]
                    && correction.right.shape == [shape[1], r]
                    && correction.coefficient.shape == [r, r]
                    && bound[correction.parameter_index].is_none(),
                "low-rank correction shape or duplicate parameter mismatch",
            )?;
            bound[correction.parameter_index] = Some(correction.clone());
        }
        Ok(Binding {
            leaves,
            compute,
            corrections: bound,
        })
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
        if self.config.hierarchy.is_some() {
            let encoded = architecture::encode(self, g, b, ids, sequence, injection)?;
            return architecture::hidden_from_encoded(self, g, b, encoded, sequence);
        }
        let c = &self.config;
        let mut x = g.embedding(self.w(b, "embedding"), ids, c.precision)?;
        if let Some(input) = injection {
            x = g.add(x, input)?;
        }
        architecture::hidden_from(self, g, b, x, sequence)
    }
    pub fn hidden_from(&self, g: &mut Graph, b: &Binding, x: Var, sequence: usize) -> Result<Var> {
        ensure(
            sequence > 0
                && sequence <= self.config.context
                && g.tensor(x).shape.len() == 2
                && g.tensor(x).shape[0].is_multiple_of(sequence)
                && g.tensor(x).shape[1] == self.config.width,
            "embedded input shape/context mismatch",
        )?;
        architecture::hidden_from(self, g, b, x, sequence)
    }
    pub fn encoder_width(&self) -> usize {
        self.config
            .hierarchy
            .as_ref()
            .map_or(self.config.width, |h| h.encoder_width)
    }
    pub fn encode(
        &self,
        g: &mut Graph,
        b: &Binding,
        ids: &Tensor,
        sequence: usize,
        injection: Option<Var>,
    ) -> Result<Var> {
        ensure(
            sequence > 0 && sequence <= self.config.context && ids.len().is_multiple_of(sequence),
            "encoder sequence/context mismatch",
        )?;
        architecture::encode(self, g, b, ids, sequence, injection)
    }
    pub fn hidden_from_encoded(
        &self,
        g: &mut Graph,
        b: &Binding,
        encoded: Var,
        sequence: usize,
    ) -> Result<Var> {
        ensure(
            sequence > 0
                && sequence <= self.config.context
                && g.tensor(encoded).shape.len() == 2
                && g.tensor(encoded).shape[0].is_multiple_of(sequence)
                && g.tensor(encoded).shape[1] == self.encoder_width(),
            "encoded input shape/context mismatch",
        )?;
        architecture::hidden_from_encoded(self, g, b, encoded, sequence)
    }
    pub fn logits_from_encoded(
        &self,
        g: &mut Graph,
        b: &Binding,
        encoded: Var,
        sequence: usize,
    ) -> Result<Var> {
        let h = self.hidden_from_encoded(g, b, encoded, sequence)?;
        let h = g.cast(h, DType::F32)?;
        architecture::output(self, g, b, h)
    }
    pub fn logits(&self, g: &mut Graph, b: &Binding, ids: &Tensor, sequence: usize) -> Result<Var> {
        let h = self.hidden(g, b, ids, sequence, None)?;
        let h = g.cast(h, DType::F32)?;
        architecture::output(self, g, b, h)
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
        architecture::new_cache(self, engine)
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
        if cache.length == 0 {
            cache.reset();
        }
        architecture::hidden_cached(self, engine, weights, tokens, cache, injection)
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
            format_args!(
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
            format_args!("tensor {name} requires F32 storage"),
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
            format_args!("tensor {name} contains nonfinite values"),
        )?;
        out.insert(name, engine.from_f32(tensor.shape(), &values)?);
    }
    Ok(out)
}
