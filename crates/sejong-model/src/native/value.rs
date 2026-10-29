//! Native categorical value output from one causal request-state computation.
use super::{atomic_directory, signature};
use crate::{Model, Parameter, load_tensors, save_tensors};
use sejong_core::{
    DType, ModelConfig, Result, Rng, ensure, json,
    native::{ByteValue, inventory_bytes, softmax},
};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FORMAT: &str = "sejong-native-value-v1";
pub const BOS: i32 = 257;
pub const SEP: i32 = 258;
#[derive(Clone, Serialize, Deserialize)]
pub struct ValueMetadata {
    pub format: String,
    pub requests: Vec<ByteValue>,
    pub prompts: Vec<ByteValue>,
    pub values: Vec<ByteValue>,
    pub model: ModelConfig,
    pub steps: usize,
    pub source_hash: String,
    pub loss: f64,
}
pub struct NativeValue {
    pub body: Model,
    pub embedding: Tensor,
    pub head: Tensor,
    pub bias: Tensor,
    pub metadata: ValueMetadata,
    seal: Option<String>,
}
impl NativeValue {
    pub fn initialize(e: &mut Engine, metadata: ValueMetadata) -> Result<Self> {
        ensure(metadata.format == FORMAT, "native value format mismatch")?;
        metadata.model.validate()?;
        inventory_bytes(&metadata.requests, "native requests")?;
        inventory_bytes(&metadata.values, "native values")?;
        ensure(
            metadata.requests.len() == metadata.prompts.len()
                && metadata
                    .prompts
                    .iter()
                    .all(|r| r.len() + 2 <= metadata.model.context),
            "native prompt inventory/context mismatch",
        )?;
        let d = metadata.model.width;
        let n = metadata.values.len();
        let mut rng = Rng::new(metadata.model.seed ^ 0x76616c7565);
        let embedding = e.from_f32(
            &[259, d],
            &(0..259 * d)
                .map(|_| rng.normal() / (d as f32).sqrt())
                .collect::<Vec<_>>(),
        )?;
        let head = e.from_f32(
            &[d, n],
            &(0..d * n)
                .map(|_| rng.normal() / (d as f32).sqrt())
                .collect::<Vec<_>>(),
        )?;
        let bias = e.full(&[1, n], DType::F32, 0.)?;
        let body = Model::initialize(e, metadata.model.clone())?;
        Ok(Self {
            body,
            embedding,
            head,
            bias,
            metadata,
            seal: None,
        })
    }
    pub fn requests(&self) -> &[ByteValue] {
        &self.metadata.requests
    }
    pub fn values(&self) -> &[ByteValue] {
        &self.metadata.values
    }
    pub fn parameters(&self) -> Vec<Parameter> {
        self.body
            .parameters
            .iter()
            .filter(|p| p.name != "embedding" && p.name != "output")
            .map(|p| Parameter {
                name: p.name.clone(),
                value: p.value.clone(),
            })
            .chain([
                Parameter {
                    name: "native_embedding".into(),
                    value: self.embedding.clone(),
                },
                Parameter {
                    name: "native_value".into(),
                    value: self.head.clone(),
                },
                Parameter {
                    name: "native_bias".into(),
                    value: self.bias.clone(),
                },
            ])
            .collect()
    }
    pub fn forward(&self, g: &mut Graph, index: usize, training: bool) -> Result<(Var, Vec<Var>)> {
        self.forward_batch(g, &[index], training)
    }
    pub fn forward_batch(
        &self,
        g: &mut Graph,
        indices: &[usize],
        training: bool,
    ) -> Result<(Var, Vec<Var>)> {
        ensure(
            !indices.is_empty() && indices.iter().all(|i| *i < self.metadata.requests.len()),
            "request outside native inventory",
        )?;
        let sequence = indices
            .iter()
            .map(|i| self.metadata.prompts[*i].len() + 2)
            .max()
            .unwrap();
        let mut ids = vec![256; indices.len() * sequence];
        let mut selected = vec![];
        for (row, &i) in indices.iter().enumerate() {
            let prompt = &self.metadata.prompts[i];
            let start = row * sequence;
            ids[start] = BOS;
            for (offset, byte) in prompt.as_bytes().iter().enumerate() {
                ids[start + 1 + offset] = *byte as i32;
            }
            ids[start + 1 + prompt.len()] = SEP;
            selected.push((start + 1 + prompt.len()) as i32);
        }
        let tensor = g.engine.from_i32(&[ids.len()], &ids)?;
        let b = self.body.bind(g, training)?;
        let embedding = g.leaf(self.embedding.clone(), training);
        let compute = g.cast(embedding, self.body.config.precision)?;
        let input = g.embedding(compute, &tensor, self.body.config.precision)?;
        let hidden = self.body.hidden_from(g, &b, input, sequence)?;
        let last = g.engine.from_i32(&[selected.len()], &selected)?;
        let selected = g.gather(hidden, &last)?;
        let selected = g.cast(selected, DType::F32)?;
        let head = g.leaf(self.head.clone(), training);
        let bias = g.leaf(self.bias.clone(), training);
        let logits = g.affine(selected, head, bias)?;
        let mut leaves = self
            .body
            .parameters
            .iter()
            .zip(b.leaves)
            .filter(|(p, _)| p.name != "embedding" && p.name != "output")
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        leaves.extend([embedding, head, bias]);
        Ok((logits, leaves))
    }
    pub fn signature(&self, e: &mut Engine) -> Result<String> {
        signature(e, &self.parameters(), &self.metadata)
    }
    pub fn seal(&mut self, e: &mut Engine) -> Result<()> {
        ensure(self.seal.is_none(), "native original already sealed")?;
        self.seal = Some(self.signature(e)?);
        Ok(())
    }
    pub fn verify(&self, e: &mut Engine) -> Result<()> {
        let expected = self.seal.as_ref().ok_or_else(|| {
            sejong_core::Error::Message("native original requires a trained sealed snapshot".into())
        })?;
        ensure(
            self.signature(e)? == *expected,
            "native value original changed after sealing",
        )
    }
    pub fn row(&self, e: &mut Engine, index: usize) -> Result<Vec<f64>> {
        self.verify(e)?;
        let mut g = Graph::new(e, false);
        let (logits, _) = self.forward(&mut g, index, false)?;
        softmax(
            &g.value(logits)?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>(),
        )
    }
    fn write(&self, e: &mut Engine, dir: &Path) -> Result<()> {
        self.body.save(
            e,
            &dir.join("body"),
            self.body.manifest(self.metadata.steps),
            None,
        )?;
        save_tensors(
            e,
            &dir.join("native.safetensors"),
            [
                ("embedding", &self.embedding),
                ("value", &self.head),
                ("bias", &self.bias),
            ],
        )?;
        std::fs::write(dir.join("value.json"), json::to_vec_pretty(&self.metadata)?)?;
        Ok(())
    }
    pub fn save(&self, e: &mut Engine, path: &Path) -> Result<()> {
        self.verify(e)?;
        atomic_directory(path, |dir| {
            self.write(e, dir)?;
            std::fs::write(dir.join("signature"), self.signature(e)?)?;
            Ok(())
        })
    }
    pub fn save_training(&self, e: &mut Engine, path: &Path) -> Result<()> {
        atomic_directory(path, |dir| self.write(e, dir))
    }
    pub fn load_training(e: &mut Engine, path: &Path) -> Result<Self> {
        let metadata: ValueMetadata = json::from_slice(&std::fs::read(path.join("value.json"))?)?;
        ensure(metadata.format == FORMAT, "native value format mismatch")?;
        inventory_bytes(&metadata.requests, "requests")?;
        inventory_bytes(&metadata.values, "values")?;
        let (body, _) = Model::load(e, &path.join("body"))?;
        let mut tensors = load_tensors(e, &path.join("native.safetensors"))?;
        let mut take = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let t = tensors
                .remove(name)
                .ok_or_else(|| sejong_core::Error::Message(format!("missing {name}")))?;
            ensure(
                t.shape == shape && t.dtype == DType::F32,
                "native value tensor contract mismatch",
            )?;
            Ok(t)
        };
        let d = metadata.model.width;
        let n = metadata.values.len();
        let embedding = take("embedding", &[259, d])?;
        let head = take("value", &[d, n])?;
        let bias = take("bias", &[1, n])?;
        ensure(
            tensors.is_empty()
                && json::to_value(&body.config)? == json::to_value(&metadata.model)?
                && metadata.requests.len() == metadata.prompts.len()
                && metadata
                    .prompts
                    .iter()
                    .all(|r| r.len() + 2 <= metadata.model.context),
            "native value stored contract mismatch",
        )?;
        Ok(Self {
            body,
            embedding,
            head,
            bias,
            metadata,
            seal: None,
        })
    }
    pub fn load(e: &mut Engine, path: &Path) -> Result<Self> {
        let mut model = Self::load_training(e, path)?;
        let expected = std::fs::read_to_string(path.join("signature"))?;
        ensure(
            model.signature(e)? == expected,
            "native value signature mismatch",
        )?;
        model.seal = Some(expected);
        Ok(model)
    }
}
