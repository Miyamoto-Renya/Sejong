//! Frozen MiniLM BERT execution and span-preserving WordPiece input.
use crate::{
    Parameter,
    native::{atomic_directory, signature},
    save_tensors,
};
use sejong_core::{DType, Error, Result, ensure, json};
use sejong_gpu::{Engine, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderIdentity {
    pub model: String,
    pub source: String,
    pub config: BertConfig,
    pub digest: String,
}
#[derive(Clone)]
pub struct WordPiece {
    tokenizer: tokenizers::Tokenizer,
}
#[derive(Clone, Debug)]
pub struct Tokenized {
    pub ids: Vec<i32>,
    pub offsets: Vec<(usize, usize)>,
}
impl WordPiece {
    pub fn load(path: &Path) -> Result<Self> {
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Message(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Message(e.to_string()))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer })
    }
    pub fn tokenize(&self, text: &str) -> Result<Tokenized> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Message(e.to_string()))?;
        Ok(Tokenized {
            ids: encoded.get_ids().iter().map(|x| *x as i32).collect(),
            offsets: encoded.get_offsets().to_vec(),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Features {
    pub rows: Vec<f32>,
    pub present: Vec<bool>,
    pub predicate_occurrences: usize,
}
/// Identity names receive first-mention aliases; their original coordinates stay in spans.
pub fn aliased(
    text: &str,
    people: &[String],
    predicates: &[String],
) -> Result<(String, Vec<(usize, usize, usize)>, usize)> {
    let names: Vec<_> = people.iter().chain(predicates).collect();
    let unique: std::collections::BTreeSet<_> = names.iter().collect();
    ensure(
        unique.len() == names.len() && names.iter().all(|s| !s.is_empty()),
        "unique nonempty identities required",
    )?;
    let mut order: Vec<_> = (0..names.len()).collect();
    order.sort_by_key(|i| std::cmp::Reverse(names[*i].len()));
    let mut person_alias = BTreeMap::new();
    let mut predicate_alias = BTreeMap::new();
    let mut out = String::new();
    let mut spans = vec![];
    let mut cursor = 0;
    let mut count = 0;
    while cursor < text.len() {
        let before = text[..cursor].chars().next_back();
        let found = order.iter().copied().find(|i| {
            let name = names[*i].as_str();
            let end = cursor + name.len();
            text[cursor..].starts_with(name)
                && before.is_none_or(|c| !c.is_alphanumeric())
                && (end == text.len()
                    || text[end..]
                        .chars()
                        .next()
                        .is_none_or(|c| !c.is_alphanumeric()))
        });
        if let Some(i) = found {
            let (slot, alias) = if i < people.len() {
                let next = person_alias.len();
                let ordinal = *person_alias.entry(i).or_insert(next);
                let alias = if ordinal < 4 {
                    ["Alice", "Bob", "Carol", "David"][ordinal].to_string()
                } else {
                    format!("Person{}", ordinal + 1)
                };
                (i, alias)
            } else {
                let index = i - people.len();
                let next = predicate_alias.len();
                let ordinal = *predicate_alias.entry(index).or_insert(next);
                count += 1;
                let alias = if ordinal < 4 {
                    ["called", "thanked", "helped", "invited"][ordinal].to_string()
                } else {
                    format!("relation{}", ordinal + 1)
                };
                (people.len() + index, alias)
            };
            let start = out.len();
            out.push_str(&alias);
            spans.push((start, out.len(), slot));
            cursor += names[i].len();
        } else {
            let c = text[cursor..].chars().next().unwrap();
            out.push(c);
            cursor += c.len_utf8();
        }
    }
    Ok((out, spans, count))
}

pub struct BertEncoder {
    pub config: BertConfig,
    pub parameters: Vec<Parameter>,
    pub tokenizer: WordPiece,
    pub identity: EncoderIdentity,
    pub tokenizer_json: Vec<u8>,
}
impl BertEncoder {
    pub fn load_shared(e: &mut Engine, path: &Path) -> Result<Rc<Self>> {
        thread_local! {static SHARED:RefCell<BTreeMap<(usize,String),Weak<BertEncoder>>>=const {RefCell::new(BTreeMap::new())};}
        let mut hash = blake3::Hasher::new();
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            hash.update(file.as_bytes());
            hash.update(&std::fs::read(path.join(file))?);
        }
        let key = (e.context_id(), hash.finalize().to_hex().to_string());
        if let Some(encoder) = SHARED.with(|cache| cache.borrow().get(&key).and_then(Weak::upgrade))
        {
            return Ok(encoder);
        }
        let encoder = Rc::new(Self::load(e, path)?);
        SHARED.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.retain(|_, value| value.strong_count() > 0);
            cache.insert(key, Rc::downgrade(&encoder));
        });
        Ok(encoder)
    }
    pub fn load(e: &mut Engine, path: &Path) -> Result<Self> {
        let config: BertConfig = json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure(
            config.hidden_size > 0
                && config.num_hidden_layers > 0
                && config.num_attention_heads > 0
                && config
                    .hidden_size
                    .is_multiple_of(config.num_attention_heads)
                && config.intermediate_size > 0
                && config.max_position_embeddings > 0
                && config.vocab_size > 0
                && config.type_vocab_size > 0
                && config.layer_norm_eps.is_finite()
                && config.layer_norm_eps > 0.,
            "positive head-aligned BERT asset configuration required",
        )?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let storage = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|x| Error::Message(x.to_string()))?;
        let mut parameters = vec![];
        for (name, t) in storage.tensors() {
            if name == "embeddings.position_ids" {
                continue;
            }
            ensure(
                t.dtype() == safetensors::Dtype::F32,
                format!("MiniLM {name} requires F32"),
            )?;
            let values: Vec<_> = t
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            ensure(
                values.iter().all(|v| v.is_finite()),
                "finite MiniLM weights required",
            )?;
            parameters.push(Parameter {
                name,
                value: e.from_f32(t.shape(), &values)?,
            });
        }
        parameters.sort_by(|a, b| a.name.cmp(&b.name));
        let tokenizer_json = std::fs::read(path.join("tokenizer.json"))?;
        let tokenizer = WordPiece::load(&path.join("tokenizer.json"))?;
        ensure(
            tokenizer.tokenizer.get_vocab_size(true) == config.vocab_size,
            "MiniLM tokenizer and embedding vocabulary differ",
        )?;
        let digest = signature(e, &parameters, &(&config, &tokenizer_json))?;
        let identity = if path.join("encoder.json").exists() {
            let recorded: EncoderIdentity =
                json::from_slice(&std::fs::read(path.join("encoder.json"))?)?;
            ensure(
                recorded.digest == digest
                    && json::to_value(&recorded.config)? == json::to_value(&config)?,
                "encoder identity differs from stored tensors/configuration",
            )?;
            recorded
        } else {
            let source: Option<json::Value> = std::fs::read(path.join("source.json"))
                .ok()
                .map(|x| json::from_slice(&x))
                .transpose()?;
            let repository = source.as_ref().and_then(|x| x["repository"].as_str());
            let revision = source.as_ref().and_then(|x| x["revision"].as_str());
            EncoderIdentity {
                model: repository.unwrap_or("bert-contextual-encoder").into(),
                source: match (repository, revision) {
                    (Some(repo), Some(rev)) => format!("https://huggingface.co/{repo}/tree/{rev}"),
                    _ => format!("content-addressed:{digest}"),
                },
                config: config.clone(),
                digest,
            }
        };
        let model = Self {
            config,
            parameters,
            tokenizer,
            identity,
            tokenizer_json,
        };
        model.validate()?;
        Ok(model)
    }
    fn validate(&self) -> Result<()> {
        let c = &self.config;
        let d = c.hidden_size;
        let f = c.intermediate_size;
        let mut expected = vec![
            (
                "embeddings.word_embeddings.weight".into(),
                vec![c.vocab_size, d],
            ),
            (
                "embeddings.position_embeddings.weight".into(),
                vec![c.max_position_embeddings, d],
            ),
            (
                "embeddings.token_type_embeddings.weight".into(),
                vec![c.type_vocab_size, d],
            ),
            ("embeddings.LayerNorm.weight".into(), vec![d]),
            ("embeddings.LayerNorm.bias".into(), vec![d]),
        ];
        for i in 0..c.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");
            for name in ["query", "key", "value"] {
                expected.push((format!("{prefix}.attention.self.{name}.weight"), vec![d, d]));
                expected.push((format!("{prefix}.attention.self.{name}.bias"), vec![d]));
            }
            for name in ["attention.output", "output"] {
                expected.push((format!("{prefix}.{name}.LayerNorm.weight"), vec![d]));
                expected.push((format!("{prefix}.{name}.LayerNorm.bias"), vec![d]));
            }
            expected.extend([
                (
                    format!("{prefix}.attention.output.dense.weight"),
                    vec![d, d],
                ),
                (format!("{prefix}.attention.output.dense.bias"), vec![d]),
                (format!("{prefix}.intermediate.dense.weight"), vec![f, d]),
                (format!("{prefix}.intermediate.dense.bias"), vec![f]),
                (format!("{prefix}.output.dense.weight"), vec![d, f]),
                (format!("{prefix}.output.dense.bias"), vec![d]),
            ]);
        }
        for (name, shape) in expected {
            ensure(
                self.parameters
                    .iter()
                    .any(|p| p.name == name && p.value.shape == shape),
                format!("MiniLM tensor contract differs: {name}"),
            )?;
        }
        Ok(())
    }
    fn w(&self, name: &str) -> &Tensor {
        &self
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap()
            .value
    }
    fn linear(&self, e: &mut Engine, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let y = e.matmul(
            x,
            self.w(&format!("{prefix}.weight")),
            false,
            true,
            DType::F32,
        )?;
        e.bias_add(
            &y,
            &self
                .w(&format!("{prefix}.bias"))
                .reshape(&[1, y.shape[1]])?,
        )
    }
    fn norm(&self, e: &mut Engine, x: &Tensor, prefix: &str) -> Result<Tensor> {
        e.layer_norm(
            x,
            self.w(&format!("{prefix}.weight")),
            &self
                .w(&format!("{prefix}.bias"))
                .reshape(&[1, x.shape[1]])?,
            self.config.layer_norm_eps,
        )
    }
    pub fn hidden(&self, e: &mut Engine, ids: &[i32]) -> Result<Tensor> {
        ensure(
            !ids.is_empty() && ids.len() <= self.config.max_position_embeddings,
            "MiniLM input length exceeds positional table",
        )?;
        let input = e.from_i32(&[ids.len()], ids)?;
        let positions = e.from_i32(&[ids.len()], &(0..ids.len() as i32).collect::<Vec<_>>())?;
        let types = e.from_i32(&[ids.len()], &vec![0; ids.len()])?;
        let words = e.embedding(
            self.w("embeddings.word_embeddings.weight"),
            &input,
            DType::F32,
        )?;
        let positions = e.embedding(
            self.w("embeddings.position_embeddings.weight"),
            &positions,
            DType::F32,
        )?;
        let types = e.embedding(
            self.w("embeddings.token_type_embeddings.weight"),
            &types,
            DType::F32,
        )?;
        let x = e.add(&words, &positions, 1., DType::F32)?;
        let x = e.add(&x, &types, 1., DType::F32)?;
        let mut x = self.norm(e, &x, "embeddings.LayerNorm")?;
        for i in 0..self.config.num_hidden_layers {
            let prefix = format!("encoder.layer.{i}");
            let q = self.linear(e, &x, &format!("{prefix}.attention.self.query"))?;
            let k = self.linear(e, &x, &format!("{prefix}.attention.self.key"))?;
            let v = self.linear(e, &x, &format!("{prefix}.attention.self.value"))?;
            let (a, _) = e.attention_mode(
                &q,
                &k,
                &v,
                ids.len(),
                ids.len(),
                0,
                self.config.num_attention_heads,
                false,
            )?;
            let a = self.linear(e, &a, &format!("{prefix}.attention.output.dense"))?;
            let sum = e.add(&x, &a, 1., DType::F32)?;
            x = self.norm(e, &sum, &format!("{prefix}.attention.output.LayerNorm"))?;
            let h = self.linear(e, &x, &format!("{prefix}.intermediate.dense"))?;
            let h = e.gelu(&h)?;
            let h = self.linear(e, &h, &format!("{prefix}.output.dense"))?;
            let sum = e.add(&x, &h, 1., DType::F32)?;
            x = self.norm(e, &sum, &format!("{prefix}.output.LayerNorm"))?;
        }
        Ok(x)
    }
    pub fn features(
        &self,
        e: &mut Engine,
        text: &str,
        people: &[String],
        predicates: &[String],
    ) -> Result<Features> {
        let (text, spans, predicate_occurrences) = aliased(text, people, predicates)?;
        let tokens = self.tokenizer.tokenize(&text)?;
        let hidden = self.hidden(e, &tokens.ids)?;
        let hidden = e.to_f32(&hidden)?;
        let d = self.config.hidden_size;
        let slots = people.len() + predicates.len();
        let mut rows = vec![0.; (1 + slots) * d];
        let mut present = vec![false; slots];
        for j in 0..d {
            rows[j] = (0..tokens.ids.len())
                .map(|r| hidden[r * d + j])
                .sum::<f32>()
                / tokens.ids.len() as f32;
        }
        for slot in 0..slots {
            let selected: Vec<_> = tokens
                .offsets
                .iter()
                .enumerate()
                .filter(|(_, offset)| {
                    offset.1 > offset.0
                        && spans.iter().any(|(start, end, s)| {
                            *s == slot && offset.0 >= *start && offset.1 <= *end
                        })
                })
                .map(|(i, _)| i)
                .collect();
            present[slot] = !selected.is_empty();
            if !selected.is_empty() {
                for j in 0..d {
                    rows[(slot + 1) * d + j] =
                        selected.iter().map(|i| hidden[i * d + j]).sum::<f32>()
                            / selected.len() as f32;
                }
            }
        }
        Ok(Features {
            rows,
            present,
            predicate_occurrences,
        })
    }
    pub fn verify(&self, e: &mut Engine) -> Result<()> {
        ensure(
            signature(e, &self.parameters, &(&self.config, &self.tokenizer_json))?
                == self.identity.digest,
            "frozen MiniLM encoder changed",
        )
    }
    pub fn save(&self, e: &mut Engine, path: &Path) -> Result<()> {
        self.verify(e)?;
        atomic_directory(path, |dir| {
            std::fs::write(dir.join("config.json"), json::to_vec_pretty(&self.config)?)?;
            std::fs::write(dir.join("tokenizer.json"), &self.tokenizer_json)?;
            save_tensors(
                e,
                &dir.join("model.safetensors"),
                self.parameters.iter().map(|p| (p.name.as_str(), &p.value)),
            )?;
            std::fs::write(
                dir.join("encoder.json"),
                json::to_vec_pretty(&self.identity)?,
            )?;
            Ok(())
        })
    }
    pub fn default_path() -> PathBuf {
        sejong_core::root().join(".cache/minilm")
    }
}
