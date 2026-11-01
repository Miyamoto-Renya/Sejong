use super::{allocation, query_work_bytes, work_bytes, workspace_sum};
use sejong_core::{DType, Result, json, native::BatchPolicy};
use sejong_gpu::{Engine, Tensor, graph::Graph};
use sejong_model::{Parameter, native::memory::Memory};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

enum Storage {
    Device(Tensor),
    File(QueryFile),
}

struct QueryFile {
    file: File,
    path: PathBuf,
    rows: Vec<f32>,
}
impl QueryFile {
    fn create(path: PathBuf) -> Result<Self> {
        let file = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            file,
            path,
            rows: Vec::new(),
        })
    }
    fn append(&mut self, values: &[f32]) -> Result<()> {
        // The run-local file preserves native FP32 bits between GPU visits.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        self.file.write_all(bytes)?;
        Ok(())
    }
    fn read(&mut self, first: usize, elements: usize) -> Result<&[f32]> {
        self.rows.resize(elements, 0.);
        self.file.seek(SeekFrom::Start(first as u64 * 4))?;
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self.rows.as_mut_ptr().cast::<u8>(), elements * 4)
        };
        self.file.read_exact(bytes)?;
        Ok(&self.rows)
    }
}
impl Drop for QueryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(super) struct QueryCache {
    storage: Storage,
    dimension: usize,
}
impl QueryCache {
    pub(super) fn prepare(
        e: &mut Engine,
        memory: &Memory,
        policy: &BatchPolicy,
        parameters: &[Parameter],
        fact_batch: usize,
        out: &Path,
    ) -> Result<Self> {
        let c = &memory.metadata.config;
        let n = memory.metadata.contexts.len();
        let d = c.dimension;
        let gradient_reserve = workspace_sum(
            parameters
                .iter()
                .map(|p| allocation(p.value.len() as u128, DType::F32).map(|size| (3, size)))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let available = e.available_memory_bytes()?.saturating_sub(gradient_reserve);
        let budget = policy.memory_bytes.map_or(available, |b| b.min(available));
        let query_bytes = query_work_bytes(memory)?;
        let fact_bytes = workspace_sum([(fact_batch as u128, work_bytes(memory, 0)?)])?;
        let elements = n
            .checked_mul(d)
            .ok_or_else(|| sejong_core::Error::Message("query cache dimensions overflow".into()))?;
        let cache_bytes = allocation(elements as u128, DType::F32)?;
        let resident = elements <= i32::MAX as usize
            && cache_bytes <= budget.saturating_sub(fact_bytes.max(query_bytes));
        let query_batch = policy.resolve(
            budget.saturating_sub(if resident { cache_bytes } else { 0 }),
            query_bytes,
            n,
        )?;
        let storage = if resident {
            Storage::Device(e.empty(&[n, d], DType::F32)?)
        } else {
            Storage::File(QueryFile::create(out.join("fact-query-cache.f32"))?)
        };
        eprintln!(
            "{}",
            json::json!({"memory_query_cache":{
                "storage":if resident { "device" } else { "file" },
                "rows":n,"bytes":elements as u128 * 4,
                "query_batch":query_batch,"fact_batch":fact_batch,
            }})
        );
        let mut cache = Self {
            storage,
            dimension: d,
        };
        for start in (0..n).step_by(query_batch) {
            let count = query_batch.min(n - start);
            let context: Vec<f32> = memory.metadata.contexts[start..start + count]
                .iter()
                .flatten()
                .map(|&x| x as f32)
                .collect();
            let input = e.from_f32(&[count * c.tokens, d], &context)?;
            let mut g = Graph::new(e, false);
            let w = memory.bind(&mut g, false, false);
            let q = memory.query(&mut g, &w, &input)?;
            let query = g.tensor(q).clone();
            drop(g);
            match &mut cache.storage {
                Storage::Device(tensor) => e.copy_into(&query, &tensor.rows(start, count)?)?,
                Storage::File(file) => file.append(&e.to_f32(&query)?)?,
            }
        }
        e.reclaim()?;
        Ok(cache)
    }

    pub(super) fn rows(&mut self, e: &Engine, start: usize, count: usize) -> Result<Tensor> {
        match &mut self.storage {
            Storage::Device(tensor) => tensor.rows(start, count),
            Storage::File(file) => e.from_f32(
                &[count, self.dimension],
                file.read(start * self.dimension, count * self.dimension)?,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QueryFile;

    #[test]
    fn streamed_queries_preserve_bits_and_release_file() {
        let path =
            std::env::temp_dir().join(format!("sejong-query-cache-{}.f32", std::process::id()));
        let values = [0., -0., 1.25, -9., f32::MIN_POSITIVE, f32::from_bits(1)];
        {
            let mut file = QueryFile::create(path.clone()).unwrap();
            file.append(&values[..2]).unwrap();
            file.append(&values[2..]).unwrap();
            for (start, count) in [(2, 4), (0, 2), (0, 6)] {
                let read = file.read(start, count).unwrap();
                assert_eq!(
                    read.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    values[start..start + count]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(!path.exists());
    }
}
