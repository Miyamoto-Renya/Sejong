use crate::{Arg, Engine};
use sejong_core::json;

/// 실행한 GEMM과 attention 내적의 산술량 및 커널 준비 시간을 누적한다.
#[derive(Clone, Debug, Default)]
pub struct ExecutionCounters {
    pub launches: u64,
    pub logical_flops: u64,
    pub issued_flops: u64,
    pub gemm_launches: u64,
    pub attention_launches: u64,
    pub compilations: u64,
    pub compile_seconds: f64,
}

impl ExecutionCounters {
    pub fn difference(&self, previous: &Self) -> Self {
        Self {
            launches: self.launches.saturating_sub(previous.launches),
            logical_flops: self.logical_flops.saturating_sub(previous.logical_flops),
            issued_flops: self.issued_flops.saturating_sub(previous.issued_flops),
            gemm_launches: self.gemm_launches.saturating_sub(previous.gemm_launches),
            attention_launches: self
                .attention_launches
                .saturating_sub(previous.attention_launches),
            compilations: self.compilations.saturating_sub(previous.compilations),
            compile_seconds: (self.compile_seconds - previous.compile_seconds).max(0.),
        }
    }
}

impl Engine {
    pub fn execution_counters(&self) -> ExecutionCounters {
        self.counters.clone()
    }
    pub fn reset_execution_counters(&mut self) {
        self.counters = ExecutionCounters::default();
    }

    pub(crate) fn count_execution(
        &mut self,
        operation: &str,
        args: &[(&str, Arg)],
        c: &json::Value,
    ) {
        self.counters.launches += 1;
        let dimension = |name: &str| -> u64 {
            args.iter()
                .find_map(|(key, value)| match value {
                    Arg::I32(value) if *key == name => Some(*value as u64),
                    _ => None,
                })
                .unwrap_or(0)
        };
        let constant = |name: &str| c[name].as_u64().unwrap_or(0);
        let round = |value: u64, tile: u64| value.div_ceil(tile.max(1)) * tile.max(1);
        let (logical, issued) = if operation == "matmul" || operation == "matmul64" {
            self.counters.gemm_launches += 1;
            let (m, n, k) = (dimension("M"), dimension("N"), dimension("K"));
            (
                2 * m * n * k,
                if operation == "matmul64" {
                    2 * m * round(n, constant("BLOCK")) * k
                } else {
                    2 * round(m, constant("BM"))
                        * round(n, constant("BN"))
                        * round(k, constant("BK"))
                },
            )
        } else if operation == "hierarchy_coefficient_transport" {
            self.counters.gemm_launches += 1;
            let rank = dimension("R");
            let features = args
                .iter()
                .find_map(|(name, arg)| match arg {
                    Arg::Ptr(t) if *name == "H" => Some(t.shape[0] as u64),
                    _ => None,
                })
                .unwrap_or(0);
            (
                2 * features * rank * rank * rank,
                2 * features
                    * round(rank, constant("BM"))
                    * round(rank, constant("BN"))
                    * round(rank, constant("BK")),
            )
        } else if operation.contains("attention") {
            self.counters.attention_launches += 1;
            let grouped = operation.starts_with("hierarchy_");
            let forward = !operation.ends_with("_dq") && !operation.ends_with("_dkv");
            let tq = if forward && !operation.starts_with("parser_") {
                dimension("TQ")
            } else {
                dimension("T")
            };
            let tk = if forward && !operation.starts_with("parser_") {
                dimension("TK")
            } else {
                tq
            };
            let offset = dimension("OFFSET");
            let heads = constant(if grouped { "HQ" } else { "H" });
            let width = constant("D");
            let issued_width = constant("BD").max(width);
            let tile = constant("BK").max(1);
            let window = constant("WINDOW");
            let causal = grouped || c["CAUSAL"].as_bool().unwrap_or(false);
            let rows = args
                .iter()
                .find_map(|(name, arg)| match arg {
                    Arg::Ptr(t) if *name == "Q" => Some(t.shape[0] as u64),
                    _ => None,
                })
                .unwrap_or(0);
            let batch = rows / tq.max(1);
            let mut pairs = 0;
            let mut issued_pairs = 0;
            for position in 0..tq {
                let valid = if causal {
                    (position + offset + 1).min(tk)
                } else {
                    tk
                };
                let valid = if window > 0 { valid.min(window) } else { valid };
                pairs += valid;
                if operation.ends_with("_dkv") {
                    let future = if causal {
                        (tq - position).min(if window > 0 { window } else { tq })
                    } else {
                        tq
                    };
                    issued_pairs += if grouped {
                        round(future, tile)
                    } else if causal {
                        round(tq, tile) - (position / tile) * tile
                    } else {
                        round(tq, tile)
                    };
                } else {
                    issued_pairs += round(valid, tile);
                }
            }
            let factor = if forward {
                4
            } else if operation.ends_with("_dq") {
                6
            } else {
                10
            };
            let delta = if operation.ends_with("_dq") {
                2 * rows * heads * width
            } else {
                0
            };
            let issued_delta = if operation.ends_with("_dq") {
                2 * rows * heads * issued_width
            } else {
                0
            };
            (
                factor * batch * heads * width * pairs + delta,
                factor * batch * heads * issued_width * issued_pairs + issued_delta,
            )
        } else {
            (0, 0)
        };
        self.counters.logical_flops = self.counters.logical_flops.saturating_add(logical);
        self.counters.issued_flops = self.counters.issued_flops.saturating_add(issued);
    }
}
