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
    /// 명시한 진단 실행의 HIP event 시간을 연산별로 합산한다.
    #[cfg(feature = "kernel-profile")]
    pub fn kernel_profile(&self) -> sejong_core::Result<Option<json::Value>> {
        let Some(timings) = &self.kernel_timings else {
            return Ok(None);
        };
        self.synchronize()?;
        let mut groups = std::collections::BTreeMap::<String, (u64, f64)>::new();
        for (operation, events) in timings {
            let group = groups.entry(operation.clone()).or_default();
            group.0 += 1;
            group.1 += events.seconds()?;
        }
        Ok(Some(
            json::json!({"scope":"HIP event durations in an explicitly instrumented run",
            "operations":groups.into_iter().map(|(operation,(launches,seconds))|
                json::json!({"operation":operation,"launches":launches,"gpu_seconds":seconds}))
                .collect::<Vec<_>>()}),
        ))
    }
    #[cfg(not(feature = "kernel-profile"))]
    pub fn kernel_profile(&self) -> sejong_core::Result<Option<json::Value>> {
        Ok(None)
    }
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
        } else if operation == "projection_qkv" || operation == "projection_gate_up" {
            self.counters.gemm_launches += 1;
            let (m, n, k) = (dimension("M"), dimension("N"), dimension("K"));
            let (width, issued_width) = if operation == "projection_qkv" {
                let kv = dimension("KV");
                (
                    n + 2 * kv,
                    round(n, constant("BN")) + 2 * round(kv, constant("BN")),
                )
            } else {
                (2 * n, 2 * round(n, constant("BN")))
            };
            (
                2 * m * k * width,
                2 * round(m, constant("BM")) * round(k, constant("BK")) * issued_width,
            )
        } else if operation == "parameter_prepare" {
            self.counters.gemm_launches += 1;
            let (a, n, k, r) = (
                dimension("M"),
                dimension("N"),
                dimension("K"),
                dimension("R"),
            );
            (
                2 * (a + k) * n * r,
                2 * round(a, constant("BM")) * round(r, constant("BN")) * round(n, constant("BK"))
                    + 2 * round(k, constant("BM"))
                        * round(n, constant("BN"))
                        * round(r, constant("BK")),
            )
        } else if operation == "parameter_residual" {
            self.counters.gemm_launches += 1;
            let (m, n, k) = (dimension("K"), dimension("N"), dimension("R"));
            (
                2 * m * n * k,
                2 * round(m, constant("BM")) * round(n, constant("BN")) * round(k, constant("BK")),
            )
        } else if operation == "parameter_basis_register" {
            let rows = dimension("N");
            let rank = constant("R");
            (
                4 * rows * rank * rank.saturating_sub(1) + 5 * rows * rank,
                8 * constant("NR") * constant("NC") * rank + 5 * constant("NR") * rank,
            )
        } else if operation == "parameter_basis" {
            let rows = dimension("N");
            let rank = constant("R");
            let last = rank.saturating_sub(1);
            let width = constant("BC").max(1);
            let q = last / width;
            let r = last % width;
            let groups = width * q * (q + 1) / 2 + r * (q + 1);
            let padded = round(rows, constant("BR"));
            (
                4 * rows * rank * rank.saturating_sub(1) + 2 * rows * groups + 5 * rows * rank,
                2 * (4 * padded * constant("BC") + padded) * groups + 5 * padded * rank,
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
            // prefix(x)=sum(round(v,tile), v=1..x).
            let prefix = |x: u64| {
                let q = x / tile;
                let r = x % tile;
                tile * tile * q * (q + 1) / 2 + tile * r * (q + 1)
            };
            let capped = |count: u64, start: u64, cap: u64| {
                let active = count.min(cap.saturating_sub(start));
                let upper = start + active;
                let sum = (upper * (upper + 1) - start * (start + 1)) / 2 + (count - active) * cap;
                let issued = prefix(upper) - prefix(start) + (count - active) * round(cap, tile);
                (sum, issued)
            };
            let cap = if window > 0 { tk.min(window) } else { tk };
            let (pairs, forward_issued) = if causal {
                capped(tq, offset, cap)
            } else {
                (tq * cap, tq * round(cap, tile))
            };
            let issued_pairs = if operation.ends_with("_dkv") {
                if grouped {
                    capped(tq, 0, if window > 0 { tq.min(window) } else { tq }).1
                } else if causal {
                    let q = tq / tile;
                    let r = tq % tile;
                    let floors = tile * q * q.saturating_sub(1) / 2 + q * r;
                    tq * round(tq, tile) - tile * floors
                } else {
                    tq * round(tq, tile)
                }
            } else {
                forward_issued
            };
            let factor = if forward {
                4
            } else if operation.ends_with("_dq") {
                6
            } else if grouped {
                8
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
