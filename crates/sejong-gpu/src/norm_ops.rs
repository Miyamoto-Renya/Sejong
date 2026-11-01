use crate::parameter_vjp::{BK, BM, BN};
use crate::{Engine, Tensor, hip::EventPair, n, p};
use sejong_core::{DType, Result, json};

#[derive(Clone, Copy)]
pub(crate) struct RmsPlan {
    br: usize,
    bg: usize,
    bd: usize,
    groups: usize,
}

impl Engine {
    pub fn norm_backward(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        dy: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor)> {
        let key = (x.shape[0], x.shape[1]);
        let dx = self.empty(&x.shape, DType::F32)?;
        let dw = self.empty(&w.shape, DType::F32)?;
        let plan = if let Some(plan) = self.rms_plans.get(&key) {
            *plan
        } else {
            let plan = self.prepare_rms_plan(x, w, dy, &dx, &dw, epsilon)?;
            self.rms_plans.insert(key, plan);
            plan
        };
        let partial = self.empty(&[plan.groups, key.1], DType::F32)?;
        self.rms_backward_pair(x, w, dy, &dx, &dw, &partial, epsilon, plan)?;
        Ok((dx, dw))
    }

    fn rms_backward_pair(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        dy: &Tensor,
        dx: &Tensor,
        dw: &Tensor,
        partial: &Tensor,
        epsilon: f32,
        plan: RmsPlan,
    ) -> Result<()> {
        let d = x.shape[1];
        self.launch(
            "rmsnorm_backward_partial",
            &[
                ("X", p(x)),
                ("W", p(w)),
                ("DY", p(dy)),
                ("DX", p(dx)),
                ("Partial", p(partial)),
                ("R", n(x.shape[0])?),
            ],
            json::json!({"D":d,"EPS":epsilon,"BR":plan.br,"BD":plan.bd}),
            [plan.groups, 1, 1],
        )?;
        self.launch(
            "rmsnorm_backward_reduce",
            &[
                ("Partial", p(partial)),
                ("DW", p(dw)),
                ("G", n(plan.groups)?),
            ],
            json::json!({"D":d,"BG":plan.bg,"BD":BN}),
            [d.div_ceil(BN), 1, 1],
        )
    }

    fn prepare_rms_plan(
        &mut self,
        x: &Tensor,
        w: &Tensor,
        dy: &Tensor,
        dx: &Tensor,
        dw: &Tensor,
        epsilon: f32,
    ) -> Result<RmsPlan> {
        let (rows, d) = (x.shape[0], x.shape[1]);
        let bd = d.next_power_of_two();
        // x·dy·d·dx의 row tile과 장치 register 예산으로 넓은 형상의 BR을 줄인다.
        let row_capacity = (self.device.registers_per_block / (4 * bd)).max(1);
        let row_limit = 1usize << row_capacity.ilog2();
        let mut plans = Vec::with_capacity(3);
        for requested in [BM / 4, BM / 2, BM] {
            let br = requested.min(row_limit).min(rows.next_power_of_two());
            if plans.last().is_some_and(|p: &RmsPlan| p.br == br) {
                continue;
            }
            let groups = rows.div_ceil(br);
            plans.push(RmsPlan {
                br,
                bg: groups.next_power_of_two().min(2 * BK),
                bd,
                groups,
            });
        }
        let partial = self.empty(&[plans[0].groups, d], DType::F32)?;
        for &plan in &plans {
            self.rms_backward_pair(x, w, dy, dx, dw, &partial, epsilon, plan)?;
        }
        self.synchronize()?;
        if plans.len() == 1 {
            return Ok(plans[0]);
        }
        // 연구의 BR4/8/16 후보를 최초 형상에서 네 번 왕복 측정한다.
        // 각 표본은 기존 BM개의 pair를 묶고 컴파일·할당은 앞에서 끝낸다.
        let mut samples = vec![Vec::<f64>::with_capacity(8); plans.len()];
        for _ in 0..4 {
            for i in (0..plans.len()).chain((0..plans.len()).rev()) {
                let timing = EventPair::start(self.device.clone())?;
                for _ in 0..BM {
                    self.rms_backward_pair(x, w, dy, dx, dw, &partial, epsilon, plans[i])?;
                }
                timing.end()?;
                self.synchronize()?;
                samples[i].push(timing.seconds()?);
            }
        }
        let mut selected = 0;
        let mut best = f64::INFINITY;
        for (i, values) in samples.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            let median = (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.;
            if median < best {
                best = median;
                selected = i;
            }
        }
        Ok(plans[selected])
    }
}
