//! Answer-directed quotient: binary role/focus/allowed flags and five request classes.
use super::resources;
use sejong_core::{Result, ensure};
const ROLE_FLAGS: usize = 2 * 2;
const FOCUS_FLAGS: usize = 2;
const ALLOWED_FLAGS: usize = 2;
const REQUEST_CLASSES: usize = 5;
pub const CHANNEL_STATES: usize = ROLE_FLAGS * FOCUS_FLAGS * ALLOWED_FLAGS * REQUEST_CLASSES;
pub type State = Vec<f64>;
pub type LocalLaws = (Vec<f64>, Vec<f64>, Vec<f64>);
#[derive(Clone)]
pub struct Operator {
    people: usize,
    idle: f64,
    scene: Vec<[f64; ROLE_FLAGS]>,
    instruction: Vec<[f64; 6]>,
    request: Vec<[f64; REQUEST_CLASSES]>,
}
fn offset(channel: usize, roles: usize, focus: usize, allowed: usize, request: usize) -> usize {
    channel * CHANNEL_STATES
        + (((roles * FOCUS_FLAGS + focus) * ALLOWED_FLAGS + allowed) * REQUEST_CLASSES + request)
}
pub fn states(people: usize) -> Result<usize> {
    resources::product(&[2, people, CHANNEL_STATES])
}
pub fn operator_bytes(people: usize) -> Result<usize> {
    let data = resources::product(&[
        people,
        2,
        ROLE_FLAGS + 6 + REQUEST_CLASSES,
        std::mem::size_of::<f64>(),
    ])?;
    Ok(data.saturating_add(std::mem::size_of::<Operator>()))
}
pub fn memory_bytes(people: usize) -> Result<usize> {
    let state = resources::bytes::<f64>(states(people)?)?;
    state
        .checked_mul(2)
        .and_then(|n| n.checked_add(operator_bytes(people).ok()?))
        .ok_or_else(|| sejong_core::Error::Message("semantic storage size overflow".into()))
}

fn people_from_laws(laws: &LocalLaws) -> Result<usize> {
    let pairs = laws
        .0
        .len()
        .checked_sub(1)
        .ok_or_else(|| sejong_core::Error::Message("scene law requires NONE".into()))?;
    let mut n = (pairs as f64).sqrt() as usize;
    while n.checked_mul(n).is_some_and(|s| s < pairs) {
        n += 1;
    }
    ensure(
        n > 0 && n.checked_mul(n) == Some(pairs),
        "scene law shape must be N²+NONE",
    )?;
    Ok(n)
}
impl Operator {
    pub fn from_laws(laws: &LocalLaws) -> Result<Self> {
        let people = people_from_laws(laws)?;
        let scene_none = people * people;
        let instruction_none = (people + 1) * (people + 1) - 1;
        let selectors = people + 3;
        let request_none = 2 * selectors;
        let (s, i, q) = laws;
        ensure(
            i.len() == instruction_none + 1
                && q.len() == request_none + 1
                && resources::normalized(s)
                && resources::normalized(i)
                && resources::normalized(q),
            "normalized input-dependent local laws required",
        )?;
        let channels = 2 * people;
        let mut result = Self {
            people,
            idle: s[scene_none] * i[instruction_none] * q[request_none],
            scene: vec![[0.; ROLE_FLAGS]; channels],
            instruction: vec![[0.; 6]; channels],
            request: vec![[0.; REQUEST_CLASSES]; channels],
        };
        for channel in 0..channels {
            let goal = channel / people;
            let entity = channel % people;
            for pair in 0..scene_none {
                let bits =
                    usize::from(pair / people == entity) + 2 * usize::from(pair % people == entity);
                result.scene[channel][bits] += s[pair] * i[instruction_none] * q[request_none];
            }
            for action in 0..instruction_none {
                let focus = action / (people + 1);
                let excluded = action % (people + 1);
                let change = if focus == people {
                    0
                } else if focus == entity {
                    2
                } else {
                    1
                };
                result.instruction[channel][change * 2 + usize::from(excluded == entity)] +=
                    i[action] * s[scene_none] * q[request_none];
            }
            for request in 0..request_none {
                let selector = request % selectors;
                let relevant = if request / selectors != goal {
                    0
                } else if selector < people {
                    usize::from(selector == entity)
                } else {
                    selector - people + 2
                };
                result.request[channel][relevant] +=
                    q[request] * s[scene_none] * i[instruction_none];
            }
        }
        Ok(result)
    }
    pub fn people(&self) -> usize {
        self.people
    }
    pub fn arithmetic_operations(&self) -> usize {
        let transitions = 1 + ROLE_FLAGS + 6 + REQUEST_CLASSES;
        let apply = self
            .people
            .saturating_mul(2)
            .saturating_mul(CHANNEL_STATES)
            .saturating_mul(transitions)
            .saturating_mul(2);
        let law_terms = self
            .people
            .saturating_mul(self.people)
            .saturating_add((self.people + 1).saturating_mul(self.people + 1))
            .saturating_add(2 * (self.people + 3));
        apply.saturating_add(
            self.people
                .saturating_mul(2)
                .saturating_mul(law_terms)
                .saturating_mul(3),
        )
    }
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.people * 2 * (ROLE_FLAGS + 6 + REQUEST_CLASSES) * std::mem::size_of::<f64>()
    }
    pub fn apply(&self, state: &[f64]) -> Result<State> {
        ensure(
            state.len() == states(self.people)?,
            "semantic quotient state shape differs",
        )?;
        let mut next = vec![0.; state.len()];
        for channel in 0..2 * self.people {
            for roles in 0..ROLE_FLAGS {
                for focus in 0..FOCUS_FLAGS {
                    for allowed in 0..ALLOWED_FLAGS {
                        for request in 0..REQUEST_CLASSES {
                            let current = offset(channel, roles, focus, allowed, request);
                            let mass = state[current];
                            if mass == 0. {
                                continue;
                            }
                            next[current] += mass * self.idle;
                            for changed in 0..ROLE_FLAGS {
                                next[offset(channel, changed, focus, allowed, request)] +=
                                    mass * self.scene[channel][changed];
                            }
                            for action in 0..6 {
                                let set_focus = action / 2;
                                let exclude = action % 2;
                                next[offset(
                                    channel,
                                    roles,
                                    if set_focus == 0 { focus } else { set_focus - 1 },
                                    if exclude == 0 { allowed } else { 0 },
                                    request,
                                )] += mass * self.instruction[channel][action];
                            }
                            for changed in 0..REQUEST_CLASSES {
                                next[offset(channel, roles, focus, allowed, changed)] +=
                                    mass * self.request[channel][changed];
                            }
                        }
                    }
                }
            }
        }
        Ok(next)
    }
}
pub fn initial(people: usize, limit_bytes: usize) -> Result<State> {
    ensure(people > 0, "semantic input requires an identity")?;
    resources::require(limit_bytes, memory_bytes(people)?, "semantic quotient")?;
    let mut state = resources::vector(states(people)?, 0., limit_bytes, "semantic quotient")?;
    for channel in 0..2 * people {
        state[offset(channel, 0, 0, 1, 0)] = 1.;
    }
    Ok(state)
}
pub fn readout(state: &[f64]) -> Result<Vec<f64>> {
    Ok(readout_bounded(state, 0)?.0)
}
pub fn readout_bounded(state: &[f64], operations: usize) -> Result<(Vec<f64>, f64)> {
    ensure(
        state.len() > 0 && state.len().is_multiple_of(2 * CHANNEL_STATES),
        "semantic quotient shape differs",
    )?;
    let people = state.len() / (2 * CHANNEL_STATES);
    let channels = 2 * people;
    let mut result = vec![0.; channels + 1];
    for channel in 0..channels {
        for roles in 0..ROLE_FLAGS {
            for focus in 0..FOCUS_FLAGS {
                result[channel] += state[offset(channel, roles, focus, 1, 1)];
                if roles & 1 != 0 {
                    result[channel] += state[offset(channel, roles, focus, 1, 2)];
                }
                if roles & 2 != 0 {
                    result[channel] += state[offset(channel, roles, focus, 1, 3)];
                }
                if focus == 1 {
                    result[channel] += state[offset(channel, roles, focus, 1, 4)];
                }
            }
        }
    }
    let resolved = result[..channels].iter().sum::<f64>();
    let rounding = sejong_core::numeric::sum_error(
        state.iter().map(|p| p.abs()).sum::<f64>() + result.iter().map(|p| p.abs()).sum::<f64>(),
        operations
            .saturating_add(state.len())
            .saturating_add(result.len()),
    );
    ensure(
        result.iter().all(|p| p.is_finite() && *p >= 0.) && resolved <= 1. + rounding,
        "semantic quotient mass drift",
    )?;
    result[channels] = (1. - resolved).max(0.);
    Ok((result, rounding))
}
pub fn execute(rows: &[LocalLaws], people: usize, limit_bytes: usize) -> Result<Vec<f64>> {
    let mut state = initial(people, limit_bytes)?;
    let mut operations = 0usize;
    for row in rows {
        let op = Operator::from_laws(row)?;
        ensure(op.people == people, "clause identity coordinates differ")?;
        operations = operations.saturating_add(op.arithmetic_operations());
        state = op.apply(&state)?;
    }
    Ok(readout_bounded(&state, operations)?.0)
}
pub fn replace_identity(text: &str, identity: &str, replacement: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0;
    for (start, _) in text.match_indices(identity) {
        let end = start + identity.len();
        let raw = text.as_bytes();
        if (start == 0 || !raw[start - 1].is_ascii_alphabetic())
            && (end == raw.len() || !raw[end].is_ascii_alphabetic())
        {
            out.push_str(&text[cursor..start]);
            out.push_str(replacement);
            cursor = end;
        }
    }
    out.push_str(&text[cursor..]);
    out
}
