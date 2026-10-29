//! Conservative imperative meanings with a bounded live-variable frontier.
use super::{
    resources,
    semantic::{self, Operator, State},
};
use sejong_core::{Result, ensure, native::Budgets};
use sejong_gpu::Engine;
use sejong_model::native::classifier::Classifier;
use std::collections::BTreeMap;

pub struct Clause {
    pub latent: Option<usize>,
    pub operators: Vec<Operator>,
}
fn recognize(
    text: &str,
    refs: &[String],
    registry: &BTreeMap<String, usize>,
) -> Option<(usize, String, usize)> {
    let raw = text.as_bytes();
    let mut hits = vec![];
    for (entity, name) in refs.iter().enumerate() {
        for (start, _) in text.match_indices(name) {
            let end = start + name.len();
            if (start == 0 || !raw[start - 1].is_ascii_alphabetic())
                && (end == raw.len() || !raw[end].is_ascii_alphabetic())
            {
                hits.push((start, end, entity));
            }
        }
    }
    if hits.len() != 1 {
        return None;
    }
    let (start, end, entity) = hits[0];
    let phrase = text[..start].trim().to_ascii_lowercase();
    let suffix = text[end..].trim();
    if !["", ".", ";"].contains(&suffix) {
        return None;
    }
    registry.get(&phrase).map(|owner| (*owner, phrase, entity))
}
fn operator(
    e: &mut Engine,
    text: &str,
    refs: &[String],
    scene: &Classifier,
    instruction: &Classifier,
    request: &Classifier,
    cache: &mut BTreeMap<String, Operator>,
) -> Result<Operator> {
    if let Some(op) = cache.get(text) {
        return Ok(op.clone());
    }
    let op = Operator::from_laws(&(
        scene.law(e, text, refs, &[])?,
        instruction.law(e, text, refs, &[])?,
        request.law(e, text, refs, &[])?,
    ))?;
    cache.insert(text.into(), op.clone());
    Ok(op)
}
pub fn bind(
    e: &mut Engine,
    text: &str,
    refs: &[String],
    scene: &Classifier,
    instruction: &Classifier,
    request: &Classifier,
    lexica: &[Classifier],
    budgets: &Budgets,
) -> Result<(Vec<f64>, f64)> {
    let limit = budgets.resolve()?;
    let mut registry = BTreeMap::new();
    for (owner, model) in lexica.iter().enumerate() {
        ensure(
            model
                .metadata
                .meanings
                .iter()
                .map(|s| s.to_ascii_lowercase())
                .collect::<Vec<_>>()
                == ["use", "ignore"],
            "lexical meanings must follow the Use/Ignore ABI",
        )?;
        for phrase in &model.metadata.phrases {
            ensure(
                registry
                    .insert(phrase.to_ascii_lowercase(), owner)
                    .is_none(),
                "lexical namespaces must be disjoint",
            )?;
        }
    }
    let mut latent_ids = BTreeMap::new();
    let mut priors = vec![];
    let mut clauses = vec![];
    let mut cache = BTreeMap::new();
    let mut compiled = 0usize;
    let operator_bytes = semantic::operator_bytes(refs.len())?;
    let laws_entries = [
        sejong_core::native::Role::Scene,
        sejong_core::native::Role::Instruction,
        sejong_core::native::Role::Request,
    ]
    .into_iter()
    .try_fold(0usize, |n, role| -> Result<usize> {
        Ok(n.saturating_add(role.classes_for(refs.len(), 0)?))
    })?;
    let laws_bytes = resources::bytes::<f64>(laws_entries)?;
    for part in text
        .split_inclusive(['.', ';', '?'])
        .filter(|s| s.chars().any(|c| c.is_alphanumeric()))
    {
        let recognized = recognize(part, refs, &registry);
        let meanings = if recognized.is_some() { 2 } else { 1 };
        resources::require(
            limit,
            compiled
                .saturating_add(resources::product(&[meanings, 2, operator_bytes])?)
                .saturating_add(laws_bytes),
            "semantic clause compilation workspace",
        )?;
        if let Some((owner, phrase, entity)) = recognized {
            let key = (owner, phrase.clone());
            let latent = if let Some(id) = latent_ids.get(&key) {
                *id
            } else {
                let p = lexica[owner].law(e, &phrase, &[], &[])?;
                ensure(
                    p.len() == 2 && resources::normalized(&p),
                    "normalized Use/Ignore lexical law required",
                )?;
                let id = priors.len();
                priors.push(p);
                latent_ids.insert(key, id);
                id
            };
            let operators = ["Use", "Ignore"]
                .iter()
                .map(|meaning| {
                    operator(
                        e,
                        &format!("{meaning} {}.", refs[entity]),
                        refs,
                        scene,
                        instruction,
                        request,
                        &mut cache,
                    )
                })
                .collect::<Result<_>>()?;
            clauses.push(Clause {
                latent: Some(latent),
                operators,
            });
        } else {
            clauses.push(Clause {
                latent: None,
                operators: vec![operator(
                    e,
                    part,
                    refs,
                    scene,
                    instruction,
                    request,
                    &mut cache,
                )?],
            });
        }
        let used = clauses
            .iter()
            .map(|c: &Clause| {
                c.operators
                    .iter()
                    .map(Operator::memory_bytes)
                    .sum::<usize>()
            })
            .sum::<usize>()
            + cache.values().map(Operator::memory_bytes).sum::<usize>();
        compiled = used;
        resources::require(limit, used, "compiled lexical/semantic clauses")?;
    }
    ensure(!clauses.is_empty(), "query requires at least one clause")?;
    drop(cache);
    drop(registry);
    drop(latent_ids);
    execute_bounded(&clauses, &priors, refs.len(), budgets)
}
/// Executes already compiled conditional operators with bounded live choices.
pub fn execute(
    clauses: &[Clause],
    priors: &[Vec<f64>],
    people: usize,
    budgets: &Budgets,
) -> Result<Vec<f64>> {
    Ok(execute_bounded(clauses, priors, people, budgets)?.0)
}
fn execute_bounded(
    clauses: &[Clause],
    priors: &[Vec<f64>],
    people: usize,
    budgets: &Budgets,
) -> Result<(Vec<f64>, f64)> {
    for clause in clauses {
        ensure(
            clause.latent.map_or(clause.operators.len() == 1, |id| {
                id < priors.len() && clause.operators.len() == priors[id].len()
            }),
            "conditional lexical operator arity mismatch",
        )?;
    }
    for prior in priors {
        ensure(
            !prior.is_empty() && resources::normalized(prior),
            "normalized lexical prior required",
        )?;
    }
    let limit = budgets.resolve()?;
    let states = semantic::states(people)?;
    let state_bytes = resources::bytes::<f64>(states)?;
    let base = clauses
        .iter()
        .map(|c| {
            c.operators
                .iter()
                .map(Operator::memory_bytes)
                .sum::<usize>()
        })
        .sum::<usize>()
        + priors
            .iter()
            .map(|p| p.len() * std::mem::size_of::<f64>())
            .sum::<usize>();
    resources::require(
        limit,
        base.saturating_add(state_bytes.saturating_mul(2)),
        "lexical base state storage",
    )?;
    let mut last = BTreeMap::new();
    for (position, clause) in clauses.iter().enumerate() {
        if let Some(key) = clause.latent {
            last.insert(key, position);
        }
    }
    let mut branches = BTreeMap::from([(
        Vec::<usize>::new(),
        semantic::initial(people, limit.saturating_sub(base))?,
    )]);
    let mut active = vec![];
    let mut operations = 0usize;
    for (position, clause) in clauses.iter().enumerate() {
        if let Some(key) = clause.latent {
            if !active.contains(&key) {
                let positive: Vec<_> = priors[key]
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 0.)
                    .collect();
                let count = branches.len().checked_mul(positive.len()).ok_or_else(|| {
                    sejong_core::Error::Message("live lexical shape overflow".into())
                })?;
                let assignment_bytes = resources::bytes::<usize>(active.len() + 1)?;
                let per_state = state_bytes
                    .saturating_add(assignment_bytes)
                    .saturating_add(std::mem::size_of::<(Vec<usize>, State)>());
                let requested = base.saturating_add(
                    count
                        .saturating_add(branches.len())
                        .saturating_mul(per_state),
                );
                resources::require(limit, requested, "live lexical frontier")?;
                operations = operations.saturating_add(count.saturating_mul(states));
                let mut expanded = BTreeMap::new();
                for (assignment, state) in branches {
                    for &(choice, p) in &positive {
                        let mut assignment = assignment.clone();
                        assignment.push(choice);
                        expanded.insert(assignment, state.iter().map(|v| v * p).collect::<State>());
                    }
                }
                branches = expanded;
                active.push(key);
            }
        }
        let axis = clause
            .latent
            .and_then(|key| active.iter().position(|k| *k == key));
        resources::require(
            limit,
            base.saturating_add(branches.len().saturating_add(1).saturating_mul(
                state_bytes.saturating_add(resources::bytes::<usize>(active.len())?),
            )),
            "lexical operator workspace",
        )?;
        for (assignment, state) in branches.iter_mut() {
            let selected = axis.map_or(0, |axis| assignment[axis]);
            operations =
                operations.saturating_add(clause.operators[selected].arithmetic_operations());
            *state = clause.operators[selected].apply(state)?;
        }
        if let Some(key) = clause.latent {
            if last[&key] == position {
                let axis = active.iter().position(|id| *id == key).unwrap();
                operations = operations.saturating_add(branches.len().saturating_mul(states));
                let mut merged: BTreeMap<Vec<usize>, State> = BTreeMap::new();
                for (mut assignment, state) in branches {
                    assignment.remove(axis);
                    let destination = merged.entry(assignment).or_insert_with(|| vec![0.; states]);
                    for (to, from) in destination.iter_mut().zip(state) {
                        *to += from;
                    }
                }
                branches = merged;
                active.remove(axis);
            }
        }
    }
    ensure(
        active.is_empty() && branches.len() == 1,
        "lexical frontier finalization failed",
    )?;
    semantic::readout_bounded(branches.values().next().unwrap(), operations)
}
