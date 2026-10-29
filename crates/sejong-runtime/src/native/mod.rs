pub mod events;
pub mod lexical;
pub mod partition;
mod resources;
pub mod semantic;
use events::Arena;
use sejong_core::{Result, Rng, ensure, json, native::*};
use sejong_gpu::Engine;
use sejong_model::native::{
    atomic_directory,
    classifier::{Classifier, QueryPlan},
    memory::Memory,
    signature,
    value::NativeValue,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub enum Fact {
    Memory(Memory),
    Native(NativeValue),
}
impl Fact {
    pub fn requests(&self) -> &[ByteValue] {
        match self {
            Self::Memory(m) => &m.metadata.requests,
            Self::Native(m) => m.requests(),
        }
    }
    pub fn values(&self) -> &[ByteValue] {
        match self {
            Self::Memory(m) => &m.metadata.values,
            Self::Native(m) => m.values(),
        }
    }
    pub fn default_mode(&self) -> ReadMode {
        if matches!(self, Self::Memory(_)) {
            ReadMode::Direct
        } else {
            ReadMode::Full
        }
    }
    pub fn verify(&self, e: &mut Engine) -> Result<()> {
        match self {
            Self::Memory(m) => m.verify(e),
            Self::Native(m) => m.verify(e),
        }
    }
    pub fn signature(&self, e: &mut Engine) -> Result<String> {
        match self {
            Self::Memory(m) => m.signature(e),
            Self::Native(m) => m.signature(e),
        }
    }
    fn load(e: &mut Engine, source: &FactSource, base: &Path) -> Result<Self> {
        match source {
            FactSource::Memory { path } => Ok(Self::Memory(Memory::load(e, &base.join(path))?)),
            FactSource::Native { path } => {
                Ok(Self::Native(NativeValue::load(e, &base.join(path))?))
            }
        }
    }
    fn save(&self, e: &mut Engine, root: &Path, name: &str) -> Result<FactSource> {
        let path = PathBuf::from(name);
        match self {
            Self::Memory(m) => {
                m.save(e, &root.join(&path))?;
                Ok(FactSource::Memory { path })
            }
            Self::Native(m) => {
                m.save(e, &root.join(&path))?;
                Ok(FactSource::Native { path })
            }
        }
    }
    pub fn row(&self, e: &mut Engine, index: usize, mode: ReadMode) -> Result<(Vec<f64>, f64)> {
        ensure(
            index < self.requests().len(),
            "request outside factual inventory",
        )?;
        match self {
            Self::Memory(m) => m.row(e, index, mode),
            Self::Native(m) => {
                ensure(
                    mode == ReadMode::Full,
                    "native value donor requires full mode",
                )?;
                Ok((m.row(e, index)?, 0.))
            }
        }
    }
}
#[derive(Serialize, Deserialize)]
struct Bundle {
    format: String,
    assembly: Assembly,
    signatures: BTreeMap<String, String>,
}
pub struct Composite {
    pub scene: Classifier,
    pub instruction: Classifier,
    pub request: Classifier,
    pub query_model: Classifier,
    pub lexica: Vec<Classifier>,
    pub facts: Vec<Fact>,
    pub names: Vec<String>,
    pub goal_routes: Vec<Vec<usize>>,
    pub catalog: Vec<String>,
    pub budgets: Budgets,
    signatures: BTreeMap<String, String>,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Task {
    Flat,
    Prefix(Program),
    Fact { goal: usize, hop: usize, row: usize },
}
#[derive(Serialize, Deserialize)]
pub struct Sample {
    pub output: Option<ByteValue>,
    pub terminal: usize,
    pub read_error: f64,
    pub probabilities: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
pub struct WeightResult {
    pub output: Vec<Vec<f64>>,
    pub route: Vec<String>,
    pub modes: Vec<ReadMode>,
    pub connection_scalars: usize,
}
#[derive(Serialize, Deserialize)]
pub struct WeightComparison {
    pub selected: WeightResult,
    pub full: WeightResult,
    pub max_absolute_error: f64,
}
#[derive(Serialize, Deserialize)]
pub struct Accounting {
    pub original_parameters: usize,
    pub connection_scalars: usize,
    pub context_scalars: usize,
    pub readout_scalars: usize,
    pub bound_scalars: usize,
    pub prototype_scalars: usize,
}
impl Composite {
    pub fn assemble(e: &mut Engine, spec: &Assembly, base: &Path) -> Result<Self> {
        inventory(&spec.catalog, "catalog")?;
        ensure(
            spec.catalog
                .iter()
                .all(|s| s.bytes().all(|b| b.is_ascii_lowercase())),
            "catalog identities use lowercase ASCII letters",
        )?;
        spec.budgets.resolve()?;
        let scene = Classifier::load(e, &base.join(&spec.scene))?;
        let instruction = Classifier::load(e, &base.join(&spec.instruction))?;
        let request = Classifier::load(e, &base.join(&spec.request))?;
        let query_model = Classifier::load(e, &base.join(&spec.query))?;
        ensure(
            scene.metadata.role == Role::Scene
                && instruction.metadata.role == Role::Instruction
                && request.metadata.role == Role::Request
                && query_model.metadata.role == Role::Query,
            "matching local original roles required",
        )?;
        let lexica = spec
            .lexica
            .iter()
            .map(|p| Classifier::load(e, &base.join(p)))
            .collect::<Result<Vec<_>>>()?;
        ensure(
            lexica.iter().all(|l| {
                l.metadata.role == Role::Lexical
                    && l.metadata
                        .meanings
                        .iter()
                        .map(|s| s.to_ascii_lowercase())
                        .collect::<Vec<_>>()
                        == ["use", "ignore"]
            }),
            "lexical original role required",
        )?;
        let mut phrases = BTreeSet::new();
        for lex in &lexica {
            for phrase in &lex.metadata.phrases {
                ensure(
                    phrases.insert(phrase.to_ascii_lowercase()),
                    "lexical namespaces must be disjoint",
                )?;
            }
        }
        ensure(
            !spec.extra.contains_key("a") && !spec.extra.contains_key("b"),
            "extra original IDs must preserve primary A/B names",
        )?;
        let mut names = vec!["a".into(), "b".into()];
        let mut facts = vec![Fact::load(e, &spec.a, base)?, Fact::load(e, &spec.b, base)?];
        for (name, source) in &spec.extra {
            ensure(!name.is_empty(), "original ID must be nonempty")?;
            names.push(name.clone());
            facts.push(Fact::load(e, source, base)?);
        }
        let declarations = if spec.goal_routes.is_empty() {
            vec![
                vec!["a".to_string()],
                vec!["a".to_string(), "b".to_string()],
            ]
        } else {
            spec.goal_routes.clone()
        };
        ensure(
            declarations.len() == 2,
            "semantic goal labels require two declared output routes",
        )?;
        let mut goal_routes = vec![];
        for route in declarations {
            ensure(!route.is_empty(), "goal route requires an original")?;
            let mut indices = vec![];
            for name in route {
                let index = names.iter().position(|n| n == &name).ok_or_else(|| {
                    sejong_core::Error::Message(format!("unknown goal-route original {name}"))
                })?;
                ensure(
                    !indices.contains(&index),
                    "probabilistic goal routes require distinct owner scopes; repeated owner correlations use a shared-world program",
                )?;
                indices.push(index);
            }
            ensure(
                spec.catalog.iter().all(|p| {
                    facts[indices[0]]
                        .requests()
                        .contains(&ByteValue::from(p.clone()))
                }),
                "first goal original must cover the catalog",
            )?;
            for owners in indices.windows(2) {
                ensure(
                    facts[owners[0]]
                        .values()
                        .iter()
                        .all(|v| facts[owners[1]].requests().contains(v)),
                    "downstream original must cover the preceding value inventory",
                )?;
            }
            goal_routes.push(indices);
        }
        let mut model = Self {
            scene,
            instruction,
            request,
            query_model,
            lexica,
            facts,
            names,
            goal_routes,
            catalog: spec.catalog.clone(),
            budgets: spec.budgets.clone(),
            signatures: BTreeMap::new(),
        };
        model.seal_snapshot(e)?;
        Ok(model)
    }
    fn current_signatures(&self, e: &mut Engine) -> Result<BTreeMap<String, String>> {
        let mut out = BTreeMap::new();
        for (name, model) in [
            ("scene", &self.scene),
            ("instruction", &self.instruction),
            ("request", &self.request),
            ("query", &self.query_model),
        ] {
            out.insert(name.into(), model.signature(e)?);
        }
        for (i, lex) in self.lexica.iter().enumerate() {
            out.insert(format!("lexical.{i}"), lex.signature(e)?);
        }
        for (name, fact) in self.names.iter().zip(&self.facts) {
            out.insert(name.clone(), fact.signature(e)?);
        }
        out.insert(
            "contract".into(),
            signature(
                e,
                &[],
                &(
                    &self.catalog,
                    &self.budgets,
                    &self.names,
                    &self.goal_routes,
                    FORMAT,
                ),
            )?,
        );
        Ok(out)
    }
    pub fn verify(&self, e: &mut Engine) -> Result<()> {
        for fact in &self.facts {
            fact.verify(e)?;
        }
        ensure(
            self.current_signatures(e)? == self.signatures,
            "original or graph contract changed; incorporate a revision",
        )
    }
    fn seal_snapshot(&mut self, e: &mut Engine) -> Result<()> {
        for fact in &self.facts {
            fact.verify(e)?;
        }
        self.signatures = self.current_signatures(e)?;
        Ok(())
    }
    pub fn save(&self, e: &mut Engine, path: &Path) -> Result<()> {
        self.verify(e)?;
        atomic_directory(path, |dir| {
            for (name, model) in [
                ("scene", &self.scene),
                ("instruction", &self.instruction),
                ("request", &self.request),
                ("query", &self.query_model),
            ] {
                model.save(e, &dir.join(name))?;
            }
            let mut lexica = vec![];
            for (i, lex) in self.lexica.iter().enumerate() {
                let p = PathBuf::from(format!("lexical-{i}"));
                lex.save(e, &dir.join(&p))?;
                lexica.push(p);
            }
            let spec = Assembly {
                scene: "scene".into(),
                instruction: "instruction".into(),
                request: "request".into(),
                query: "query".into(),
                lexica,
                a: self.facts[0].save(e, dir, "a")?,
                b: self.facts[1].save(e, dir, "b")?,
                catalog: self.catalog.clone(),
                budgets: self.budgets.clone(),
                extra: self
                    .names
                    .iter()
                    .enumerate()
                    .skip(2)
                    .map(|(i, name)| {
                        Ok((
                            name.clone(),
                            self.facts[i].save(e, dir, &format!("original-{i}"))?,
                        ))
                    })
                    .collect::<Result<_>>()?,
                goal_routes: self
                    .goal_routes
                    .iter()
                    .map(|r| r.iter().map(|i| self.names[*i].clone()).collect())
                    .collect(),
            };
            std::fs::write(
                dir.join("native.json"),
                json::to_vec_pretty(&Bundle {
                    format: FORMAT.into(),
                    assembly: spec,
                    signatures: self.signatures.clone(),
                })?,
            )?;
            Ok(())
        })
    }
    pub fn load(e: &mut Engine, path: &Path) -> Result<Self> {
        let bundle: Bundle = json::from_slice(&std::fs::read(path.join("native.json"))?)?;
        ensure(bundle.format == FORMAT, "native artifact version mismatch")?;
        let a = &bundle.assembly;
        let mut paths = vec![&a.scene, &a.instruction, &a.request, &a.query];
        paths.extend(a.lexica.iter());
        for source in std::iter::once(&a.a)
            .chain(std::iter::once(&a.b))
            .chain(a.extra.values())
        {
            paths.push(match source {
                FactSource::Memory { path } | FactSource::Native { path } => path,
            });
        }
        ensure(
            paths.iter().all(|p| {
                p.components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)))
            }),
            "artifact paths must remain within the bundle",
        )?;
        let model = Self::assemble(e, a, path)?;
        ensure(
            model.signatures == bundle.signatures,
            "bundle signature mismatch",
        )?;
        Ok(model)
    }
    fn mode(&self, q: &Query, owner: usize) -> ReadMode {
        q.modes
            .get(&self.names[owner])
            .copied()
            .unwrap_or_else(|| self.facts[owner].default_mode())
    }
    fn check_query(&self, q: &Query) -> Result<()> {
        ensure(
            q.tolerance.is_none_or(|v| v.is_finite() && v >= 0.)
                && q.goal < self.goal_routes.len()
                && q.modes.keys().all(|s| self.names.contains(s)),
            "invalid query options",
        )?;
        for owner in 0..self.facts.len() {
            ensure(
                matches!(self.facts[owner], Fact::Memory(_))
                    || self.mode(q, owner) == ReadMode::Full,
                "direct mode requires a prepared memory",
            )?;
        }
        Ok(())
    }
    fn first_task(&self, goal: usize, entity: &str) -> Result<Task> {
        let owner = self.goal_routes[goal][0];
        let row = self.facts[owner]
            .requests()
            .iter()
            .position(|v| v == &ByteValue::from(entity))
            .ok_or_else(|| {
                sejong_core::Error::Message(
                    "semantic identity outside original input inventory".into(),
                )
            })?;
        Ok(Task::Fact { goal, hop: 0, row })
    }
    fn references(&self, q: &Query) -> Result<Vec<String>> {
        let refs = if q.people.is_empty() {
            let words: BTreeSet<_> = q
                .question
                .split(|c: char| !c.is_ascii_alphabetic())
                .collect();
            self.catalog
                .iter()
                .filter(|p| words.contains(p.as_str()))
                .cloned()
                .collect()
        } else {
            q.people.clone()
        };
        inventory(&refs, "query identities")?;
        ensure(
            refs.iter().all(|r| self.catalog.contains(r)),
            "query references must name catalog identities",
        )?;
        Ok(refs)
    }
    fn binding(&self, e: &mut Engine, text: &str, refs: &[String]) -> Result<(Vec<f64>, f64)> {
        lexical::bind(
            e,
            text,
            refs,
            &self.scene,
            &self.instruction,
            &self.request,
            &self.lexica,
            &self.budgets,
        )
    }
    fn interpretation(
        &self,
        e: &mut Engine,
        q: &Query,
        refs: &[String],
        limit: usize,
    ) -> Result<(Option<(Vec<f64>, f64)>, Option<QueryPlan>, Option<Arena>)> {
        if q.story.is_empty() {
            return Ok((Some(self.binding(e, &q.question, refs)?), None, None));
        }
        inventory(&q.predicates, "predicate identities")?;
        ensure(
            q.predicates
                .iter()
                .all(|p| p.bytes().all(|b| b.is_ascii_lowercase()) && !refs.contains(p)),
            "distinct lowercase predicate identities required",
        )?;
        let plan = self
            .query_model
            .query_plan(e, &q.question, refs, &q.predicates)?;
        resources::require(limit, plan.memory_bytes(), "encoded query plan")?;
        let used = plan.predicates.iter().copied().collect::<BTreeSet<_>>();
        let mut laws: Vec<Vec<f64>> = vec![];
        let mut predicates: Vec<usize> = vec![];
        let mut scene_rows: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for event in &q.story {
            let words = event
                .split(|c: char| !c.is_ascii_alphabetic())
                .collect::<Vec<_>>();
            let hits = q
                .predicates
                .iter()
                .enumerate()
                .flat_map(|(i, p)| words.iter().filter(move |w| **w == p).map(move |_| i))
                .collect::<Vec<_>>();
            ensure(
                hits.len() == 1,
                "each story occurrence requires one declared predicate",
            )?;
            let predicate = hits[0];
            if used.contains(&predicate) {
                let row_bytes = resources::bytes::<f64>(Role::Scene.classes_for(refs.len(), 0)?)?;
                let retained = laws.iter().try_fold(
                    plan.memory_bytes()
                        .saturating_add(resources::bytes::<Vec<f64>>(laws.capacity())?)
                        .saturating_add(resources::bytes::<usize>(predicates.capacity())?),
                    |n, row| -> Result<usize> {
                        Ok(n.saturating_add(resources::bytes::<f64>(row.capacity())?))
                    },
                )?;
                let retained =
                    scene_rows
                        .iter()
                        .try_fold(retained, |n, (text, row)| -> Result<usize> {
                            Ok(n.saturating_add(text.capacity())
                                .saturating_add(resources::bytes::<f64>(row.capacity())?)
                                .saturating_add(std::mem::size_of::<(String, Vec<f64>)>()))
                        })?;
                resources::require(
                    limit,
                    retained
                        .saturating_add(row_bytes.saturating_mul(2))
                        .saturating_add(event.len())
                        .saturating_add(std::mem::size_of::<(Vec<f64>, usize)>()),
                    "story occurrence laws",
                )?;
                let canonical =
                    semantic::replace_identity(event, &q.predicates[predicate], "called");
                let law = if let Some(law) = scene_rows.get(&canonical) {
                    law.clone()
                } else {
                    let law = self.scene.law(e, &canonical, refs, &[])?;
                    scene_rows.insert(canonical, law.clone());
                    law
                };
                laws.push(law);
                predicates.push(predicate);
            }
        }
        drop(scene_rows);
        drop(used);
        let arena = Arena::with_people(
            laws,
            predicates,
            refs.len(),
            limit.saturating_sub(plan.memory_bytes()),
        )?;
        Ok((None, Some(plan), Some(arena)))
    }
    pub fn query(&self, e: &mut Engine, q: &Query) -> Result<Answer> {
        self.verify(e)?;
        self.check_query(q)?;
        let refs = self.references(q)?;
        let limit = self.budgets.resolve()?;
        let (flat, plan, mut arena) = self.interpretation(e, q, &refs, limit)?;
        let values = self
            .goal_routes
            .iter()
            .flat_map(|route| self.facts[*route.last().unwrap()].values().iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let fixed = plan
            .as_ref()
            .map_or(0, QueryPlan::memory_bytes)
            .saturating_add(
                flat.as_ref()
                    .map_or(0, |law| law.0.capacity() * std::mem::size_of::<f64>()),
            )
            .saturating_add(
                values
                    .iter()
                    .map(|v| v.0.capacity() + std::mem::size_of::<ByteValue>())
                    .sum::<usize>(),
            );
        let fixed = fixed.saturating_add(resources::bytes::<f64>(resources::product(&[
            2,
            values.len() + 1,
        ])?)?);
        let base_numerical = flat.as_ref().map_or(0., |law| law.1);
        let mut known =
            resources::vector(values.len() + 1, 0., limit, "terminal probability vector")?;
        let mut pending = BTreeMap::new();
        let initial = if flat.is_some() {
            std::mem::size_of::<(Task, f64)>()
        } else {
            plan.as_ref()
                .unwrap()
                .roots
                .iter()
                .filter_map(|(root, _)| root.as_ref())
                .map(|p| {
                    std::mem::size_of::<(Task, f64)>()
                        + p.steps.capacity() * std::mem::size_of::<Step>()
                })
                .sum()
        };
        resources::require(
            limit,
            fixed
                .saturating_add(initial)
                .saturating_add(resources::bytes::<f64>(known.len())?)
                .saturating_add(arena.as_ref().map_or(0, Arena::memory_bytes)),
            "initial query frontier",
        )?;
        if flat.is_some() {
            pending.insert(Task::Flat, 1.);
        } else {
            for (root, mass) in &plan.as_ref().unwrap().roots {
                if let Some(program) = root {
                    add(&mut pending, Task::Prefix(program.clone()), *mass);
                } else {
                    known[values.len()] += mass;
                }
            }
        }
        let mut cache: BTreeMap<(usize, usize), (Vec<f64>, f64)> = BTreeMap::new();
        let (mut error, mut full_rows, mut direct_rows, mut components) = (0f64, 0, 0, 0);
        let mut operations = known.len();
        while !pending.is_empty() {
            let remaining = pending.values().sum::<f64>();
            let rounding = sejong_core::numeric::next_up(
                base_numerical
                    + rounding_error(
                        &known,
                        &pending,
                        operations
                            .saturating_add(arena.as_ref().map_or(0, Arena::arithmetic_operations)),
                    ),
            );
            if !q.full
                && certified(
                    &known,
                    remaining,
                    error,
                    rounding,
                    q.tolerance.unwrap_or(0.),
                )
                .is_some()
            {
                break;
            }
            let base = fixed.saturating_add(live_bytes(&known, &pending, &cache)?);
            let capacity = resources::require(limit, base, "query frontier and row storage")
                .and_then(|()| {
                    if let Some(arena) = arena.as_mut() {
                        arena.set_limit(limit.saturating_sub(base))
                    } else {
                        Ok(())
                    }
                });
            if let Err(err) = capacity {
                if !q.full && matches!(err, sejong_core::Error::Capacity { .. }) {
                    break;
                }
                return Err(err);
            }
            let (task, mass) = pending
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, v)| (k.clone(), *v))
                .unwrap();
            match task.clone() {
                Task::Flat => {
                    if q.max_components.is_some_and(|n| components >= n) {
                        break;
                    }
                    let growth =
                        resources::product(&[2, refs.len(), std::mem::size_of::<(Task, f64)>()])?;
                    if !fits(
                        q,
                        limit,
                        base.saturating_add(growth),
                        "semantic frontier expansion",
                    )? {
                        break;
                    }
                    let law = &flat.as_ref().unwrap().0;
                    pending.remove(&task);
                    components += 1;
                    known[values.len()] += mass * law[2 * refs.len()];
                    for goal in 0..2 {
                        for person in 0..refs.len() {
                            add(
                                &mut pending,
                                self.first_task(goal, &refs[person])?,
                                mass * law[goal * refs.len() + person],
                            );
                        }
                    }
                    operations = operations.saturating_add(law.len().saturating_mul(3));
                }
                Task::Prefix(prefix) => {
                    if q.max_components.is_some_and(|n| components >= n) {
                        break;
                    }
                    let actions =
                        match self
                            .query_model
                            .query_expand(e, plan.as_ref().unwrap(), &prefix)
                        {
                            Ok(actions) => actions,
                            Err(sejong_core::Error::Capacity { .. }) if !q.full => break,
                            Err(err) => return Err(err),
                        };
                    let stop = actions.iter().any(|(step, p)| step.is_none() && *p > 0.);
                    let growth = actions.iter().filter(|(_, p)| *p > 0.).try_fold(
                        resources::bytes::<(Option<Step>, f64)>(actions.capacity())?,
                        |sum, (step, _)| -> Result<usize> {
                            Ok(sum.saturating_add(if step.is_some() {
                                std::mem::size_of::<(Task, f64)>().saturating_add(
                                    resources::bytes::<Step>(prefix.steps.len() + 1)?,
                                )
                            } else {
                                resources::product(&[
                                    2,
                                    refs.len(),
                                    std::mem::size_of::<(Task, f64)>(),
                                ])?
                            }))
                        },
                    )?;
                    if !fits(
                        q,
                        limit,
                        base.saturating_add(growth)
                            .saturating_add(arena.as_ref().unwrap().memory_bytes()),
                        "query prefix expansion",
                    )? {
                        break;
                    }
                    arena
                        .as_mut()
                        .unwrap()
                        .set_limit(limit.saturating_sub(base).saturating_sub(growth))?;
                    let law = if stop {
                        match arena.as_mut().unwrap().program(&prefix) {
                            Ok(p) => Some(p),
                            Err(err)
                                if !q.full
                                    && matches!(err, sejong_core::Error::Capacity { .. }) =>
                            {
                                break;
                            }
                            Err(err) => return Err(err),
                        }
                    } else {
                        None
                    };
                    pending.remove(&task);
                    for (step, p) in actions {
                        if p == 0. {
                            continue;
                        }
                        let weight = mass * p;
                        if let Some(step) = step {
                            let mut next = prefix.clone();
                            next.steps.push(step);
                            add(&mut pending, Task::Prefix(next), weight);
                        } else {
                            components += 1;
                            let law = law.as_ref().unwrap();
                            known[values.len()] += weight * law[2 * refs.len()];
                            for goal in 0..2 {
                                for person in 0..refs.len() {
                                    add(
                                        &mut pending,
                                        self.first_task(goal, &refs[person])?,
                                        weight * law[goal * refs.len() + person],
                                    );
                                }
                            }
                        }
                        operations = operations.saturating_add(3 * (2 * refs.len() + 1));
                    }
                }
                Task::Fact {
                    goal,
                    hop,
                    row: index,
                } => {
                    let stage = self.goal_routes[goal][hop];
                    let downstream = hop + 1 < self.goal_routes[goal].len();
                    let growth = if downstream {
                        resources::bytes::<(Task, f64)>(self.facts[stage].values().len())?
                    } else {
                        0
                    };
                    let cache_growth = if cache.contains_key(&(stage, index)) {
                        0
                    } else {
                        resources::bytes::<f64>(self.facts[stage].values().len())?.saturating_add(std::mem::size_of::<((usize,usize),(Vec<f64>,f64))>())
                    };
                    if !fits(
                        q,
                        limit,
                        base.saturating_add(growth)
                            .saturating_add(cache_growth)
                            .saturating_add(arena.as_ref().map_or(0, Arena::memory_bytes)),
                        "factual row and frontier expansion",
                    )? {
                        break;
                    }
                    if !cache.contains_key(&(stage, index)) {
                        if q.max_rows.is_some_and(|n| full_rows + direct_rows >= n) {
                            break;
                        }
                        let mode = self.mode(q, stage);
                        let row = match self.facts[stage].row(e, index, mode) {
                            Ok(row) => row,
                            Err(sejong_core::Error::Capacity { .. }) if !q.full => break,
                            Err(err) => return Err(err),
                        };
                        if mode == ReadMode::Direct {
                            direct_rows += 1;
                        } else {
                            full_rows += 1;
                        }
                        cache.insert((stage, index), row);
                    }
                    pending.remove(&task);
                    let (law, radius) = &cache[&(stage, index)];
                    if *radius > 0. {
                        error = sejong_core::numeric::next_up(
                            error + sejong_core::numeric::next_up(mass * radius),
                        )
                        .min(1.);
                    }
                    if downstream {
                        let next = self.goal_routes[goal][hop + 1];
                        for (value, p) in self.facts[stage].values().iter().zip(law) {
                            let row = self.facts[next]
                                .requests()
                                .iter()
                                .position(|v| v == value)
                                .unwrap();
                            add(
                                &mut pending,
                                Task::Fact {
                                    goal,
                                    hop: hop + 1,
                                    row,
                                },
                                mass * p,
                            );
                        }
                    } else {
                        for (value, p) in self.facts[stage].values().iter().zip(law) {
                            known[values.binary_search(value).unwrap()] += mass * p;
                        }
                    }
                    operations = operations.saturating_add(law.len().saturating_mul(3));
                }
            }
        }
        let remaining = pending.values().sum::<f64>();
        let numerical_error = sejong_core::numeric::next_up(
            base_numerical
                + rounding_error(
                    &known,
                    &pending,
                    operations
                        .saturating_add(arena.as_ref().map_or(0, Arena::arithmetic_operations)),
                ),
        );
        let prediction = certified(
            &known,
            remaining,
            error,
            numerical_error,
            q.tolerance.unwrap_or(0.),
        );
        let mass = known.iter().sum::<f64>() + remaining;
        ensure(
            (mass - 1.).abs() <= numerical_error,
            "composite probability mass drift",
        )?;
        let lower = known
            .iter()
            .map(|p| {
                sejong_core::numeric::next_down(
                    sejong_core::numeric::next_down(p - error) - numerical_error,
                )
                .max(0.)
            })
            .collect();
        let upper = known
            .iter()
            .map(|p| {
                sejong_core::numeric::next_up(
                    sejong_core::numeric::next_up(
                        sejong_core::numeric::next_up(p + remaining) + error,
                    ) + numerical_error,
                )
                .min(1.)
            })
            .collect();
        let output = prediction
            .filter(|i| *i < values.len())
            .map(|i| values[i].clone());
        let reason = match prediction {
            None => "read_error_or_budget_or_tie",
            Some(i) if i == values.len() => "semantic_unresolved",
            _ => "certified_full_path_map",
        }
        .into();
        Ok(Answer {
            unresolved: known[values.len()],
            values,
            probabilities: known,
            lower,
            upper,
            remaining,
            read_error: error,
            numerical_error,
            certified: prediction.is_some(),
            prediction,
            output,
            reason,
            full_rows,
            direct_rows,
            components,
            event_nodes: arena.as_ref().map_or(0, Arena::count),
        })
    }
    pub fn incorporate_revision(
        &mut self,
        e: &mut Engine,
        stage: usize,
        revision: Memory,
    ) -> Result<()> {
        self.verify(e)?;
        ensure(
            stage < self.facts.len(),
            "revision stage must name a registered original",
        )?;
        revision.verify(e)?;
        let old = match &self.facts[stage] {
            Fact::Memory(memory) => memory,
            _ => {
                return Err(sejong_core::Error::Message(
                    "direct-memory revision requires a memory original".into(),
                ));
            }
        };
        ensure(
            old.metadata.requests == revision.metadata.requests
                && old.metadata.values == revision.metadata.values
                && old.metadata.contexts == revision.metadata.contexts
                && old.metadata.value_targets == revision.metadata.value_targets,
            "K/V revision preserves declared input and output inventories",
        )?;
        ensure(
            old.body_signature(e)? == revision.body_signature(e)?
                && old.port_signature(e)? == revision.port_signature(e)?
                && old.contract_signature(e)? == revision.contract_signature(e)?,
            "K/V revision must preserve its query body and affine entry",
        )?;
        self.facts[stage] = Fact::Memory(revision);
        self.seal_snapshot(e)
    }
    pub fn sample(&self, e: &mut Engine, q: &Query, rng: &mut Rng) -> Result<Sample> {
        let mut complete = q.clone();
        complete.full = true;
        complete.max_rows = None;
        complete.max_components = None;
        let law = self.query(e, &complete)?;
        ensure(
            law.remaining == 0.,
            "sampling requires a complete terminal law",
        )?;
        let bits = f64::MANTISSA_DIGITS;
        let draw = (rng.next_u64() >> (u64::BITS - bits)) as f64 / 2f64.powi(bits as i32);
        let mut accumulated = 0.;
        let mut terminal = law.probabilities.len() - 1;
        for (index, p) in law.probabilities.iter().enumerate() {
            accumulated += p;
            if draw < accumulated {
                terminal = index;
                break;
            }
        }
        Ok(Sample {
            output: law.values.get(terminal).cloned(),
            terminal,
            read_error: law.read_error,
            probabilities: law.probabilities,
        })
    }
    pub fn weights(&self, e: &mut Engine, request: &WeightRequest) -> Result<WeightResult> {
        self.verify(e)?;
        ensure(
            !request.route.is_empty(),
            "continuous route requires an original",
        )?;
        let modes = if request.modes.is_empty() {
            (0..request.route.len())
                .map(|i| {
                    if i == 0 {
                        ReadMode::Full
                    } else {
                        ReadMode::Direct
                    }
                })
                .collect::<Vec<_>>()
        } else {
            request.modes.clone()
        };
        ensure(
            modes.len() == request.route.len(),
            "one read mode is required per route stage",
        )?;
        let limit = self.budgets.resolve()?;
        let retained = request.inputs.iter().try_fold(
            resources::bytes::<Vec<f64>>(request.inputs.len())?,
            |n, row| -> Result<usize> { Ok(n.saturating_add(resources::bytes::<f64>(row.len())?)) },
        )?;
        resources::require(limit, retained, "continuous input rows")?;
        let mut output = request.inputs.clone();
        let mut dimension = None;
        let mut used = BTreeSet::new();
        for (hop, (name, mode)) in request.route.iter().zip(&modes).enumerate() {
            let stage = self.names.iter().position(|n| n == name).ok_or_else(|| {
                sejong_core::Error::Message(format!("unknown continuous-route original {name}"))
            })?;
            let memory = match &self.facts[stage] {
                Fact::Memory(m) => m,
                _ => {
                    return Err(sejong_core::Error::Message(
                        "continuous reading requires memory originals".into(),
                    ));
                }
            };
            let d = memory.metadata.config.dimension;
            if let Some(previous) = dimension {
                ensure(previous == d, "continuous route dimensions must agree")?;
            }
            dimension = Some(d);
            let width = if *mode == ReadMode::Full {
                resources::product(&[memory.metadata.config.tokens, d])?
            } else {
                d
            };
            let current = output.iter().try_fold(
                resources::bytes::<Vec<f64>>(output.len())?,
                |n, row| -> Result<usize> {
                    Ok(n.saturating_add(resources::bytes::<f64>(row.capacity())?))
                },
            )?;
            let expanded = resources::bytes::<f64>(resources::product(&[output.len(), width])?)?;
            let result = resources::bytes::<f64>(resources::product(&[output.len(), d])?)?
                .saturating_add(resources::bytes::<Vec<f64>>(output.len())?);
            resources::require(
                limit,
                current.saturating_add(expanded).saturating_add(result),
                "continuous route workspace",
            )?;
            if hop > 0 {
                for row in &mut output {
                    let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.);
                    for x in row.iter_mut() {
                        *x /= norm;
                    }
                    if *mode == ReadMode::Full {
                        row.try_reserve_exact(width.saturating_sub(row.len()))
                            .map_err(|err| {
                                sejong_core::Error::Message(format!(
                                    "continuous input allocation: {err}"
                                ))
                            })?;
                        row.resize(width, 0.);
                    }
                }
            }
            output = memory.read(e, &output, *mode)?;
            if *mode == ReadMode::Direct {
                used.insert(stage);
            }
        }
        let connection_scalars = used
            .into_iter()
            .map(|stage| match &self.facts[stage] {
                Fact::Memory(m) => m.port.as_ref().map_or(0, |p| p.len()),
                _ => 0,
            })
            .sum();
        Ok(WeightResult {
            output,
            route: request.route.clone(),
            modes,
            connection_scalars,
        })
    }
    pub fn compare_weights(
        &self,
        e: &mut Engine,
        request: &WeightRequest,
    ) -> Result<WeightComparison> {
        let selected = self.weights(e, request)?;
        let mut reference = request.clone();
        reference.modes = vec![ReadMode::Full; request.route.len()];
        // A first direct request is replayed in the first token for the full comparison.
        if selected.modes.first() == Some(&ReadMode::Direct) {
            let first = self
                .names
                .iter()
                .position(|n| n == &request.route[0])
                .unwrap();
            let memory = match &self.facts[first] {
                Fact::Memory(m) => m,
                _ => unreachable!(),
            };
            for row in &mut reference.inputs {
                row.resize(
                    memory.metadata.config.tokens * memory.metadata.config.dimension,
                    0.,
                );
            }
        }
        let full = self.weights(e, &reference)?;
        let max_absolute_error = selected
            .output
            .iter()
            .flatten()
            .zip(full.output.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        Ok(WeightComparison {
            selected,
            full,
            max_absolute_error,
        })
    }
    pub fn accounting(&self) -> Accounting {
        let classifiers = [
            &self.scene,
            &self.instruction,
            &self.request,
            &self.query_model,
        ];
        let mut encoders = BTreeSet::new();
        let mut original_parameters: usize = 0;
        for model in classifiers.into_iter().chain(self.lexica.iter()) {
            original_parameters += model.trainable_parameter_count();
            if let Some(identity) = model.encoder_identity() {
                if encoders.insert(identity.digest.clone()) {
                    original_parameters +=
                        model.parameter_count() - model.trainable_parameter_count();
                }
            }
        }
        let (
            mut connection_scalars,
            mut context_scalars,
            mut readout_scalars,
            mut bound_scalars,
            mut prototype_scalars,
        ) = (0, 0, 0, 0, 0);
        for fact in &self.facts {
            match fact {
                Fact::Memory(m) => {
                    original_parameters += m
                        .parameters
                        .iter()
                        .filter(|p| p.name != "readout" && p.name != "bias")
                        .map(|p| p.value.len())
                        .sum::<usize>();
                    connection_scalars += m.port.as_ref().map_or(0, |p| p.len());
                    context_scalars += m.metadata.contexts.iter().map(Vec::len).sum::<usize>();
                    readout_scalars += m.parameter("readout").len() + m.parameter("bias").len();
                    prototype_scalars +=
                        m.metadata.value_targets.iter().map(Vec::len).sum::<usize>();
                    bound_scalars += m.metadata.query_radius.len() + m.metadata.row_radius.len();
                }
                Fact::Native(m) => {
                    original_parameters +=
                        m.parameters().iter().map(|p| p.value.len()).sum::<usize>()
                }
            }
        }
        Accounting {
            original_parameters,
            connection_scalars,
            context_scalars,
            readout_scalars,
            bound_scalars,
            prototype_scalars,
        }
    }
    pub fn joint(&self, e: &mut Engine, q: &Query) -> Result<JointAnswer> {
        self.verify(e)?;
        self.check_query(q)?;
        let arity = q.joint.len();
        let limit = self.budgets.resolve()?;
        ensure(
            arity > 0 && q.joint.iter().all(|p| self.catalog.contains(p)),
            "joint requires catalog identities",
        )?;
        let mut requests = q
            .joint
            .iter()
            .cloned()
            .map(ByteValue::from)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let count = resources::power(requests.len(), arity)?;
        let mut probabilities = resources::vector(count, 0., limit, "initial joint tensor")?;
        let index = q.joint.iter().fold(0, |i, p| {
            i * requests.len() + requests.binary_search(&ByteValue::from(p.clone())).unwrap()
        });
        probabilities[index] = 1.;
        let (mut error, mut direct_rows, mut full_rows) = (0., 0, 0);
        let mut operations = count;
        for &stage in &self.goal_routes[q.goal] {
            let fact = &self.facts[stage];
            let mode = self.mode(q, stage);
            let input = requests.len();
            let output = fact.values().len();
            let entries = resources::product(&[input, output])?;
            let kernel_bytes = resources::bytes::<f64>(entries)?;
            let radii_bytes = resources::bytes::<f64>(input)?;
            let retained = resources::bytes::<f64>(probabilities.len())?
                .saturating_add(kernel_bytes)
                .saturating_add(radii_bytes)
                .saturating_add(resources::bytes::<f64>(output)?);
            resources::require(limit, retained, "joint rows and input storage")?;
            let mut kernel = Vec::new();
            kernel.try_reserve_exact(entries).map_err(|err| {
                sejong_core::Error::Message(format!("joint row allocation: {err}"))
            })?;
            let mut radii = Vec::new();
            radii.try_reserve_exact(input).map_err(|err| {
                sejong_core::Error::Message(format!("joint radii allocation: {err}"))
            })?;
            for request in &requests {
                let index = fact
                    .requests()
                    .iter()
                    .position(|r| r == request)
                    .ok_or_else(|| {
                        sejong_core::Error::Message("joint input outside fact inventory".into())
                    })?;
                let (law, radius) = fact.row(e, index, mode)?;
                kernel.extend(law);
                radii.push(radius);
                if mode == ReadMode::Direct {
                    direct_rows += 1;
                } else {
                    full_rows += 1;
                }
            }
            (probabilities, error) = partition::bounded_pushforward(
                &probabilities,
                input,
                &kernel,
                output,
                arity,
                &radii,
                error,
                limit.saturating_sub(radii_bytes),
            )?;
            operations =
                operations.saturating_add(partition::operation_bound(input, output, arity));
            requests = fact.values().to_vec();
        }
        let numerical_error =
            sejong_core::numeric::normalized_tolerance(&probabilities, operations);
        ensure(
            (probabilities.iter().sum::<f64>() - 1.).abs() <= numerical_error,
            "joint probability mass drift",
        )?;
        Ok(JointAnswer {
            shape: vec![requests.len(); arity],
            values: requests,
            probabilities,
            read_error: error,
            numerical_error,
            full_rows,
            direct_rows,
        })
    }
}
fn fits(q: &Query, limit: usize, required: usize, operation: &str) -> Result<bool> {
    match resources::require(limit, required, operation) {
        Ok(()) => Ok(true),
        Err(sejong_core::Error::Capacity { .. }) if !q.full => Ok(false),
        Err(err) => Err(err),
    }
}
fn add(pending: &mut BTreeMap<Task, f64>, task: Task, p: f64) {
    if p > 0. {
        *pending.entry(task).or_insert(0.) += p;
    }
}
fn live_bytes(
    known: &[f64],
    pending: &BTreeMap<Task, f64>,
    cache: &BTreeMap<(usize, usize), (Vec<f64>, f64)>,
) -> Result<usize> {
    let mut used = resources::bytes::<f64>(known.len())?;
    for task in pending.keys() {
        used = used.saturating_add(std::mem::size_of::<(Task, f64)>());
        if let Task::Prefix(p) = task {
            used = used.saturating_add(resources::bytes::<Step>(p.steps.len())?);
        }
    }
    for (law, _) in cache.values() {
        used = used
            .saturating_add(resources::bytes::<f64>(law.capacity())?)
            .saturating_add(std::mem::size_of::<((usize, usize), (Vec<f64>, f64))>());
    }
    Ok(used)
}
fn rounding_error(known: &[f64], pending: &BTreeMap<Task, f64>, operations: usize) -> f64 {
    let absolute =
        known.iter().map(|v| v.abs()).sum::<f64>() + pending.values().map(|v| v.abs()).sum::<f64>();
    sejong_core::numeric::sum_error(
        absolute,
        operations
            .saturating_add(known.len())
            .saturating_add(pending.len()),
    )
}
fn certified(
    known: &[f64],
    remaining: f64,
    error: f64,
    rounding: f64,
    tolerance: f64,
) -> Option<usize> {
    if known.len() < 2 {
        return None;
    }
    let mut first = 0;
    let mut second = 1;
    if known[second] > known[first] {
        std::mem::swap(&mut first, &mut second);
    }
    for i in 2..known.len() {
        if known[i] > known[first] {
            second = first;
            first = i;
        } else if known[i] > known[second] {
            second = i;
        }
    }
    let gap = sejong_core::numeric::next_down(known[first] - known[second]);
    let threshold = [2. * error, 2. * rounding, tolerance]
        .into_iter()
        .fold(remaining, |sum, v| sejong_core::numeric::next_up(sum + v));
    if gap > threshold { Some(first) } else { None }
}

#[cfg(test)]
mod checks;
