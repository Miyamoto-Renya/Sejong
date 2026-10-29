//! Input-sized relational kernels and ordered decisions for shared event choices.
use super::resources;
use sejong_core::{
    Result, ensure,
    native::{Program, Step},
};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone)]
struct Node {
    variable: usize,
    children: Vec<usize>,
}
type Choice = (usize, Vec<usize>);
pub struct Arena {
    people: usize,
    nodes: Vec<Node>,
    unique: BTreeMap<Choice, usize>,
    choices: BTreeMap<Choice, usize>,
    laws: Vec<Vec<f64>>,
    predicates: Vec<usize>,
    limit: usize,
    storage: usize,
    workspace: usize,
    operations: usize,
    scans: BTreeMap<(usize, usize, bool, bool), usize>,
    domains: Vec<Vec<usize>>,
    kernels: BTreeMap<(usize, bool, bool), Vec<f64>>,
    components: BTreeMap<(usize, usize, Vec<(usize, bool, bool)>), Vec<f64>>,
}
impl Arena {
    pub fn new(laws: Vec<Vec<f64>>, predicates: Vec<usize>, limit_bytes: usize) -> Result<Self> {
        let entries = laws.first().map_or(0, |p| p.len().saturating_sub(1));
        let people = (entries as f64).sqrt() as usize;
        Self::with_people(laws, predicates, people, limit_bytes)
    }
    pub fn with_people(
        laws: Vec<Vec<f64>>,
        predicates: Vec<usize>,
        people: usize,
        limit: usize,
    ) -> Result<Self> {
        let outcomes = resources::product(&[people, people])?
            .checked_add(1)
            .ok_or_else(|| sejong_core::Error::Message("event outcome shape overflow".into()))?;
        ensure(
            people > 0
                && laws.len() == predicates.len()
                && laws
                    .iter()
                    .all(|r| r.len() == outcomes && resources::normalized(r)),
            "normalized input-sized event laws required",
        )?;
        let domains_entries = laws
            .iter()
            .map(|p| p.iter().filter(|p| **p > 0.).count())
            .sum::<usize>();
        let laws_bytes = laws.iter().try_fold(
            resources::bytes::<Vec<f64>>(laws.capacity())?,
            |n, row| -> Result<usize> {
                Ok(n.saturating_add(resources::bytes::<f64>(row.capacity())?))
            },
        )?;
        let storage = laws_bytes
            .saturating_add(resources::bytes::<usize>(domains_entries)?)
            .saturating_add(resources::bytes::<Vec<usize>>(laws.len())?)
            .saturating_add(resources::bytes::<usize>(predicates.capacity())?);
        resources::require(limit, storage, "event input storage")?;
        let domains: Vec<Vec<usize>> = laws
            .iter()
            .map(|p| {
                p.iter()
                    .enumerate()
                    .filter(|(_, p)| **p > 0.)
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        Ok(Self {
            people,
            nodes: vec![],
            unique: BTreeMap::new(),
            choices: BTreeMap::new(),
            laws,
            predicates,
            limit,
            storage,
            workspace: 0,
            operations: 0,
            scans: BTreeMap::new(),
            domains,
            kernels: BTreeMap::new(),
            components: BTreeMap::new(),
        })
    }
    pub fn count(&self) -> usize {
        self.nodes.len()
    }
    pub fn memory_bytes(&self) -> usize {
        self.storage
    }
    pub fn arithmetic_operations(&self) -> usize {
        self.operations
    }
    pub fn set_limit(&mut self, limit: usize) -> Result<()> {
        resources::require(
            limit,
            self.storage.saturating_add(self.workspace),
            "event live storage",
        )?;
        self.limit = limit;
        Ok(())
    }
    fn charge(&mut self, bytes: usize, operation: &str) -> Result<()> {
        let required = self
            .storage
            .checked_add(bytes)
            .ok_or_else(|| sejong_core::Error::Message("event storage overflow".into()))?;
        resources::require(
            self.limit,
            required.saturating_add(self.workspace),
            operation,
        )?;
        self.storage = required;
        Ok(())
    }
    fn terminals(&self) -> usize {
        self.people + 1
    }
    fn variable(&self, id: usize) -> usize {
        if id < self.terminals() {
            usize::MAX
        } else {
            self.nodes[id - self.terminals()].variable
        }
    }
    fn restricted(&self, id: usize, var: usize, outcome: usize) -> usize {
        if self.variable(id) == var {
            self.nodes[id - self.terminals()].children[outcome]
        } else {
            id
        }
    }
    fn choose(&mut self, var: usize, children: Vec<usize>) -> Result<usize> {
        enum Work {
            Enter(Choice),
            Finish(Choice, usize, Vec<Choice>),
        }
        let requested = (var, children);
        if let Some(id) = self.choices.get(&requested) {
            return Ok(*id);
        }
        let mut work = vec![Work::Enter(requested.clone())];
        let mut transient = requested.1.len() * std::mem::size_of::<usize>();
        while let Some(item) = work.pop() {
            match item {
                Work::Enter(key) => {
                    if self.choices.contains_key(&key) {
                        continue;
                    }
                    let (first_var, children) = (&key.0, &key.1);
                    let first = children
                        .iter()
                        .map(|id| self.variable(*id))
                        .min()
                        .unwrap()
                        .min(*first_var);
                    if first < *first_var {
                        let mut subkeys = Vec::new();
                        for outcome in 0..self.domains[first].len() {
                            let values = children
                                .iter()
                                .map(|id| self.restricted(*id, first, outcome))
                                .collect::<Vec<_>>();
                            subkeys.push((*first_var, values));
                        }
                        transient = transient.saturating_add(
                            subkeys
                                .iter()
                                .map(|x| x.1.len() * std::mem::size_of::<usize>())
                                .sum::<usize>(),
                        );
                        resources::require(
                            self.limit,
                            self.storage
                                .saturating_add(self.workspace)
                                .saturating_add(transient),
                            "event reorder workspace",
                        )?;
                        work.push(Work::Finish(key, first, subkeys.clone()));
                        for child in subkeys.into_iter().rev() {
                            if !self.choices.contains_key(&child) {
                                work.push(Work::Enter(child));
                            }
                        }
                    } else {
                        let values = children
                            .iter()
                            .enumerate()
                            .map(|(o, id)| self.restricted(*id, *first_var, o))
                            .collect();
                        let id = self.node(*first_var, values)?;
                        self.cache_choice(key, id)?;
                    }
                }
                Work::Finish(key, first, subkeys) => {
                    let mut branches = Vec::with_capacity(subkeys.len());
                    for child in &subkeys {
                        branches.push(self.choices[child]);
                    }
                    transient = transient.saturating_sub(
                        subkeys
                            .iter()
                            .map(|x| x.1.len() * std::mem::size_of::<usize>())
                            .sum::<usize>(),
                    );
                    let id = self.node(first, branches)?;
                    self.cache_choice(key, id)?;
                }
            }
        }
        Ok(self.choices[&requested])
    }
    fn cache_choice(&mut self, key: Choice, id: usize) -> Result<()> {
        if !self.choices.contains_key(&key) {
            self.charge(
                std::mem::size_of::<(Choice, usize)>() + key.1.len() * std::mem::size_of::<usize>(),
                "event choice cache",
            )?;
            self.choices.insert(key, id);
        }
        Ok(())
    }
    fn node(&mut self, var: usize, children: Vec<usize>) -> Result<usize> {
        if children.iter().all(|c| *c == children[0]) {
            return Ok(children[0]);
        }
        let key = (var, children);
        if let Some(id) = self.unique.get(&key) {
            return Ok(*id);
        }
        self.charge(
            std::mem::size_of::<Node>()
                + std::mem::size_of::<(Choice, usize)>()
                + 2 * key.1.len() * std::mem::size_of::<usize>(),
            "event decision storage",
        )?;
        let id = self.nodes.len() + self.terminals();
        self.nodes.push(Node {
            variable: var,
            children: key.1.clone(),
        });
        self.unique.insert(key, id);
        Ok(id)
    }
    fn scan(&mut self, anchor: usize, step: &Step) -> Result<usize> {
        if anchor == self.people {
            return Ok(self.people);
        }
        let key = (anchor, step.predicate, step.reverse, step.last);
        if let Some(id) = self.scans.get(&key) {
            return Ok(*id);
        }
        let mut events = self
            .predicates
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == step.predicate)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if !step.last {
            events.reverse();
        }
        let mut root = self.people;
        for event in events {
            let children = self.domains[event]
                .iter()
                .map(|o| {
                    if *o == self.people * self.people {
                        return root;
                    }
                    let (from, to) = if step.reverse {
                        (o % self.people, o / self.people)
                    } else {
                        (o / self.people, o % self.people)
                    };
                    if from == anchor { to } else { root }
                })
                .collect();
            root = self.choose(event, children)?;
        }
        self.charge(
            std::mem::size_of::<((usize, usize, bool, bool), usize)>(),
            "event scan cache",
        )?;
        self.scans.insert(key, root);
        Ok(root)
    }
    fn compose(&mut self, root: usize, step: &Step) -> Result<usize> {
        if root < self.terminals() {
            return self.scan(root, step);
        }
        let mut reachable = BTreeSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if id >= self.terminals() && reachable.insert(id) {
                pending.extend(self.nodes[id - self.terminals()].children.iter().copied());
            }
        }
        resources::require(
            self.limit,
            self.storage
                .saturating_add(self.workspace)
                .saturating_add(reachable.len() * (std::mem::size_of::<usize>() * 3)),
            "event substitution workspace",
        )?;
        let mut transformed = BTreeMap::new();
        for value in 0..self.terminals() {
            transformed.insert(value, self.scan(value, step)?);
        }
        for id in reachable {
            let node = self.nodes[id - self.terminals()].clone();
            let children = node
                .children
                .iter()
                .map(|child| transformed[child])
                .collect();
            let changed = self.choose(node.variable, children)?;
            transformed.insert(id, changed);
        }
        Ok(transformed[&root])
    }
    fn law(&mut self, root: usize) -> Result<Vec<f64>> {
        let width = self.terminals();
        let mut reachable = BTreeSet::new();
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if id >= width && reachable.insert(id) {
                pending.extend(self.nodes[id - width].children.iter().copied());
            }
        }
        let workspace = resources::product(&[
            reachable.len().saturating_add(width),
            width,
            std::mem::size_of::<f64>(),
        ])?;
        resources::require(
            self.limit,
            self.storage
                .saturating_add(self.workspace)
                .saturating_add(workspace),
            "event law workspace",
        )?;
        let mut laws = BTreeMap::new();
        for value in 0..width {
            let mut p = vec![0.; width];
            p[value] = 1.;
            laws.insert(value, p);
        }
        for id in reachable {
            let node = &self.nodes[id - width];
            self.operations = self
                .operations
                .saturating_add(node.children.len().saturating_mul(width).saturating_mul(2));
            let mut p = vec![0.; width];
            for (outcome, child) in node.children.iter().enumerate() {
                let weight = self.laws[node.variable][self.domains[node.variable][outcome]];
                for (i, v) in laws[child].iter().enumerate() {
                    p[i] += weight * v;
                }
            }
            laws.insert(id, p);
        }
        Ok(laws.remove(&root).unwrap())
    }
    fn kernel(&mut self, step: &Step) -> Result<Vec<f64>> {
        let key = (step.predicate, step.reverse, step.last);
        if let Some(k) = self.kernels.get(&key) {
            return Ok(k.clone());
        }
        let width = self.terminals();
        let bytes = resources::bytes::<f64>(resources::product(&[width, width])?)?;
        self.charge(
            bytes + std::mem::size_of::<((usize, bool, bool), Vec<f64>)>(),
            "event scan kernel",
        )?;
        let mut result = vec![0.; width * width];
        result[self.people * width + self.people] = 1.;
        let mut events = self
            .predicates
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == step.predicate)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if !step.last {
            events.reverse();
        }
        self.operations = self.operations.saturating_add(
            events.len().saturating_mul(self.people).saturating_mul(
                self.people
                    .saturating_mul(self.people)
                    .saturating_add(self.people.saturating_mul(2))
                    .saturating_add(1),
            ),
        );
        for anchor in 0..self.people {
            let mut hits = vec![0.; self.people];
            let mut absent = 1.;
            for &event in &events {
                let law = &self.laws[event];
                let mut row = vec![0.; self.people];
                for pair in 0..self.people * self.people {
                    let (from, to) = if step.reverse {
                        (pair % self.people, pair / self.people)
                    } else {
                        (pair / self.people, pair % self.people)
                    };
                    if from == anchor {
                        row[to] += law[pair];
                    }
                }
                let miss = (1. - row.iter().sum::<f64>()).max(0.);
                for target in 0..self.people {
                    hits[target] = hits[target] * miss + row[target];
                }
                absent *= miss;
            }
            result[anchor * width..anchor * width + self.people].copy_from_slice(&hits);
            result[anchor * width + self.people] = absent;
        }
        self.kernels.insert(key, result.clone());
        Ok(result)
    }
    pub fn program(&mut self, program: &Program) -> Result<Vec<f64>> {
        ensure(
            program.goal < 2 && program.anchor < self.people,
            "program goal/anchor outside input coordinates",
        )?;
        let key = (
            program.goal,
            program.anchor,
            program
                .steps
                .iter()
                .map(|s| (s.predicate, s.reverse, s.last))
                .collect::<Vec<_>>(),
        );
        if let Some(law) = self.components.get(&key) {
            return Ok(law.clone());
        }
        let previous = self.workspace;
        let width = self.terminals();
        let unique = program
            .steps
            .iter()
            .map(|s| s.predicate)
            .collect::<BTreeSet<_>>();
        let scratch = resources::bytes::<f64>(
            width
                .saturating_mul(3)
                .saturating_add(2 * self.people + 1)
                .saturating_add(if unique.len() == program.steps.len() {
                    resources::product(&[width, width])?
                } else {
                    0
                }),
        )?
        .saturating_add(resources::bytes::<usize>(self.laws.len())?)
        .saturating_add(resources::bytes::<(usize, bool, bool)>(
            program.steps.len(),
        )?);
        self.workspace = previous.saturating_add(scratch);
        if let Err(error) = resources::require(
            self.limit,
            self.storage.saturating_add(self.workspace),
            "event program workspace",
        ) {
            self.workspace = previous;
            return Err(error);
        }
        let result = (|| {
            let p = if unique.len() == program.steps.len() {
                let mut p = vec![0.; self.terminals()];
                p[program.anchor] = 1.;
                for step in &program.steps {
                    let kernel = self.kernel(step)?;
                    self.operations = self
                        .operations
                        .saturating_add(p.len().saturating_mul(p.len()).saturating_mul(2));
                    let mut next = vec![0.; p.len()];
                    for from in 0..p.len() {
                        for to in 0..p.len() {
                            next[to] += p[from] * kernel[from * p.len() + to];
                        }
                    }
                    p = next;
                }
                p
            } else {
                let mut root = program.anchor;
                for step in &program.steps {
                    root = self.compose(root, step)?;
                }
                self.law(root)?
            };
            let mut result = vec![0.; 2 * self.people + 1];
            for i in 0..self.people {
                result[program.goal * self.people + i] = p[i];
            }
            result[2 * self.people] = p[self.people];
            ensure(
                (result.iter().sum::<f64>() - 1.).abs()
                    <= resources::sum_error(&result, self.operations),
                "event computation mass drift",
            )?;
            self.charge(
                resources::bytes::<f64>(result.len())?
                    + key.2.len() * std::mem::size_of::<(usize, bool, bool)>()
                    + std::mem::size_of::<(usize, usize, Vec<(usize, bool, bool)>)>(),
                "event component cache",
            )?;
            self.components.insert(key, result.clone());
            Ok(result)
        })();
        self.workspace = previous;
        result
    }
}
