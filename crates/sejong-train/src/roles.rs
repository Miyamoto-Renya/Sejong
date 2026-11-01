//! Variable-cardinality semantic supervision and teacher-forced program training.
use crate::Adam;
use sejong_core::{
    DType, Error, Result, Rng, TrainConfig, ensure, json,
    native::{LocalExample, Program, Role, RolePair, RoleTraining},
};
use sejong_gpu::{
    Engine, Tensor,
    graph::{Graph, Var},
};
use sejong_model::{
    native::{
        atomic_directory,
        classifier::{Classifier, ClassifierMetadata, FORMAT},
    },
    save_tensors,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    format: String,
    source_hash: String,
    rng_state: u64,
    step: usize,
    batch: usize,
    optimizer: TrainConfig,
}
pub fn role_fingerprint(spec: &RoleTraining) -> Result<String> {
    Ok(blake3::hash(&json::to_vec(&(
        FORMAT,
        &spec.role,
        &spec.model,
        &spec.examples,
        &spec.phrases,
        &spec.meanings,
        &spec.pairs,
        spec.steps,
        spec.learning_rate,
        spec.batch_size,
        &spec.loss,
        &spec.optimizer,
        &spec.batch,
    ))?)
    .to_hex()
    .to_string())
}
fn remap(slot: usize, names: &[String], target: &[String]) -> Result<usize> {
    let name = names
        .get(slot)
        .ok_or_else(|| Error::Message("paired identity coordinate outside inventory".into()))?;
    target
        .iter()
        .position(|s| s == name)
        .ok_or_else(|| Error::Message("paired views must preserve identity inventories".into()))
}
fn inventories(a: &LocalExample, b: &LocalExample) -> Result<()> {
    ensure(
        a.people.len() == b.people.len() && a.predicates.len() == b.predicates.len(),
        "paired identity inventory dimensions differ",
    )?;
    for i in 0..a.people.len() {
        remap(i, &a.people, &b.people)?;
    }
    for i in 0..a.predicates.len() {
        remap(i, &a.predicates, &b.predicates)?;
    }
    Ok(())
}
fn program_remap(
    p: &Option<Program>,
    source: &LocalExample,
    target: &LocalExample,
) -> Result<Option<Program>> {
    p.as_ref()
        .map(|p| {
            let mut p = p.clone();
            p.anchor = remap(p.anchor, &source.people, &target.people)?;
            for step in &mut p.steps {
                step.predicate = remap(step.predicate, &source.predicates, &target.predicates)?;
            }
            Ok(p)
        })
        .transpose()
}
fn aligned(
    role: Role,
    source: &LocalExample,
    target: &LocalExample,
    index: usize,
) -> Result<usize> {
    let n = source.people.len();
    Ok(match role {
        Role::Scene => {
            if index == n * n {
                index
            } else {
                remap(index / n, &source.people, &target.people)? * n
                    + remap(index % n, &source.people, &target.people)?
            }
        }
        Role::Instruction => {
            let slots = n + 1;
            let a = index / slots;
            let b = index % slots;
            (if a == n {
                n
            } else {
                remap(a, &source.people, &target.people)?
            }) * slots
                + if b == n {
                    n
                } else {
                    remap(b, &source.people, &target.people)?
                }
        }
        Role::Request => {
            let slots = n + 3;
            if index == 2 * slots {
                index
            } else {
                index / slots * slots
                    + if index % slots < n {
                        remap(index % slots, &source.people, &target.people)?
                    } else {
                        index % slots
                    }
            }
        }
        Role::Lexical => index,
        Role::Query => {
            return Err(Error::Message(
                "query alignment requires conditional coordinates".into(),
            ));
        }
    })
}
fn query_mapping(source: &LocalExample, target: &LocalExample, node: usize) -> Result<Vec<usize>> {
    if node == 0 {
        let n = source.people.len();
        let mut result = (0..2 * n)
            .map(|i| Ok(i / n * n + remap(i % n, &source.people, &target.people)?))
            .collect::<Result<Vec<_>>>()?;
        result.push(2 * n);
        Ok(result)
    } else {
        let p = source.predicates.len();
        let mut result = (0..4 * p)
            .map(|i| Ok(remap(i / 4, &source.predicates, &target.predicates)? * 4 + i % 4))
            .collect::<Result<Vec<_>>>()?;
        result.push(4 * p);
        Ok(result)
    }
}
fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
/// Jensen-Shannon and its derivatives for a prepared coordinate mapping.
fn consistency(
    left: &[f64],
    right: &[f64],
    mapping: impl Fn(usize) -> usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut aligned = vec![f64::NEG_INFINITY; left.len()];
    for (i, q) in right.iter().enumerate() {
        let j = mapping(i);
        aligned[j] = logaddexp(aligned[j], *q);
    }
    let mut dl = vec![0.; left.len()];
    let mut dr = vec![0.; right.len()];
    let mut loss = 0.;
    for (i, p) in left.iter().enumerate() {
        if p.is_finite() {
            let log_mean = logaddexp(*p, aligned[i]) - 2f64.ln();
            let term = p.exp() * (*p - log_mean) / 2.;
            loss += term;
            dl[i] = term;
        }
    }
    for (i, q) in right.iter().enumerate() {
        if q.is_finite() {
            let log_mean = logaddexp(left[mapping(i)], *q) - 2f64.ln();
            let term = q.exp() * (*q - log_mean) / 2.;
            loss += term;
            dr[i] = term;
        }
    }
    (loss, dl, dr)
}
enum ClassCoordinates {
    Scene(Vec<usize>),
    Instruction(Vec<usize>),
    Request(Vec<usize>),
    Lexical,
}
impl ClassCoordinates {
    fn prepare(role: Role, source: &LocalExample, target: &LocalExample) -> Result<Self> {
        let people = (0..source.people.len())
            .map(|i| remap(i, &source.people, &target.people))
            .collect::<Result<Vec<_>>>()?;
        Ok(match role {
            Role::Scene => Self::Scene(people),
            Role::Instruction => Self::Instruction(people),
            Role::Request => Self::Request(people),
            Role::Lexical => Self::Lexical,
            Role::Query => {
                return Err(Error::Message(
                    "query alignment uses conditional coordinates".into(),
                ));
            }
        })
    }
    fn map(&self, index: usize) -> usize {
        match self {
            Self::Scene(people) => {
                let n = people.len();
                if index == n * n {
                    index
                } else {
                    people[index / n] * n + people[index % n]
                }
            }
            Self::Instruction(people) => {
                let n = people.len();
                let slots = n + 1;
                let (a, b) = (index / slots, index % slots);
                (if a == n { n } else { people[a] }) * slots + if b == n { n } else { people[b] }
            }
            Self::Request(people) => {
                let n = people.len();
                let slots = n + 3;
                if index == 2 * slots {
                    index
                } else {
                    index / slots * slots
                        + if index % slots < n {
                            people[index % slots]
                        } else {
                            index % slots
                        }
                }
            }
            Self::Lexical => index,
        }
    }
}
enum PairObjective {
    Supervised,
    QueryConsistency {
        root: Vec<usize>,
        action: Vec<usize>,
    },
    QueryContrast([Option<LocalExample>; 2]),
    ClassConsistency(ClassCoordinates),
    ClassContrast([usize; 2]),
}
struct PairPlan {
    weights: [f64; 2],
    targets: [usize; 2],
    objective: PairObjective,
}
impl PairPlan {
    fn prepare(model: &Classifier, spec: &RoleTraining, pair: &RolePair) -> Result<Self> {
        inventories(&pair.left, &pair.right)?;
        let names = if pair.equivalent {
            ["supervised", "consistency"]
        } else {
            ["supervised", "contrast"]
        };
        let weights = spec.loss.normalized(&names)?;
        let targets = [
            model.target_index(&pair.left),
            model.target_index(&pair.right),
        ];
        let objective = if weights[1] == 0. {
            PairObjective::Supervised
        } else if spec.role == Role::Query {
            if pair.equivalent {
                ensure(
                    program_remap(&pair.right.program, &pair.right, &pair.left)?
                        == pair.left.program,
                    "equivalent query views require aligned target programs",
                )?;
                PairObjective::QueryConsistency {
                    root: query_mapping(&pair.right, &pair.left, 0)?,
                    action: query_mapping(&pair.right, &pair.left, 1)?,
                }
            } else {
                let mut alternates = [None, None];
                for (slot, source, target) in
                    [(0, &pair.right, &pair.left), (1, &pair.left, &pair.right)]
                {
                    let alternate = program_remap(&source.program, source, target)?;
                    if alternate != target.program && model.supports_program(target, &alternate)? {
                        let mut row = target.clone();
                        row.program = alternate;
                        alternates[slot] = Some(row);
                    }
                }
                PairObjective::QueryContrast(alternates)
            }
        } else {
            if pair.equivalent {
                PairObjective::ClassConsistency(ClassCoordinates::prepare(
                    spec.role,
                    &pair.right,
                    &pair.left,
                )?)
            } else {
                PairObjective::ClassContrast([
                    aligned(spec.role, &pair.right, &pair.left, targets[1])?,
                    aligned(spec.role, &pair.left, &pair.right, targets[0])?,
                ])
            }
        };
        Ok(Self {
            weights: [weights[0], weights[1]],
            targets,
            objective,
        })
    }
}
fn softplus(x: f64) -> f64 {
    if x > 0. {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let a = x.exp();
        a / (1. + a)
    }
}
fn gradient_sum(g: &mut Graph, leaves: &[Vec<Var>], count: usize) -> Result<Vec<Option<Tensor>>> {
    let mut gradients = vec![None; count];
    for view in leaves {
        for (i, v) in view.iter().enumerate() {
            if let Some(incoming) = g.gradient(*v).cloned() {
                gradients[i] = Some(if let Some(current) = gradients[i].take() {
                    g.engine.add(&current, &incoming, 1., DType::F32)?
                } else {
                    incoming
                });
            }
        }
    }
    Ok(gradients)
}
fn supervised(
    g: &mut Graph,
    model: &Classifier,
    row: &LocalExample,
    scale: f64,
) -> Result<(f64, Vec<Option<Tensor>>)> {
    let (eval, leaves) = model.evaluate(g, row, true)?;
    let target = model.target_index(row);
    let gold = eval.log_probabilities[target];
    ensure(
        gold.is_finite(),
        "local target lies outside observed identities/program grammar",
    )?;
    let mut derivative = vec![0.; eval.log_probabilities.len()];
    derivative[target] = -scale;
    let roots = eval.seed(g, &derivative)?;
    g.backward_seeded(&roots)?;
    Ok((
        -gold * scale,
        gradient_sum(g, &[leaves], model.parameters().len())?,
    ))
}
fn pair_loss(
    g: &mut Graph,
    model: &Classifier,
    pair: &RolePair,
    plan: &PairPlan,
    scale: f64,
) -> Result<(f64, Vec<Option<Tensor>>)> {
    let weights = plan.weights;
    let (left, lv) = model.evaluate(g, &pair.left, true)?;
    let (right, rv) = model.evaluate(g, &pair.right, true)?;
    let [li, ri] = plan.targets;
    let mut leaves = vec![lv, rv];
    let mut dl = vec![0.; left.log_probabilities.len()];
    let mut dr = vec![0.; right.log_probabilities.len()];
    let mut loss = 0.;
    let mut seed_left = weights[0] > 0.;
    let mut seed_right = seed_left;
    if weights[0] > 0. {
        ensure(
            left.log_probabilities[li].is_finite() && right.log_probabilities[ri].is_finite(),
            "paired target lies outside observed identities/program grammar",
        )?;
        dl[li] = -weights[0] * scale / 2.;
        dr[ri] = -weights[0] * scale / 2.;
        loss =
            -(left.log_probabilities[li] + right.log_probabilities[ri]) * weights[0] * scale / 2.;
    }
    let mut roots = vec![];
    match &plan.objective {
        PairObjective::Supervised => {}
        PairObjective::QueryConsistency { root, action } => {
            let lc = left.conditionals().unwrap();
            let rc = right.conditionals().unwrap();
            let factor = weights[1] * scale / lc.len() as f64;
            let mut a = vec![];
            let mut b = vec![];
            for node in 0..lc.len() {
                let mapping = if node == 0 { root } else { action };
                let log_l: Vec<_> = lc[node].iter().map(|p| p.ln()).collect();
                let log_r: Vec<_> = rc[node].iter().map(|p| p.ln()).collect();
                let (value, mut da, mut db) = consistency(&log_l, &log_r, |i| mapping[i]);
                loss += value * factor;
                for x in da.iter_mut().chain(&mut db) {
                    *x *= factor;
                }
                a.push(da);
                b.push(db);
            }
            roots.extend(left.seed_conditionals(g, &a)?);
            roots.extend(right.seed_conditionals(g, &b)?);
        }
        PairObjective::QueryContrast(alternates) => {
            for (row, gold, derivative, seed) in [
                (&alternates[0], &left, &mut dl, &mut seed_left),
                (&alternates[1], &right, &mut dr, &mut seed_right),
            ] {
                if let Some(row) = row {
                    let (other, vars) = model.evaluate(g, row, true)?;
                    leaves.push(vars);
                    let delta = other.log_probabilities[0] - gold.log_probabilities[0];
                    let factor = weights[1] * scale / 2.;
                    loss += factor * softplus(delta);
                    let gradient = factor * sigmoid(delta);
                    derivative[0] -= gradient;
                    *seed = true;
                    roots.extend(other.seed(g, &[gradient])?);
                }
            }
        }
        PairObjective::ClassConsistency(mapping) => {
            let (value, a, b) =
                consistency(&left.log_probabilities, &right.log_probabilities, |i| {
                    mapping.map(i)
                });
            loss += value * weights[1] * scale;
            for (i, a) in a.iter().enumerate() {
                dl[i] += a * weights[1] * scale;
            }
            for (i, b) in b.iter().enumerate() {
                dr[i] += b * weights[1] * scale;
            }
            seed_left = true;
            seed_right = true;
        }
        PairObjective::ClassContrast(others) => {
            for (law, derivative, gold, other, seed) in [
                (&left, &mut dl, li, others[0], &mut seed_left),
                (&right, &mut dr, ri, others[1], &mut seed_right),
            ] {
                if gold != other {
                    let delta = law.log_probabilities[other] - law.log_probabilities[gold];
                    let factor = weights[1] * scale / 2.;
                    loss += factor * softplus(delta);
                    let gradient = factor * sigmoid(delta);
                    derivative[gold] -= gradient;
                    derivative[other] += gradient;
                    *seed = true;
                }
            }
        }
    }
    if seed_left {
        roots.extend(left.seed(g, &dl)?);
    }
    if seed_right {
        roots.extend(right.seed(g, &dr)?);
    }
    g.backward_seeded(&roots)?;
    Ok((loss, gradient_sum(g, &leaves, model.parameters().len())?))
}
fn save_checkpoint(
    e: &mut Engine,
    model: &Classifier,
    opt: &Adam,
    root: &Path,
    rng: &Rng,
    source_hash: &str,
    batch: usize,
    config: &TrainConfig,
) -> Result<()> {
    atomic_directory(root, |dir| {
        model.save(e, &dir.join("model"))?;
        let parameters = model.parameters();
        let tensors = opt.tensors(&parameters);
        save_tensors(
            e,
            &dir.join("optimizer.safetensors"),
            tensors.iter().map(|(n, t)| (n.as_str(), t)),
        )?;
        std::fs::write(
            dir.join("training.json"),
            json::to_vec_pretty(&Checkpoint {
                format: FORMAT.into(),
                source_hash: source_hash.into(),
                rng_state: rng.state,
                step: opt.step,
                batch,
                optimizer: config.clone(),
            })?,
        )?;
        Ok(())
    })
}
pub fn train_role(e: &mut Engine, spec: &RoleTraining, out: &Path) -> Result<Classifier> {
    train_role_resume(e, spec, out, spec.resume.as_deref())
}
pub fn train_role_resume(
    e: &mut Engine,
    spec: &RoleTraining,
    out: &Path,
    resume: Option<&Path>,
) -> Result<Classifier> {
    ensure(
        !out.exists(),
        format_args!("output already exists: {}", out.display()),
    )?;
    ensure(
        spec.steps > 0 && spec.learning_rate.is_finite() && spec.learning_rate > 0.,
        "positive role training steps and learning rate required",
    )?;
    let observations = spec
        .examples
        .len()
        .checked_add(spec.pairs.len())
        .ok_or_else(|| Error::Message("observation count overflow".into()))?;
    ensure(observations > 0, "local labeled observations required")?;
    for row in spec
        .examples
        .iter()
        .chain(spec.pairs.iter().flat_map(|p| [&p.left, &p.right]))
    {
        if spec.role != Role::Query {
            ensure(
                row.label
                    < spec
                        .role
                        .classes_for(row.people.len(), spec.meanings.len())?,
                "role label outside input-sized output inventory",
            )?;
        }
    }
    let source_hash = role_fingerprint(spec)?;
    let (mut model, mut optimizer, mut rng, resolved) = if let Some(path) = resume {
        let checkpoint: Checkpoint = json::from_slice(&std::fs::read(path.join("training.json"))?)?;
        ensure(
            checkpoint.format == FORMAT && checkpoint.source_hash == source_hash,
            "role checkpoint training contract differs",
        )?;
        let model = Classifier::load(e, &path.join("model"))?;
        ensure(
            model.metadata.steps == checkpoint.step && model.metadata.source_hash == source_hash,
            "role checkpoint step or data fingerprint differs",
        )?;
        let optimizer = Adam::load_parameters(
            e,
            &model.parameters(),
            &path.join("optimizer.safetensors"),
            checkpoint.step,
        )?;
        (
            model,
            optimizer,
            Rng::new(checkpoint.rng_state),
            Some((checkpoint.batch, checkpoint.optimizer)),
        )
    } else {
        let metadata = ClassifierMetadata {
            role: spec.role,
            phrases: spec.phrases.clone(),
            meanings: spec.meanings.clone(),
            steps: 0,
            source_hash: source_hash.clone(),
            loss: 0.,
        };
        let model = Classifier::initialize_with_encoder(
            e,
            spec.model.clone(),
            metadata,
            spec.encoder.as_deref(),
        )?;
        let optimizer = Adam::new(e, &model.parameters())?;
        (model, optimizer, Rng::new(spec.model.seed), None)
    };
    model.open_training();
    ensure(
        optimizer.step < spec.steps,
        "role checkpoint has reached target steps",
    )?;
    let pair_plans = spec
        .pairs
        .iter()
        .map(|pair| PairPlan::prepare(&model, spec, pair))
        .collect::<Result<Vec<_>>>()?;
    let mut cost = 0;
    for row in &spec.examples {
        cost = cost.max(model.example_memory_bytes(row)?);
    }
    for (pair, plan) in spec.pairs.iter().zip(&pair_plans) {
        let mut pair_cost = model
            .example_memory_bytes(&pair.left)?
            .checked_add(model.example_memory_bytes(&pair.right)?)
            .ok_or_else(|| Error::Message("pair memory size overflow".into()))?;
        if let PairObjective::QueryContrast(alternates) = &plan.objective {
            for row in alternates.iter().flatten() {
                pair_cost = pair_cost
                    .checked_add(model.example_memory_bytes(row)?)
                    .ok_or_else(|| Error::Message("pair memory size overflow".into()))?;
            }
        }
        cost = cost.max(pair_cost);
    }
    let mut policy = spec.batch.clone();
    if spec.batch_size > 0 {
        policy.max_examples = Some(
            policy
                .max_examples
                .map_or(spec.batch_size, |n| n.min(spec.batch_size)),
        );
    }
    let (batch, config) = if let Some((batch, config)) = resolved {
        ensure(
            cost <= e.available_memory_bytes()?,
            "checkpoint batch exceeds available parser training memory",
        )?;
        let expected =
            spec.optimizer
                .configuration(spec.steps, spec.learning_rate, batch, observations)?;
        ensure(
            json::to_value(&config)? == json::to_value(expected)?,
            "checkpoint optimizer policy differs",
        )?;
        (batch, config)
    } else {
        let batch = policy.resolve(e.available_memory_bytes()?, cost, observations)?;
        let config =
            spec.optimizer
                .configuration(spec.steps, spec.learning_rate, batch, observations)?;
        (batch, config)
    };
    if !spec.examples.is_empty() {
        spec.loss.normalized(&["supervised"])?;
    }
    std::fs::create_dir_all(out)?;
    while optimizer.step < spec.steps {
        let mut loss = 0.;
        for _ in 0..batch {
            let index = rng.index(observations);
            let mut graph = Graph::new(e, true);
            let (value, gradients) = if index < spec.examples.len() {
                supervised(&mut graph, &model, &spec.examples[index], 1. / batch as f64)?
            } else {
                pair_loss(
                    &mut graph,
                    &model,
                    &spec.pairs[index - spec.examples.len()],
                    &pair_plans[index - spec.examples.len()],
                    1. / batch as f64,
                )?
            };
            loss += value;
            drop(graph);
            optimizer.accumulate(e, gradients)?;
        }
        ensure(loss.is_finite(), "nonfinite role training loss")?;
        optimizer.update(e, &model.parameters(), &config, spec.learning_rate)?;
        model.metadata.steps = optimizer.step;
        model.metadata.loss = loss;
        if optimizer.step == 1
            || optimizer.step == spec.steps
            || (spec.checkpoint_every > 0 && optimizer.step.is_multiple_of(spec.checkpoint_every))
        {
            eprintln!("{:?} {} loss {:.6}", spec.role, optimizer.step, loss);
        }
        if spec.checkpoint_every > 0 && optimizer.step.is_multiple_of(spec.checkpoint_every) {
            save_checkpoint(
                e,
                &model,
                &optimizer,
                &out.join("checkpoints")
                    .join(format!("step-{:08}", optimizer.step)),
                &rng,
                &source_hash,
                batch,
                &config,
            )?;
        }
    }
    model.save(e, &out.join("model"))?;
    save_checkpoint(
        e,
        &model,
        &optimizer,
        &out.join("checkpoint"),
        &rng,
        &source_hash,
        batch,
        &config,
    )?;
    std::fs::write(out.join("training-spec.json"), json::to_vec_pretty(spec)?)?;
    Ok(model)
}
