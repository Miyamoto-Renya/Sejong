use super::{events::Arena, lexical, partition, resources, semantic};
use sejong_core::{
    Result, Rng, ensure,
    native::{Budgets, Program, Step},
    numeric,
};
use std::collections::BTreeMap;
fn categorical(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut p = (0..n)
        .map(|_| rng.uniform() as f64 + f32::EPSILON as f64)
        .collect::<Vec<_>>();
    let sum = p.iter().sum::<f64>();
    for v in &mut p {
        *v /= sum;
    }
    p
}
fn close(a: &[f64], b: &[f64], operations: usize) -> Result<()> {
    let tolerance = numeric::sum_error(
        a.iter()
            .map(|v| v.abs())
            .sum::<f64>()
            .max(b.iter().map(|v| v.abs()).sum()),
        operations,
    );
    ensure(
        a.len() == b.len() && a.iter().zip(b).all(|(a, b)| (a - b).abs() <= tolerance),
        "independent exact execution comparison differs",
    )
}
fn full_register(rows: &[semantic::LocalLaws], people: usize) -> Vec<f64> {
    let scene_none = people * people;
    let instruction_none = (people + 1) * (people + 1) - 1;
    let selectors = people + 3;
    let request_none = 2 * selectors;
    let mut state = BTreeMap::from([(
        (scene_none, people, (1usize << people) - 1, request_none),
        1.,
    )]);
    for (s, i, q) in rows {
        let mut next = BTreeMap::new();
        for (&(scene, focus, allowed, request), &mass) in &state {
            *next.entry((scene, focus, allowed, request)).or_insert(0.) +=
                mass * s[scene_none] * i[instruction_none] * q[request_none];
            for pair in 0..scene_none {
                *next.entry((pair, focus, allowed, request)).or_insert(0.) +=
                    mass * s[pair] * i[instruction_none] * q[request_none];
            }
            for action in 0..instruction_none {
                let (f, x) = (action / (people + 1), action % (people + 1));
                let f = if f == people { focus } else { f };
                let a = if x == people {
                    allowed
                } else {
                    allowed & !(1 << x)
                };
                *next.entry((scene, f, a, request)).or_insert(0.) +=
                    mass * s[scene_none] * i[action] * q[request_none];
            }
            for r in 0..request_none {
                *next.entry((scene, focus, allowed, r)).or_insert(0.) +=
                    mass * s[scene_none] * i[instruction_none] * q[r];
            }
        }
        state = next;
    }
    let mut out = vec![0.; 2 * people + 1];
    for ((scene, focus, allowed, request), mass) in state {
        if request == request_none {
            continue;
        }
        let selector = request % selectors;
        let selected = if selector < people {
            selector
        } else if selector == people && scene < scene_none {
            scene / people
        } else if selector == people + 1 && scene < scene_none {
            scene % people
        } else if selector == people + 2 {
            focus
        } else {
            people
        };
        if selected < people && allowed & (1 << selected) != 0 {
            out[request / selectors * people + selected] += mass;
        }
    }
    out[2 * people] = (1. - out[..2 * people].iter().sum::<f64>()).max(0.);
    out
}
fn decode(mut index: usize, size: usize, arity: usize) -> Vec<usize> {
    let mut x = vec![0; arity];
    for j in (0..arity).rev() {
        x[j] = index % size;
        index /= size;
    }
    x
}
fn oracle_events(
    laws: &[Vec<f64>],
    predicates: &[usize],
    program: &Program,
    people: usize,
) -> Vec<f64> {
    let domains = laws
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.)
                .map(|(o, p)| (o, *p))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut worlds = vec![(vec![], 1.)];
    for domain in domains {
        let mut next = vec![];
        for (prefix, p) in worlds {
            for (o, w) in &domain {
                let mut prefix = prefix.clone();
                prefix.push(*o);
                next.push((prefix, p * w));
            }
        }
        worlds = next;
    }
    let mut out = vec![0.; 2 * people + 1];
    for (world, p) in worlds {
        let mut person = program.anchor;
        for step in &program.steps {
            if person == people {
                break;
            }
            let mut result = people;
            let mut order = (0..world.len()).collect::<Vec<_>>();
            if !step.last {
                order.reverse();
            }
            for i in order {
                if predicates[i] != step.predicate || world[i] == people * people {
                    continue;
                }
                let pair = world[i];
                let (from, to) = if step.reverse {
                    (pair % people, pair / people)
                } else {
                    (pair / people, pair % people)
                };
                if from == person {
                    result = to;
                }
            }
            person = result;
        }
        out[if person == people {
            2 * people
        } else {
            program.goal * people + person
        }] += p;
    }
    out
}
#[test]
fn input_sized_exact_composition() -> Result<()> {
    let people = 6usize;
    let mut rng = Rng::new(51691);
    let limit = Budgets::default().resolve()?;
    let rows = (0..3)
        .map(|_| {
            let mut s = categorical(&mut rng, people * people + 1);
            let mut i = categorical(&mut rng, (people + 1) * (people + 1));
            let mut q = categorical(&mut rng, 2 * (people + 3) + 1);
            for p in [&mut s, &mut i, &mut q] {
                for x in p.iter_mut() {
                    *x *= 0.25;
                }
                *p.last_mut().unwrap() += 0.75;
            }
            (s, i, q)
        })
        .collect::<Vec<_>>();
    let oracle = full_register(&rows, people);
    close(
        &oracle,
        &semantic::execute(&rows, people, limit)?,
        resources::power(people + 1, people)?.saturating_mul(rows.len()),
    )?;
    let mut laws = vec![vec![0.; people * people + 1]; 3];
    laws[0][5 * people + 1] = 0.4;
    laws[0][5 * people + 2] = 0.6;
    laws[1][people + 5] = 0.35;
    laws[1][2 * people + 5] = 0.65;
    laws[2][5 * people + 1] = 0.7;
    laws[2][people * people] = 0.3;
    let predicates = vec![5, 0, 5];
    let mut arena = Arena::new(laws.clone(), predicates.clone(), limit)?;
    for last in [false, true] {
        for reverse in [false, true] {
            let program = Program {
                goal: 1,
                anchor: if reverse { 1 } else { 5 },
                steps: (0..8)
                    .map(|i| Step {
                        predicate: if i % 2 == 0 { 5 } else { 0 },
                        reverse,
                        last,
                    })
                    .collect(),
            };
            close(
                &oracle_events(&laws, &predicates, &program, people),
                &arena.program(&program)?,
                laws.len() * program.steps.len() * people * people * people,
            )?;
        }
    }
    let kernel = vec![0.2, 0.8, 0.55, 0.45, 0.7, 0.3];
    let direct = vec![0.21, 0.79, 0.52, 0.48, 0.705, 0.295];
    let radii = vec![0.01, 0.03, 0.005];
    let arity = 5;
    let joint = categorical(&mut rng, resources::power(3, arity)?);
    let output = partition::pushforward(&joint, 3, &kernel, 2, arity, limit)?;
    let mut oracle = vec![0.; resources::power(2, arity)?];
    for world in 0..resources::power(2, 3)? {
        let choices = decode(world, 2, 3);
        let p = (0..3)
            .map(|row| kernel[row * 2 + choices[row]])
            .product::<f64>();
        for (index, mass) in joint.iter().enumerate() {
            let input = decode(index, 3, arity);
            let target = input.iter().fold(0, |n, r| n * 2 + choices[*r]);
            oracle[target] += p * mass;
        }
    }
    let operations = partition::operation_bound(3, 2, arity);
    close(&output, &oracle, operations)?;
    let observable = (0..output.len())
        .map(|i| (i as f64).sin())
        .collect::<Vec<_>>();
    let pulled = partition::pullback(&observable, 2, &kernel, 3, arity, limit)?;
    let lhs = output
        .iter()
        .zip(&observable)
        .map(|(p, f)| p * f)
        .sum::<f64>();
    let rhs = joint.iter().zip(&pulled).map(|(p, f)| p * f).sum::<f64>();
    close(&[lhs], &[rhs], operations)?;
    let (approx, error) =
        partition::bounded_pushforward(&joint, 3, &direct, 2, arity, &radii, 0., limit)?;
    let tv = output
        .iter()
        .zip(&approx)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 2.;
    ensure(
        tv <= error + numeric::sum_error(1., operations),
        "shared-row error propagation differs",
    )?;
    ensure(
        matches!(
            partition::pushforward(&joint, 3, &kernel, 2, arity, std::mem::size_of::<f64>()),
            Err(sejong_core::Error::Capacity { .. })
        ),
        "joint allocation must report byte capacity",
    )?;
    let mut none_s = vec![0.; people * people + 1];
    *none_s.last_mut().unwrap() = 1.;
    let mut none_q = vec![0.; 2 * (people + 3) + 1];
    *none_q.last_mut().unwrap() = 1.;
    let mut use_law = vec![0.; (people + 1) * (people + 1)];
    use_law[people] = 1.;
    let mut ignore = vec![0.; use_law.len()];
    ignore[people * (people + 1)] = 1.;
    let operators = vec![
        semantic::Operator::from_laws(&(none_s.clone(), use_law, none_q.clone()))?,
        semantic::Operator::from_laws(&(none_s.clone(), ignore, none_q))?,
    ];
    let mut clauses = vec![];
    let priors = vec![vec![0.75, 0.25]; 32];
    let mut state = semantic::initial(people, limit)?;
    for (key, prior) in priors.iter().enumerate() {
        clauses.push(lexical::Clause {
            latent: Some(key),
            operators: operators.clone(),
        });
        clauses.push(lexical::Clause {
            latent: Some(key),
            operators: operators.clone(),
        });
        let mut next = vec![0.; semantic::states(people)?];
        for (choice, p) in prior.iter().enumerate() {
            let first = operators[choice].apply(&state)?;
            let second = operators[choice].apply(&first)?;
            for (to, from) in next.iter_mut().zip(second) {
                *to += p * from;
            }
        }
        state = next;
    }
    let mut request = vec![0.; 2 * (people + 3) + 1];
    request[people + 2] = 1.;
    let mut none_i = vec![0.; (people + 1) * (people + 1)];
    *none_i.last_mut().unwrap() = 1.;
    let final_op = semantic::Operator::from_laws(&(none_s, none_i, request))?;
    clauses.push(lexical::Clause {
        latent: None,
        operators: vec![final_op.clone()],
    });
    state = final_op.apply(&state)?;
    close(
        &semantic::readout(&state)?,
        &lexical::execute(&clauses, &priors, people, &Budgets::default())?,
        clauses.len() * semantic::states(people)?,
    )?;
    Ok(())
}
