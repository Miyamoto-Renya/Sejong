//! Complete local learning and gold-answer checks for the native model graph.
use sejong_core::{GpuConfig, Result, Rng, ensure, json, native::*};
use sejong_gpu::Engine;
use sejong_model::native::{classifier::Classifier, memory::Memory, value::NativeValue};
use sejong_runtime::native::{Composite, Fact};
use std::{collections::BTreeMap, path::PathBuf};
fn read<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    Ok(json::from_slice(&std::fs::read(path)?)?)
}
fn gold(answer: &Answer, expected: &ByteValue, label: &str) -> Result<()> {
    ensure(
        answer.output.as_ref() == Some(expected),
        format!(
            "{label}: expected {expected}, got {:?}, reason {}",
            answer.output, answer.reason
        ),
    )
}
fn owner(model: &Composite, name: &str) -> Result<usize> {
    model
        .names
        .iter()
        .position(|n| n == name)
        .ok_or_else(|| sejong_core::Error::Message(format!("unknown original {name}")))
}
fn full_modes(model: &Composite) -> BTreeMap<String, ReadMode> {
    model
        .names
        .iter()
        .map(|name| (name.clone(), ReadMode::Full))
        .collect()
}
fn shared_coordinates(index: usize, size: usize, identities: &[String]) -> bool {
    let mut value = index;
    let mut coordinates = vec![0; identities.len()];
    for coordinate in coordinates.iter_mut().rev() {
        *coordinate = value % size;
        value /= size;
    }
    identities.iter().enumerate().all(|(axis, name)| {
        identities[..axis]
            .iter()
            .enumerate()
            .all(|(previous, other)| name != other || coordinates[axis] == coordinates[previous])
    })
}
fn main() -> Result<()> {
    let data = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "data/native-v6".into());
    let out = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| ".cache/native-v6".into());
    std::fs::create_dir_all(&out)?;
    let mut e = Engine::new(GpuConfig::default())?;
    for i in 0..6 {
        let spec: RoleTraining = read(&data.join(format!("role-{i}.json")))?;
        let path = out.join(format!("role-{i}"));
        if path.join("model").exists() {
            let m = Classifier::load(&mut e, &path.join("model"))?;
            ensure(
                m.metadata.source_hash == sejong_train::roles::role_fingerprint(&spec)?,
                "stored role source differs from current training specification",
            )?;
            drop(m);
        } else {
            drop(sejong_train::roles::train_role(&mut e, &spec, &path)?);
        }
    }
    for name in ["a", "b"] {
        let spec: MemoryTraining = read(&data.join(format!("{name}.json")))?;
        let path = out.join(name);
        if path.join("model").exists() {
            let m = Memory::load(&mut e, &path.join("model"))?;
            ensure(
                m.metadata.source_hash == sejong_train::memory::memory_fingerprint(&spec)?,
                "stored memory source or recipe differs",
            )?;
            drop(m);
        } else {
            drop(sejong_train::memory::train_memory(&mut e, &spec, &path)?);
        }
    }
    let value_spec: ValueTraining = read(&data.join("native-b.json"))?;
    let path = out.join("native-b");
    if path.join("model").exists() {
        let m = NativeValue::load(&mut e, &path.join("model"))?;
        ensure(
            m.metadata.source_hash == sejong_train::memory::value_fingerprint(&value_spec)?,
            "stored native-value source or recipe differs",
        )?;
        drop(m);
    } else {
        drop(sejong_train::memory::train_value(
            &mut e,
            &value_spec,
            &path,
        )?);
    }
    let assembly: Assembly = read(&data.join("assembly.json"))?;
    let mut model = Composite::assemble(&mut e, &assembly, &out)?;
    let expected: json::Value = read(&data.join("gold.json"))?;
    let mut results = BTreeMap::new();
    let mut queries = vec![];
    let mut atomic_hits = vec![];
    for name in ["a", "b"] {
        let stage = owner(&model, name)?;
        let spec: MemoryTraining = read(&data.join(format!("{name}.json")))?;
        let mut hits = 0;
        for (i, label) in spec.labels.iter().enumerate() {
            let (law, _) = model.facts[stage].row(&mut e, i, ReadMode::Full)?;
            let winner = (0..law.len())
                .max_by(|a, b| law[*a].total_cmp(&law[*b]))
                .unwrap();
            hits += usize::from(winner == *label);
        }
        ensure(
            hits == spec.labels.len(),
            format!("{name}: atomic facts {hits}/{}", spec.labels.len()),
        )?;
        atomic_hits.push((hits, spec.labels.len()));
    }
    results.insert("atomic-facts".into(), json::to_value(atomic_hits)?);
    for (file, key) in [
        ("query.json", "flat"),
        ("story.json", "story"),
        ("story-two.json", "story-two"),
    ] {
        let mut q: Query = read(&data.join(file))?;
        q.full = true;
        let wanted: ByteValue = json::from_value(expected[key].clone())?;
        let direct = model.query(&mut e, &q)?;
        gold(&direct, &wanted, &format!("{key} direct"))?;
        q.modes = full_modes(&model);
        let full = model.query(&mut e, &q)?;
        gold(&full, &wanted, &format!("{key} full"))?;
        ensure(
            full.probabilities
                .iter()
                .zip(direct.lower.iter().zip(&direct.upper))
                .all(|(p, (l, u))| {
                    sejong_core::numeric::next_up(*p + full.numerical_error) >= *l
                        && sejong_core::numeric::next_down(*p - full.numerical_error) <= *u
                }),
            "full probabilities outside direct interval",
        )?;
        eprintln!(
            "{key}: {} / direct rows {} error {:.8}",
            wanted, direct.direct_rows, direct.read_error
        );
        results.insert(key.to_string(), json::to_value(&direct)?);
        queries.push((q, wanted));
    }
    let people = queries[0].0.people.clone();
    let mut lexical: Query = read(&data.join("lexical.json"))?;
    lexical.full = true;
    let lexical_result = model.query(&mut e, &lexical)?;
    let wanted: ByteValue = json::from_value(expected["lexical"].clone())?;
    gold(&lexical_result, &wanted, "lexical")?;
    results.insert("lexical".into(), json::to_value(&lexical_result)?);
    let mut permuted: Query = read(&data.join("query.json"))?;
    permuted.people = people.clone();
    permuted.people.rotate_left(1);
    let permutation_answer = model.query(&mut e, &permuted)?;
    let permutation_gold: ByteValue = json::from_value(expected["flat"].clone())?;
    gold(
        &permutation_answer,
        &permutation_gold,
        "identity-coordinate permutation",
    )?;
    results.insert("permutation".into(), json::to_value(permutation_answer)?);
    let mut renamed: Query = read(&data.join("renamed.json"))?;
    renamed.full = true;
    let renamed_answer = model.query(&mut e, &renamed)?;
    let renamed_gold: ByteValue = json::from_value(expected["renamed"].clone())?;
    gold(
        &renamed_answer,
        &renamed_gold,
        "renamed identities with independent local facts",
    )?;
    results.insert("renamed".into(), json::to_value(renamed_answer)?);
    let none = Query {
        question: format!(
            "{} called {}. What is the weather?",
            people[0],
            people.get(1).unwrap_or(&people[0])
        ),
        people: people.clone(),
        full: true,
        ..Default::default()
    };
    let none = model.query(&mut e, &none)?;
    ensure(
        none.output.is_none() && none.reason == "semantic_unresolved",
        "NONE case was turned into a factual answer",
    )?;
    results.insert("none".into(), json::to_value(none)?);
    let mut joint: Query = read(&data.join("joint.json"))?;
    let direct = model.joint(&mut e, &joint)?;
    joint.modes = full_modes(&model);
    let full = model.joint(&mut e, &joint)?;
    let n = direct.values.len();
    ensure(
        direct
            .probabilities
            .iter()
            .enumerate()
            .all(|(i, p)| *p == 0. || shared_coordinates(i, n, &joint.joint)),
        "same row draw was split across repeated requests",
    )?;
    let tv: f64 = direct
        .probabilities
        .iter()
        .zip(&full.probabilities)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 2.;
    ensure(
        tv <= sejong_core::numeric::next_up(
            sejong_core::numeric::next_up(direct.read_error + direct.numerical_error)
                + full.numerical_error,
        ),
        "joint TV exceeds propagated bound",
    )?;
    results.insert(
        "joint".into(),
        json::json!({"tv":tv,"bound":direct.read_error}),
    );
    let mut mixed = assembly.clone();
    mixed.b = FactSource::Native {
        path: "native-b/model".into(),
    };
    let mixed = Composite::assemble(&mut e, &mixed, &out)?;
    let mut q: Query = read(&data.join("query.json"))?;
    q.full = true;
    let mixed_answer = mixed.query(&mut e, &q)?;
    let wanted: ByteValue = json::from_value(expected["flat"].clone())?;
    gold(&mixed_answer, &wanted, "memory/native factual mixture")?;
    ensure(
        mixed_answer.direct_rows > 0 && mixed_answer.full_rows > 0,
        "mixed query failed to execute full/direct originals",
    )?;
    results.insert("mixed-native".into(), json::to_value(mixed_answer)?);
    let mut native_hits = 0;
    let mixed_b = owner(&mixed, "b")?;
    for (i, label) in value_spec.labels.iter().enumerate() {
        let (law, _) = mixed.facts[mixed_b].row(&mut e, i, ReadMode::Full)?;
        let winner = (0..law.len())
            .max_by(|a, b| law[*a].total_cmp(&law[*b]))
            .unwrap();
        native_hits += usize::from(winner == *label);
    }
    ensure(
        native_hits == value_spec.labels.len(),
        format!("native B facts {native_hits}/{}", value_spec.labels.len()),
    )?;
    results.insert(
        "native-facts".into(),
        json::json!([native_hits, value_spec.labels.len()]),
    );
    if !out.join("model").exists() {
        model.save(&mut e, &out.join("model"))?;
    }
    let restored = Composite::load(&mut e, &out.join("model"))?;
    for (q, wanted) in &queries {
        let answer = restored.query(&mut e, q)?;
        gold(&answer, wanted, "restored graph")?;
    }
    let mut rng = Rng::new(91);
    let sampled = restored.sample(&mut e, &queries[0].0, &mut rng)?;
    results.insert("sample".into(), json::to_value(sampled)?);
    drop(restored);
    let a: MemoryTraining = read(&data.join("a.json"))?;
    let route = WeightRequest {
        inputs: vec![a.contexts[0].clone()],
        route: vec!["a".into(), "b".into()],
        modes: vec![ReadMode::Full, ReadMode::Direct],
    };
    let compared = model.compare_weights(&mut e, &route)?;
    results.insert("continuous".into(), json::to_value(compared)?);
    let revision: RevisionTraining = read(&data.join("revision.json"))?;
    let revision_owner = owner(&model, "b")?;
    let old_b = match &model.facts[revision_owner] {
        Fact::Memory(m) => m,
        _ => unreachable!(),
    };
    let old_signature = old_b.signature(&mut e)?;
    let old_port = e.to_f64(old_b.port.as_ref().unwrap())?;
    let revised = if out.join("revision-local/model").exists() {
        Memory::load(&mut e, &out.join("revision-local/model"))?
    } else {
        sejong_train::memory::revise_memory(
            &mut e,
            old_b,
            &revision,
            &out.join("revision-local"),
            None,
        )?
    };
    ensure(
        e.to_f64(revised.port.as_ref().unwrap())? == old_port,
        "revision changed the query entrance",
    )?;
    ensure(
        old_b.signature(&mut e)? == old_signature,
        "local revision changed the old snapshot",
    )?;
    model.incorporate_revision(&mut e, revision_owner, revised)?;
    let mut q: Query = read(&data.join("query.json"))?;
    q.full = true;
    let answer = model.query(&mut e, &q)?;
    let revised_expected: ByteValue = json::from_value(expected["revision"].clone())?;
    gold(&answer, &revised_expected, "revised fact propagation")?;
    if !out.join("revised-model").exists() {
        model.save(&mut e, &out.join("revised-model"))?;
    }
    results.insert("revision".into(), json::to_value(answer)?);
    results.insert("accounting".into(), json::to_value(model.accounting())?);
    results.insert("gpu_peak_bytes".into(), json::json!(e.peak_bytes()));
    std::fs::write(out.join("answers.json"), json::to_vec_pretty(&results)?)?;
    eprintln!("native gold-answer integration passed: {}", out.display());
    Ok(())
}
