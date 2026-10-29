//! Combined dynamic parser execution and paired-program checkpoint continuation.
#[path = "../../sejong-model/examples/parser_system.rs"]
mod parser_system;
use sejong_core::{
    GpuConfig, Result, ensure, json,
    native::{Role, RolePair, RoleTraining},
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::{load_tensors, native::classifier::Classifier};
use sejong_train::{
    Adam,
    roles::{train_role, train_role_resume},
};
fn main() -> Result<()> {
    parser_system::check()?;
    let mut e = Engine::new(GpuConfig::default())?;
    let left = parser_system::query();
    let mut right = left.clone();
    right.text.push_str(" Please.");
    right.people.reverse();
    right.predicates.reverse();
    right.program = left.program.clone().map(|mut p| {
        p.anchor = left.people.len() - 1 - p.anchor;
        for s in &mut p.steps {
            s.predicate = left.predicates.len() - 1 - s.predicate;
        }
        p
    });
    let spec = RoleTraining {
        role: Role::Query,
        model: parser_system::config(),
        steps: 2,
        learning_rate: 0.0002,
        examples: vec![],
        phrases: vec![],
        meanings: vec![],
        encoder: None,
        pairs: vec![RolePair {
            left,
            right,
            equivalent: true,
        }],
        resume: None,
        batch_size: 1,
        checkpoint_every: 1,
        loss: Default::default(),
        optimizer: Default::default(),
        batch: Default::default(),
    };
    let root = sejong_core::root()
        .join(".cache")
        .join(format!("parser-dynamic-resume-{}", std::process::id()));
    let full = root.join("full");
    let resumed = root.join("resumed");
    let reference = train_role(&mut e, &spec, &full)?;
    let checkpoint = full.join("checkpoints/step-00000001");
    let checkpoint_model = Classifier::load(&mut e, &checkpoint.join("model"))?;
    let parameters = checkpoint_model.parameters();
    let optimizer = Adam::load_parameters(
        &mut e,
        &parameters,
        &checkpoint.join("optimizer.safetensors"),
        1,
    )?;
    let restored = optimizer.tensors(&parameters);
    let recorded = load_tensors(&mut e, &checkpoint.join("optimizer.safetensors"))?;
    for (name, tensor) in restored {
        ensure(
            e.to_f32(&tensor)? == e.to_f32(&recorded[&name])?,
            "optimizer checkpoint restoration differs",
        )?;
    }
    drop(optimizer);
    drop(checkpoint_model);
    let continued = train_role_resume(&mut e, &spec, &resumed, Some(&checkpoint))?;
    let mut weight_error = 0f32;
    for (a, b) in reference.parameters().iter().zip(continued.parameters()) {
        let a = e.to_f32(&a.value)?;
        let b = e.to_f32(&b.value)?;
        for (a, b) in a.into_iter().zip(b) {
            weight_error = weight_error.max((a - b).abs());
        }
    }
    let a = load_tensors(&mut e, &full.join("checkpoint/optimizer.safetensors"))?;
    let b = load_tensors(&mut e, &resumed.join("checkpoint/optimizer.safetensors"))?;
    let mut moment_error = 0f32;
    for (name, tensor) in a {
        let a = e.to_f32(&tensor)?;
        let b = e.to_f32(&b[&name])?;
        for (a, b) in a.into_iter().zip(b) {
            moment_error = moment_error.max((a - b).abs());
        }
    }
    let mut laws = vec![];
    for m in [&reference, &continued] {
        let mut graph = Graph::new(&mut e, false);
        let (eval, _) = m.evaluate(&mut graph, &spec.pairs[0].left, false)?;
        laws.push(eval.log_probabilities[0]);
    }
    let probability_error = (laws[0].exp() - laws[1].exp()).abs();
    ensure(
        probability_error <= f32::EPSILON.sqrt() as f64,
        "continued program probability differs beyond FP32 reduction precision",
    )?;
    let saved: json::Value =
        json::from_slice(&std::fs::read(resumed.join("checkpoint/training.json"))?)?;
    let original: json::Value =
        json::from_slice(&std::fs::read(full.join("checkpoint/training.json"))?)?;
    ensure(
        saved == original,
        "resumed RNG, step or optimizer configuration differs",
    )?;
    let loaded = Classifier::load(&mut e, &resumed.join("model"))?;
    ensure(
        loaded.metadata.steps == 2,
        "resumed parser update count differs",
    )?;
    println!(
        "Paired program resume: checkpoint restoration/RNG/config=exact, final weight_max_delta={weight_error:.9}, moment_max_delta={moment_error:.9}, program_probability_delta={probability_error:.12}"
    );
    std::fs::remove_dir_all(root)?;
    Ok(())
}
