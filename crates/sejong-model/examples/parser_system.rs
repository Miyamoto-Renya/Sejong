//! One combined check of dynamic semantic coordinates, lexical sequences and program decoding.
use sejong_core::{
    DType, GpuConfig, ModelConfig, Result, Rng, ensure,
    native::{LocalExample, Program, Role, Step},
};
use sejong_gpu::{Engine, graph::Graph};
use sejong_model::native::{
    bert::BertEncoder,
    classifier::{Classifier, ClassifierMetadata},
};

pub fn config() -> ModelConfig {
    ModelConfig {
        width: 32,
        layers: 2,
        heads: 4,
        ffn_width: 80,
        precision: DType::F32,
        ..Default::default()
    }
}
pub fn query() -> LocalExample {
    LocalExample{text:"person0 called person1; person2 thanked person3; person4 helped person5. Follow invited and linked in that order, then return the selected address.".into(),people:(0..6).map(|n|format!("person{n}")).collect(),predicates:["called","thanked","helped","invited","linked"].iter().map(|s|s.to_string()).collect(),label:0,program:Some(Program{goal:1,anchor:5,steps:[0,2,4].into_iter().map(|predicate|Step{predicate,reverse:predicate==2,last:predicate==4}).collect()})}
}
fn model(e: &mut Engine, role: Role, phrases: Vec<String>) -> Result<Classifier> {
    Classifier::initialize(
        e,
        config(),
        ClassifierMetadata {
            role,
            phrases,
            meanings: if role == Role::Lexical {
                vec!["use".into(), "ignore".into()]
            } else {
                vec![]
            },
            steps: 0,
            source_hash: "dynamic-parser-contract".into(),
            loss: 0.,
        },
    )
}
fn normalized(law: &[f64]) -> Result<()> {
    ensure(
        law.iter().all(|p| p.is_finite() && *p >= 0.)
            && (law.iter().sum::<f64>() - 1.).abs() < 1e-10,
        "parser law must be finite and normalized",
    )
}
pub fn check() -> Result<()> {
    let mut e = Engine::new(GpuConfig::default())?;
    let encoder = BertEncoder::load_shared(&mut e, &BertEncoder::default_path())?;
    let capacity = encoder.config.max_position_embeddings;
    let input = encoder.tokenizer.tokenize(&"x ".repeat(capacity - 2))?;
    ensure(
        input.ids.len() == capacity,
        "asset position-capacity fixture differs from tokenizer",
    )?;
    let hidden = encoder.hidden(&mut e, &input.ids)?;
    ensure(
        hidden.shape == [capacity, encoder.config.hidden_size],
        "asset-sized encoder result differs",
    )?;
    let mut overflow = input.ids.clone();
    overflow.push(0);
    ensure(
        encoder.hidden(&mut e, &overflow).is_err(),
        "encoder must apply the loaded position-table size",
    )?;
    drop(hidden);
    let input = query();
    for role in [Role::Scene, Role::Instruction, Role::Request] {
        let m = model(&mut e, role, vec![])?;
        let law = m.law(&mut e, &input.text, &input.people, &[])?;
        ensure(
            law.len() == role.classes_for(6, 0)?,
            "semantic law must follow actual N",
        )?;
        normalized(&law)?;
    }
    let phrase = "길이가 충분히 긴 문장을 보존하는 명시적 어휘 항목".to_string();
    ensure(phrase.len() > 24, "long lexical fixture required")?;
    let lexical = model(&mut e, Role::Lexical, vec![phrase.clone()])?;
    normalized(&lexical.law(&mut e, &phrase, &[], &[])?)?;
    let m = model(&mut e, Role::Query, vec![])?;
    let plan = m.query_plan(&mut e, &input.text, &input.people, &input.predicates)?;
    normalized(&plan.roots.iter().map(|(_, p)| *p).collect::<Vec<_>>())?;
    ensure(
        plan.roots.len() == 13 && plan.predicates.len() == 5 && plan.memory_bytes() > 0,
        "query plan must preserve N=6/P=5 coordinates",
    )?;
    let target = input.program.as_ref().unwrap();
    let mut prefix = Program {
        goal: target.goal,
        anchor: target.anchor,
        steps: vec![],
    };
    for step in &target.steps {
        let next = m.query_expand(&mut e, &plan, &prefix)?;
        normalized(&next.iter().map(|(_, p)| *p).collect::<Vec<_>>())?;
        ensure(
            next.iter()
                .any(|(s, p)| s.as_ref() == Some(step) && *p > 0.),
            "next decoder distribution must include target transition",
        )?;
        prefix.steps.push(step.clone());
    }
    ensure(
        m.query_expand(&mut e, &plan, &prefix)?
            .iter()
            .any(|(s, _)| s.is_none()),
        "depth-three decoder must expose STOP",
    )?;
    let parameters = m.parameters();
    let mut g = Graph::new(&mut e, true);
    let (eval, leaves) = m.evaluate(&mut g, &input, true)?;
    ensure(
        eval.log_probabilities[0].is_finite(),
        "teacher-forced path must have finite log probability",
    )?;
    let roots = eval.seed(&mut g, &[-1.])?;
    g.backward_seeded(&roots)?;
    let mut rng = Rng::new(9485);
    let mut perturbations = vec![];
    let mut analytic = 0.;
    for (p, v) in parameters.iter().zip(leaves) {
        if let Some(gradient) = g.gradient(v) {
            let derivative = g.engine.to_f32(&gradient.clone())?;
            let original = g.engine.to_f32(&p.value)?;
            let direction: Vec<_> = (0..p.value.len())
                .map(|_| rng.normal() / (m.config.width as f32).sqrt())
                .collect();
            analytic += derivative
                .iter()
                .zip(&direction)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>();
            perturbations.push((p.value.clone(), original, direction));
        }
    }
    drop(g);
    let epsilon = f32::EPSILON.cbrt();
    let mut losses = vec![];
    for sign in [1., -1.] {
        for (tensor, original, direction) in &perturbations {
            let values: Vec<_> = original
                .iter()
                .zip(direction)
                .map(|(a, b)| a + sign * epsilon * b)
                .collect();
            let temporary = e.from_f32(&tensor.shape, &values)?;
            e.copy_into(&temporary, tensor)?;
        }
        let mut g = Graph::new(&mut e, false);
        let (eval, _) = m.evaluate(&mut g, &input, false)?;
        losses.push(-eval.log_probabilities[0]);
    }
    let numerical = (losses[0] - losses[1]) / (2. * epsilon as f64);
    let relative =
        (analytic - numerical).abs() / analytic.abs().max(numerical.abs()).max(f32::EPSILON as f64);
    ensure(
        relative < 0.03,
        format!("compositional query gradient relative error {relative}"),
    )?;
    for (tensor, original, _) in &perturbations {
        let temporary = e.from_f32(&tensor.shape, original)?;
        e.copy_into(&temporary, tensor)?;
    }
    let path = sejong_core::root()
        .join(".cache")
        .join(format!("parser-dynamic-artifact-{}", std::process::id()));
    m.save(&mut e, &path)?;
    let restored = Classifier::load(&mut e, &path)?;
    ensure(
        restored.signature(&mut e)? == m.signature(&mut e)?,
        "dynamic parser artifact signature differs",
    )?;
    std::fs::remove_dir_all(path)?;
    println!(
        "Dynamic parser: N=6/P=5/depth=3, lexical_bytes={}, BERT_positions={}, query_gradient_relative_error={relative:.7}, artifact=exact",
        phrase.len(),
        capacity
    );
    Ok(())
}
#[allow(dead_code)]
fn main() -> Result<()> {
    check()
}
