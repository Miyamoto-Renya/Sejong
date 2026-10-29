use anyhow::Result;
use clap::{Parser, Subcommand};
use sejong_core::{Config, GpuConfig, Manifest, json};
use sejong_gpu::Engine;
use sejong_model::Model;
use sejong_runtime::{Sampling, TextStream};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "sejong",
    version,
    about = "Sejong independent-module language model"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Assemble prepared independent originals into one native composite.
    Connect {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Train a semantic, program or lexical original from local labels.
    TrainRole {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Prepare a query body and train its factual K/V weights.
    TrainMemory {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Train a native value head over request states.
    TrainValue {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Sample the complete native probability law.
    Sample {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Execute continuous full/direct memory routes.
    ReadWeights {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        compare: bool,
    },
    /// Recompute direct-read bounds and save a prepared original.
    Prepare {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Execute a semantic, story or shared-fact query.
    Query {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        request: PathBuf,
    },
    /// Train a local K/V revision and incorporate it into a new bundle.
    Revise {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        stage: String,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Initialize an independent Transformer and save its weights.
    Init {
        #[arg(long, default_value = "configs/sejong.toml")]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Train from UTF-8 text files; resume with the saved optimizer and RNG.
    Train {
        #[arg(long, default_value = "configs/sejong.toml")]
        config: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Generate bytes using the model's private KV cache.
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 512)]
        max_bytes: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        raw: bool,
    },
    /// Display model configuration and training provenance.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Connect { spec, out } => {
            let input: sejong_core::native::Assembly = json::from_slice(&std::fs::read(&spec)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            let model = sejong_runtime::native::Composite::assemble(
                &mut e,
                &input,
                spec.parent().unwrap_or(std::path::Path::new(".")),
            )?;
            model.save(&mut e, &out)?;
            println!("{}", out.display());
        }
        Commands::TrainRole { spec, out, resume } => {
            let mut input: sejong_core::native::RoleTraining =
                json::from_slice(&std::fs::read(spec)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            if resume.is_some() {
                input.resume = resume;
            }
            sejong_train::native::train_role(&mut e, &input, &out)?;
            println!("{}", out.display());
        }
        Commands::TrainMemory { spec, out, resume } => {
            let mut input: sejong_core::native::MemoryTraining =
                json::from_slice(&std::fs::read(spec)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            if resume.is_some() {
                input.resume = resume;
            }
            sejong_train::native::train_memory(&mut e, &input, &out)?;
            println!("{}", out.display());
        }
        Commands::TrainValue { spec, out, resume } => {
            let mut input: sejong_core::native::ValueTraining =
                json::from_slice(&std::fs::read(spec)?)?;
            if resume.is_some() {
                input.resume = resume;
            }
            let mut e = Engine::new(GpuConfig::default())?;
            sejong_train::native::train_value(&mut e, &input, &out)?;
            println!("{}", out.join("model").display());
        }
        Commands::Sample {
            model,
            request,
            seed,
        } => {
            let input = json::from_slice(&std::fs::read(request)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            let composite = sejong_runtime::native::Composite::load(&mut e, &model)?;
            let mut rng = sejong_core::Rng::new(seed);
            println!(
                "{}",
                json::to_string_pretty(&composite.sample(&mut e, &input, &mut rng)?)?
            );
        }
        Commands::ReadWeights {
            model,
            request,
            compare,
        } => {
            let input = json::from_slice(&std::fs::read(request)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            let composite = sejong_runtime::native::Composite::load(&mut e, &model)?;
            let value = if compare {
                json::to_value(composite.compare_weights(&mut e, &input)?)?
            } else {
                json::to_value(composite.weights(&mut e, &input)?)?
            };
            println!("{}", json::to_string_pretty(&value)?);
        }
        Commands::Prepare { model, out } => {
            let mut e = Engine::new(GpuConfig::default())?;
            let mut memory = sejong_model::native::memory::Memory::load(&mut e, &model)?;
            memory.calibrate(&mut e)?;
            memory.save(&mut e, &out)?;
            println!("{}", out.display());
        }
        Commands::Query { model, request } => {
            let input: sejong_core::native::Query = json::from_slice(&std::fs::read(request)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            let composite = sejong_runtime::native::Composite::load(&mut e, &model)?;
            let result = if input.joint.is_empty() {
                json::to_value(composite.query(&mut e, &input)?)?
            } else {
                json::to_value(composite.joint(&mut e, &input)?)?
            };
            println!("{}", json::to_string_pretty(&result)?);
        }
        Commands::Revise {
            model,
            stage,
            spec,
            out,
            resume,
        } => {
            let input: sejong_core::native::RevisionTraining =
                json::from_slice(&std::fs::read(spec)?)?;
            let mut e = Engine::new(GpuConfig::default())?;
            let mut composite = sejong_runtime::native::Composite::load(&mut e, &model)?;
            let index = composite
                .names
                .iter()
                .position(|name| name == &stage)
                .ok_or_else(|| anyhow::anyhow!("unknown original {stage}"))?;
            let original = match &composite.facts[index] {
                sejong_runtime::native::Fact::Memory(memory) => memory,
                _ => anyhow::bail!("K/V revision requires a prepared memory original"),
            };
            std::fs::create_dir_all(&out)?;
            let revised = sejong_train::native::revise_memory(
                &mut e,
                original,
                &input,
                &out.join("local"),
                resume.as_deref().or(input.resume.as_deref()),
            )?;
            composite.incorporate_revision(&mut e, index, revised)?;
            composite.save(&mut e, &out.join("model"))?;
            println!("{}", out.display());
        }
        Commands::Init { config, out } => {
            let config = Config::load(&config)?;
            let mut engine = Engine::new(config.gpu)?;
            let model = Model::initialize(&mut engine, config.model)?;
            model.save(&mut engine, &out, model.manifest(0), None)?;
            println!(
                "{} parameters saved to {}",
                model.parameter_count(),
                out.display()
            );
        }
        Commands::Train {
            config,
            data,
            out,
            resume,
        } => {
            let config = Config::load(&config)?;
            let summary = sejong_train::train(&config, &data, &out, resume.as_deref())?;
            println!(
                "{} updates | loss {:.6} | {}",
                summary.steps,
                summary.last_loss,
                summary.model_directory.display()
            );
        }
        Commands::Generate {
            model,
            prompt,
            max_bytes,
            temperature,
            top_k,
            top_p,
            seed,
            config,
            raw,
        } => {
            let gpu = if let Some(path) = config {
                Config::load(&path)?.gpu
            } else {
                GpuConfig::default()
            };
            let mut engine = Engine::new(gpu)?;
            let model_directory = model;
            let options = Sampling {
                max_bytes,
                temperature,
                top_k,
                top_p,
                seed,
            };
            let stdout = std::io::stdout();
            if raw {
                use std::io::Write;
                let mut output = stdout.lock();
                sejong_runtime::generate_artifact(
                    &mut engine,
                    &model_directory,
                    prompt.as_bytes(),
                    &options,
                    |b| {
                        output.write_all(&[b])?;
                        Ok(())
                    },
                )?;
                output.flush()?;
                return Ok(());
            }
            let mut stream = TextStream::new(stdout.lock());
            sejong_runtime::generate_artifact(
                &mut engine,
                &model_directory,
                prompt.as_bytes(),
                &options,
                |b| stream.push(b),
            )?;
            stream.finish()?;
            println!();
        }
        Commands::Inspect { model } => {
            let value = if let Some(name) =
                ["native.json", "memory.json", "role.json", "value.json"]
                    .into_iter()
                    .find(|name| model.join(name).exists())
            {
                json::from_slice::<json::Value>(&std::fs::read(model.join(name))?)?
            } else {
                json::to_value(Manifest::load(&model)?)?
            };
            let mut value = value;
            if model.join("native.json").exists() {
                let mut engine = Engine::new(GpuConfig::default())?;
                let composite = sejong_runtime::native::Composite::load(&mut engine, &model)?;
                value["accounting"] = json::to_value(composite.accounting())?;
            }
            println!("{}", json::to_string_pretty(&value)?);
        }
    }
    Ok(())
}
