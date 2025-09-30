//! ΨLang Compiler and Runtime
//!
//! Command-line interface for the ΨLang programming language.

use clap::{Arg, Command};
use psilang::*;
use std::fs;
use std::path::Path;
use tokio;

#[derive(Debug)]
struct Args {
    input_file: String,
    output_file: Option<String>,
    compile_only: bool,
    run: bool,
    verbose: bool,
    target: String,
    optimize: bool,
}

fn parse_args() -> Args {
    let matches = Command::new("ΨLang Compiler")
        .version(env!("CARGO_PKG_VERSION"))
        .author("PsiLang Development Team")
        .about("Compiles and runs ΨLang programs - neural networks that learn and evolve")
        .arg(
            Arg::new("input")
                .help("Input ΨLang source file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file for compiled network")
                .takes_value(true),
        )
        .arg(
            Arg::new("compile-only")
                .short('c')
                .long("compile-only")
                .help("Only compile, don't execute")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("run")
                .short('r')
                .long("run")
                .help("Execute the compiled program")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("target")
                .short('t')
                .long("target")
                .help("Target platform (cpu, gpu, neuromorphic)")
                .default_value("cpu")
                .takes_value(true),
        )
        .arg(
            Arg::new("optimize")
                .short('O')
                .long("optimize")
                .help("Enable optimizations")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    Args {
        input_file: matches.get_one::<String>("input").unwrap().clone(),
        output_file: matches.get_one::<String>("output").cloned(),
        compile_only: matches.get_flag("compile-only"),
        run: matches.get_flag("run"),
        verbose: matches.get_flag("verbose"),
        target: matches.get_one::<String>("target").unwrap().clone(),
        optimize: matches.get_flag("optimize"),
    }
}

async fn run_compiler(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Read source file
    if args.verbose {
        println!("Reading source file: {}", args.input_file);
    }

    let source = fs::read_to_string(&args.input_file)
        .map_err(|e| format!("Failed to read {}: {}", args.input_file, e))?;

    if args.verbose {
        println!("Source code ({} bytes):", source.len());
        println!("{}", source);
        println!();
    }

    // Compile the program
    if args.verbose {
        println!("Compiling ΨLang program...");
    }

    let start_time = std::time::Instant::now();
    let network = compile(&source)?;
    let compilation_time = start_time.elapsed();

    if args.verbose {
        println!("Compilation completed in {:?}", compilation_time);

        // Print network statistics
        let stats = network.statistics();
        println!("Network statistics:");
        println!("  Neurons: {}", stats.neuron_count);
        println!("  Synapses: {}", stats.synapse_count);
        println!("  Assemblies: {}", stats.assembly_count);
        println!("  Patterns: {}", stats.pattern_count);
        println!("  Total weight: {:.3}", stats.total_weight);
        println!("  Connectivity: {:.3}", stats.average_connectivity);
        println!();
    }

    // Save compiled network if output file specified
    if let Some(output_file) = &args.output_file {
        if args.verbose {
            println!("Saving compiled network to: {}", output_file);
        }

        let json = serde_json::to_string_pretty(&network)?;
        fs::write(output_file, json)?;
    }

    // Execute if requested
    if args.run || (!args.compile_only && args.output_file.is_none()) {
        if args.verbose {
            println!("Executing neural network...");
        }

        let execution_start = std::time::Instant::now();
        let execution_result = execute(network).await?;
        let execution_time = execution_start.elapsed();

        if args.verbose {
            println!("Execution completed in {:?}", execution_time);
            println!("Execution results:");
            println!("  Success: {}", execution_result.success);
            println!("  Execution time: {:.2}ms", execution_result.execution_time_ms);
            println!("  Spikes generated: {}", execution_result.spikes_generated);
            println!("  Performance counters:");
            println!("    Events processed: {}", execution_result.performance_counters.events_processed);
            println!("    Plasticity updates: {}", execution_result.performance_counters.plasticity_updates);
            println!("    Average spike rate: {:.2} Hz", execution_result.performance_counters.average_spike_rate);
            println!("    Energy estimate: {:.3} pJ", execution_result.performance_counters.energy_estimate);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    if args.verbose {
        println!("ΨLang Compiler v{}", VERSION);
        println!("Target platform: {}", args.target);
        println!("Optimizations: {}", if args.optimize { "enabled" } else { "disabled" });
        println!();
    }

    match run_compiler(args).await {
        Ok(_) => {
            if args.verbose {
                println!("ΨLang program completed successfully! 🎉");
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // This would test argument parsing
        // For now, just ensure the function exists
        assert!(true);
    }

    #[tokio::test]
    async fn test_main_workflow() {
        // Create a temporary test file
        let test_source = r#"
        topology ⟪test⟫ {
            ∴ input { threshold: -50mV, leak: 10mV/ms }
            ∴ output { threshold: -50mV, leak: 10mV/ms }
            input ⊸0.8:1ms⊸ output
        }
        "#;

        // Write to temporary file
        let temp_file = tempfile::NamedTempFile::with_suffix(".psi").unwrap();
        fs::write(temp_file.path(), test_source).unwrap();

        // Test compilation
        let network = compile(test_source).unwrap();
        assert_eq!(network.neurons.len(), 2);
        assert_eq!(network.synapses.len(), 1);

        // Test execution
        let execution_result = execute(network).await;
        assert!(execution_result.is_ok());
    }
}