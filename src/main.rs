use std::path::PathBuf;
use clap::{Parser, Subcommand};

pub mod commands;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {

    Embed {
        input: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Embed { input }) => {
            println!("{:?}", input)

        }
        None => {}
    }
}