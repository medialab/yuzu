use crate::commands::embed::qwen3_embed;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod commands;
pub mod utils;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Embed { input: PathBuf },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Embed { input }) => qwen3_embed(input),
        None => {}
    }
}
