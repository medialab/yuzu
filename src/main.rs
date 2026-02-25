use crate::commands::embed::qwen3_embed;
use clap::{Parser, Subcommand, Args};
use std::path::PathBuf;

pub mod commands;
pub mod utils;

#[derive(Debug)]
pub enum CLIError {
    Custom(String)
}

pub type CLIResult<T> = Result<T, CLIError>;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Args, Debug)]
pub struct CommonArgs {
    #[arg(short, long)]
    delimiter: Option<String>
}

#[derive(Subcommand)]
enum Commands {
    Embed { input: PathBuf },
    Lang(commands::lang::LangArgs)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Embed { input }) => qwen3_embed(&input),
        Some(Commands::Lang(args)) => commands::lang::action(args).unwrap(),
        None => {}
    }
}
