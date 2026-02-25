use clap::{Args, Parser, Subcommand};

pub mod commands;
pub mod utils;

#[derive(Debug)]
pub enum CLIError {
    Custom(String),
}

pub type CLIResult<T> = Result<T, CLIError>;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct YuzuArgs {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Args, Debug)]
pub struct CommonArgs {
    #[arg(short, long)]
    delimiter: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    Embed(commands::embed::EmbedArgs),
    Lang(commands::lang::LangArgs),
}

fn main() {
    let args = YuzuArgs::parse();

    let result = match args.command {
        Some(Commands::Embed(args)) => commands::embed::action(args),
        Some(Commands::Lang(args)) => commands::lang::action(args),
        None => Ok(()),
    };

    if let Err(error) = result {
        match error {
            CLIError::Custom(msg) => {
                eprintln!("{}", msg);
            }
        }
    }
}
