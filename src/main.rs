use std::io;
use std::process;
use std::str::Utf8Error;

use clap::{Args, Parser, Subcommand};

pub mod commands;
pub mod utils;

#[derive(Debug)]
pub enum CLIError {
    Io(io::Error),
    Custom(String),
}

impl From<String> for CLIError {
    fn from(value: String) -> Self {
        Self::Custom(value)
    }
}

impl From<io::Error> for CLIError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<Utf8Error> for CLIError {
    fn from(value: Utf8Error) -> Self {
        Self::Custom(value.to_string())
    }
}

impl From<simd_csv::Error> for CLIError {
    fn from(value: simd_csv::Error) -> Self {
        if !value.is_io_error() {
            Self::Custom(value.to_string())
        } else {
            match value.into_kind() {
                simd_csv::ErrorKind::Io(inner) => Self::Io(inner),
                _ => unreachable!(),
            }
        }
    }
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
    /// When set, first row of CSV input will not be interpreted as a header.
    #[arg(short, long)]
    no_headers: bool,
    /// The field delimiter for reading CSV data. Must be a single character.
    /// Will default to a comma.
    #[arg(short, long)]
    delimiter: Option<utils::io::Delimiter>,
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
                process::exit(1);
            }
            CLIError::Io(err) if err.kind() == io::ErrorKind::BrokenPipe => {
                process::exit(0);
            }
            CLIError::Io(err) => {
                eprintln!("{}", err);
                process::exit(1);
            }
        }
    }
}
