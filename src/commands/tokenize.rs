use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::{FromStr, from_utf8};

use clap::Args;
use simd_csv::{ByteRecord, Selection, Selector};
use tokenizers::{Encoding, PaddingDirection};

use crate::utils::hf::{DEFAULT_MODEL, EmbeddingModel, print_models_list};
use crate::utils::{
    io::{self, CsvNames},
    iter::IteratorExt,
};
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

#[derive(Debug, Clone)]
enum Operation {
    Tokens,
    Counts,
    Fits,
}

impl Operation {
    fn as_column_name(&self, plural: bool) -> &[u8] {
        match self {
            Self::Tokens => {
                if plural {
                    b"tokens"
                } else {
                    b"token"
                }
            }
            Self::Counts => b"token_count",
            Self::Fits => b"fits",
        }
    }
}

impl FromStr for Operation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "token" | "tokens" => Self::Tokens,
            "count" | "counts" | "token_count" => Self::Counts,
            "fits" => Self::Fits,
            _ => return Err(format!("unknown --emit type {s}")),
        })
    }
}

#[inline]
fn unpadded_range(encoding: &Encoding, direction: PaddingDirection) -> Range<usize> {
    match direction {
        PaddingDirection::Right => {
            let unpadded_len = encoding
                .get_attention_mask()
                .iter()
                .position(|m| *m == 0)
                .unwrap_or(encoding.len());

            0..unpadded_len
        }
        PaddingDirection::Left => {
            let first_unpadded_pos = encoding
                .get_attention_mask()
                .iter()
                .position(|m| *m == 1)
                .unwrap_or(encoding.len());

            first_unpadded_pos..encoding.len()
        }
    }
}

#[derive(Args, Debug)]
pub struct TokenizeArgs {
    /// CSV column containing the text to tokenize
    #[arg(
        required_unless_present = "list_models",
        conflicts_with = "list_models"
    )]
    text_column: Option<Selector>,

    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,

    /// If given, print a list of supported models then exit
    #[arg(long)]
    list_models: bool,

    /// Id of the model on HuggingFace.
    #[arg(
        short, long,
        default_value = DEFAULT_MODEL,
        value_name = "model"
    )]
    model: EmbeddingModel,

    /// Chunk size in number of rows.
    #[arg(long, default_value = "32", value_name = "n")]
    chunk_size: NonZeroUsize,

    /// Columns, separated by commas, that the tokenization process should emit:
    ///     - "tokens": tokens joined by spaces or by what was given to --sep
    ///                 or a single token when using --explode.
    ///     - "count":  number of tokens
    ///     - "fits":   whether the number of tokens fits in the model's context
    ///                 window.
    #[arg(
        short,
        long,
        default_value = "tokens",
        value_delimiter = ',',
        verbatim_doc_comment
    )]
    emit: Vec<Operation>,

    /// Csv-separable column names. Must be aligned with --emit. Will take sensible defaults.
    /// (i.e. token(s), token_count, and/or fits depending on the options requested).
    #[arg(short, long)]
    column_names: Option<CsvNames>,

    /// If given, "explode" the output by priting a copy of the record per token. This can be
    /// useful to compute aggregation at the token level.
    #[arg(long)]
    explode: bool,

    /// Separator to use when joining the tokens in the output. The default space character used to
    /// do so is usually safe since most models normalize spaces in the tokens for easier debugging.
    #[arg(long, default_value = " ", value_name = "text")]
    sep: String,

    /// Whether to keep the original column.
    #[arg(short, long)]
    keep: bool,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long, value_name = "path")]
    output: Option<String>,

    #[command(flatten)]
    parallelization: ParallelizationArgs,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(mut args: TokenizeArgs) -> CLIResult<()> {
    if args.list_models {
        print_models_list();
        return Ok(());
    }

    let col_names = args
        .column_names
        .take()
        .map(CsvNames::into_record)
        .unwrap_or_else(|| {
            args.emit
                .iter()
                .map(|op| op.as_column_name(!args.explode))
                .collect()
        });

    if args.emit.len() != col_names.len() {
        Err(format!(
            "--emit ({}) and --column-names ({}) don't have the same number of elements!",
            args.emit.len(),
            col_names.len()
        ))?;
    }

    let tokenizer = args.model.tokenizer()?;

    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    let headers = reader.byte_headers()?.clone();

    let text_column_index = reader.select_one(args.text_column.as_ref().unwrap())?;

    let out_sel = if args.keep {
        Selection::full(headers.len())
    } else {
        Selection::without(text_column_index, headers.len())
    };

    let mut writer = io::Output::new(&args.output).csv_writer()?;

    if reader.has_headers() {
        writer.write_record(out_sel.select(&headers).chain(&col_names))?;
    }

    let mut records = Vec::new();

    for chunk in reader.into_byte_records().chunks(args.chunk_size.get()) {
        records.clear();

        let mut texts = Vec::with_capacity(chunk.len());

        for result in chunk {
            let record = result?;
            texts.push(from_utf8(&record[text_column_index])?.to_string());
            records.push(out_sel.select(&record).collect::<ByteRecord>());
        }

        let encodings = tokenizer.encode_batch(texts, true)?;

        for (record, encoding) in records.iter_mut().zip(encodings.into_iter()) {
            let tokens =
                &encoding.get_tokens()[unpadded_range(&encoding, args.model.padding_direction)];

            if args.explode {
                for token in tokens {
                    record.truncate(out_sel.len());

                    for operation in &args.emit {
                        match operation {
                            Operation::Tokens => record.push_field(token.as_bytes()),
                            Operation::Counts => record.fmt_field(&tokens.len()),
                            Operation::Fits => {
                                record.push_field(if tokens.len() < args.model.max_length {
                                    b"true"
                                } else {
                                    b"false"
                                })
                            }
                        }
                    }
                    writer.write_byte_record(record)?;
                }
            } else {
                for operation in &args.emit {
                    match operation {
                        Operation::Tokens => record.push_field(tokens.join(&args.sep).as_bytes()),
                        Operation::Counts => record.fmt_field(&tokens.len()),
                        Operation::Fits => {
                            record.push_field(if tokens.len() < args.model.max_length {
                                b"true"
                            } else {
                                b"false"
                            })
                        }
                    }
                }
                writer.write_byte_record(record)?;
            }
        }
    }

    Ok(())
}
