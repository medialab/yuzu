use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::from_utf8;

use clap::Args;
use simd_csv::{ByteRecord, Selection, Selector};
use tokenizers::{Encoding, PaddingDirection};

use crate::utils::hf::{DEFAULT_MODEL, EmbeddingModel, print_models_list};
use crate::utils::{io, iter::IteratorExt};
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

#[derive(Debug, Clone)]
enum Operation {
    Tokens,
    Explode,
    Counts,
    Fits
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

    /// The results wanted from the tokenization. Wether it's 'tokens' (tokenized column),
    /// 'counts' (number of tokens), 'fits' (fits in the context window of the model).
    /// To have multiple outputs at once, separate your options with a comma (i.e. tokens,counts)
    #[arg(short, long, default_value = "tokens")]
    emit: String,

    /// Csv-separable column names. Must be aligned with --emit. Will take sensible defaults.
    /// (i.e. tokens, counts, and/or fits depending on the options requested).
    #[arg(short, long)]
    column_names: Option<String>,

    /// If given, "explode" the output by priting a copy of the record per token. This can be
    /// useful to compute aggregation at the token level. Cannot be used with --count nor --fits
    /// nor --keep-tokenized.
    #[arg(long)]
    explode: bool,

    /// csv-separable column names. Must be aligned with --emit and would take sensible defaults.
    /// (i.e. "tokens", "count", and/or "fits" depending on those requested).
    #[arg(long, default_value = "token")]
    explode_name: String,

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

pub fn action(args: TokenizeArgs) -> CLIResult<()> {
    if args.list_models {
        print_models_list();
        return Ok(());
    }

    let mut operations = Vec::new();
    let mut col_names: Vec<Vec<u8>> = Vec::new();

    let emit_ops = args.emit.split(",");
    if let Some(column_names) = args.column_names {
        let names = column_names.split(",");
        if names.clone().count() != emit_ops.clone().count() {
            Err("Number of column-names doesn't correspond to the number of columns emitted")?;
        }

        for (op, name) in emit_ops.zip(names) {
            match op {
                "tokens" => if !args.explode { operations.push(Operation::Tokens); col_names.push(name.as_bytes().to_vec()) }
                            else { operations.push(Operation::Explode); col_names.push(args.explode_name.as_bytes().to_vec()) },
                "counts" => { operations.push(Operation::Counts); col_names.push(name.as_bytes().to_vec()); }
                "fits" => { operations.push(Operation::Fits); col_names.push(name.as_bytes().to_vec()); },
                _ => Err("--emit did not receive a correct format (i.e. 'tokens,counts,fits' or less options)" )?,
            }
        }
    } else {
        for op in emit_ops {
            match op {
                "tokens" => if !args.explode { operations.push(Operation::Tokens); col_names.push(b"tokens".to_vec()) }
                            else { operations.push(Operation::Explode); col_names.push(args.explode_name.as_bytes().to_vec()) },
                "counts" => { operations.push(Operation::Counts); col_names.push(b"counts".to_vec()); }
                "fits" => { operations.push(Operation::Fits); col_names.push(b"fits".to_vec()); },
                _ => Err("--emit did not receive a correct format (i.e. 'tokens,counts,fits' or less options)" )?,
            }
        }
    }

    args.parallelization.build_rayon_global_thread_pool();

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
        let mut new_headers: ByteRecord = out_sel.select(&headers).collect();
        for name in col_names {
            new_headers.push_field(&name);
        }

        writer.write_byte_record(&new_headers)?;
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

            for operation in &operations {
                match operation {
                    Operation::Tokens => record.push_field(tokens.join(&args.sep).as_bytes()),
                    Operation::Counts => record.fmt_field(&tokens.len()),
                    Operation::Fits => record.push_field(if tokens.len() < args.model.max_length
                                                { "true".as_bytes() } else { "false".as_bytes() }),
                    Operation::Explode => for token in tokens {
                        record.truncate(out_sel.len());
                        record.push_field(token.as_bytes());
                    }
                }
            }
            writer.write_byte_record(record)?;
        }
    }

    Ok(())
}
