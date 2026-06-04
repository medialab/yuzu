use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::from_utf8;

use clap::Args;
use simd_csv::{ByteRecord, Selection, Selector};
use tokenizers::{Encoding, PaddingDirection};

use crate::utils::hf::{EmbeddingModel, print_models_list};
use crate::utils::{io, iter::IteratorExt};
use crate::{CLIResult, CommonArgs};

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

    /// Id of the model on HuggingFace. Defaults to ibm-granite/granite-embedding-107m-multilingual.
    #[arg(short, long)]
    model: Option<EmbeddingModel>,

    /// Chunk size in number of rows.
    #[arg(long, default_value = "32")]
    chunk_size: NonZeroUsize,

    /// If given, will count the number of tokens instead. Cannot be used with --explode.
    #[arg(long)]
    count: bool,

    /// Name of the column to append. Defaults to "tokens", or "token" when used with --explode
    /// or "token_count" when used with --count.
    #[arg(short, long)]
    column: Option<String>,

    /// Separator to use when joining the tokens in the output. The default space character used to
    /// do so is usually safe since most models normalize spaces in the tokens for easier debugging.
    #[arg(long, default_value = " ")]
    sep: String,

    /// Whether to keep the tokenized column.
    #[arg(short, long)]
    keep: bool,

    /// If given, "explode" the output by priting a copy of the record per token. This can be
    /// useful to compute aggregation at the token level. Cannot be used with --count.
    #[arg(long)]
    explode: bool,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: TokenizeArgs) -> CLIResult<()> {
    if args.explode && args.count {
        Err("--count is not compatible with --explode!")?;
    }

    if args.list_models {
        print_models_list();
        return Ok(());
    }

    let model = args.model.unwrap_or_default();
    let tokenizer_path = model.tokenizer_path()?;
    let tokenizer = model.tokenizer(&tokenizer_path)?;

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

        let new_column_name = match &args.column {
            Some(name) => name.as_bytes(),
            None => {
                if args.count {
                    &b"token_count"[..]
                } else if args.explode {
                    &b"token"[..]
                } else {
                    &b"tokens"[..]
                }
            }
        };

        new_headers.push_field(new_column_name);

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
            let tokens = &encoding.get_tokens()[unpadded_range(&encoding, model.padding_direction)];

            if args.explode {
                for token in tokens {
                    record.truncate(out_sel.len());
                    record.push_field(token.as_bytes());

                    writer.write_byte_record(record)?;
                }
            } else {
                if args.count {
                    record.fmt_field(&tokens.len());
                } else {
                    record.push_field(tokens.join(&args.sep).as_bytes());
                }

                writer.write_byte_record(record)?;
            }
        }
    }

    Ok(())
}
