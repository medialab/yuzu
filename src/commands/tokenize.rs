use std::num::NonZeroUsize;
use std::ops::Range;
use std::str::from_utf8;

use clap::Args;
use simd_csv::{ByteRecord, Selection, Selector};
use tokenizers::{Encoding, PaddingDirection};

use crate::utils::hf::{DEFAULT_MODEL, EmbeddingModel, print_models_list};
use crate::utils::{io, iter::IteratorExt};
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

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

const COUNTS_HEADER: &str = "Count Tokens Options";

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

    /// If given, "explode" the output by priting a copy of the record per token. This can be
    /// useful to compute aggregation at the token level. Cannot be used with --count nor --fits
    /// nor --keep-tokenized.
    #[arg(long)]
    explode: bool,

    /// Name of the token column to append. Defaults to "tokens", or "token" when used with --explode.
    #[arg(long, value_name = "name")]
    token_column: Option<String>,

    /// If given, will count the number of tokens instead. Cannot be used with --explode nor --fits.
    /// --keep will be automatically set to true.
    #[arg(long, help_heading = COUNTS_HEADER)]
    count: bool,

    /// Used with --counts. Name of the column to append the result of --counts.
    #[arg(short, long,
        value_name = "name",
        help_heading = COUNTS_HEADER)]
    count_column: Option<String>,

    /// If given, will display a boolean indicating whether the number of tokens fits
    /// in the chosen model's context window. Cannot be used with --explode nor --count.
    /// --keep will be automatically set to true.
    #[arg(long, help_heading = COUNTS_HEADER)]
    fits: bool,

    /// Used with --fits. Name of the column to append the result of --fits.
    #[arg(short, long, value_name = "name", help_heading = COUNTS_HEADER)]
    fits_column: Option<String>,

    /// Whether to keep the tokenized column.
    #[arg(long, help_heading = COUNTS_HEADER)]
    keep_tokenized: bool,

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
    if args.explode{
        if args.count {
            Err("--count is not compatible with --explode!")?;
        }
        if args.fits {
            Err("--fits is not compatible with --explode!")?;
        }
        if args.keep_tokenized {
            Err("Cannot keep original tokenized column if using --explode!")?;
        }
    }

    if !args.count && args.count_column.is_some() {
        Err("--count-column have to be used with --count!")?;
    }

    if !args.fits && args.fits_column.is_some() {
        Err("--fits-column have to be used with --fits!")?;
    }

    if args.list_models {
        print_models_list();
        return Ok(());
    }

    let have_tokens_col = !(args.count || args.fits) || args.keep_tokenized ;

    // !have_tokens_col -> !explode because (count || fits) -> !explode
    if !have_tokens_col && args.token_column.is_some() {
        Err(concat!["You didn't ask to keep tokenized column but still gave it a name!",
                    "Maybe you forgot --keep-tokenized?"])?;
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

        let mut new_columns_names: Vec<&[u8]> = Vec::with_capacity(2);
        macro_rules! push_column_name_or {
            ($token_column:expr, $default:expr) => {
                new_columns_names.push(
                    $token_column
                        .as_ref()
                        .map_or($default, |name| name.as_bytes()),
                );
            };
        }

        if args.explode {
            push_column_name_or!(args.token_column, &b"token"[..]);
        } else {
            if have_tokens_col { push_column_name_or!(args.token_column, &b"tokens"[..]); }
            if args.count { push_column_name_or!(args.count_column, &b"token_count"[..]); }
            if args.fits { push_column_name_or!(args.fits_column, &b"fits"[..]); }
        }

        for field in new_columns_names { new_headers.push_field(field); }

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

            if args.explode {
                for token in tokens {
                    record.truncate(out_sel.len());
                    record.push_field(token.as_bytes());

                    writer.write_byte_record(record)?;
                }
            } else {
                if have_tokens_col {
                    record.push_field(tokens.join(&args.sep).as_bytes());
                }
                if args.count {
                    record.fmt_field(&tokens.len());
                }
                if args.fits {
                    record.push_field(if tokens.len() < args.model.max_length {
                        "true".as_bytes()
                    } else {
                        "false".as_bytes()
                    });
                }

                writer.write_byte_record(record)?;
            }
        }
    }

    Ok(())
}
