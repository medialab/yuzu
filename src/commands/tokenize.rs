use std::ops::Range;
use std::str::from_utf8;

use clap::Args;
use simd_csv::Selector;
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

    /// Separator to use when joining the tokens in the output. The default space character used to
    /// do so is usually safe since most models normalize spaces in the tokens for easier debugging.
    #[arg(long, default_value = " ")]
    sep: String,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: TokenizeArgs) -> CLIResult<()> {
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

    let text_column_index = reader.select_one(args.text_column.as_ref().unwrap())?;
    let mut writer = io::Output::new(&args.output).csv_writer()?;

    if reader.has_headers() {
        let mut new_headers = reader.byte_headers()?.clone();
        new_headers.push_field(b"tokens");

        writer.write_byte_record(&new_headers)?;
    }

    let mut records = Vec::new();

    for chunk in reader.into_byte_records().chunks(32) {
        records.clear();

        let mut texts = Vec::with_capacity(chunk.len());

        for result in chunk {
            let record = result?;
            texts.push(from_utf8(&record[text_column_index])?.to_string());
            records.push(record);
        }

        let encodings = tokenizer.encode_batch(texts, true)?;

        for (record, encoding) in records.iter_mut().zip(encodings.into_iter()) {
            let tokens = &encoding.get_tokens()[unpadded_range(&encoding, model.padding_direction)];

            record.push_field(tokens.join(&args.sep).as_bytes());

            writer.write_byte_record(record)?;
        }
    }

    Ok(())
}
