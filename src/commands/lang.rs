use std::str::from_utf8;

use clap::Args;
use pariter::IteratorExt;
use simd_csv::{ByteRecord, Selector};
use whichlang::detect_language;
use paltoquet::tokenizers::{WordTokenizer, WordTokenKind};

use crate::utils::io::{Input, Output};
use crate::utils::iter::IteratorExt as _;
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

#[derive(Args, Debug, Clone)]
pub struct LangArgs {
    /// Column containing text to classify
    column: Selector,

    /// Path to input CSV file (will use stdin if not given or if path is "-").
    input: Option<String>,

    /// Whether to emit full English name of detected lang instead of ISO-639-3 code.
    #[arg(long)]
    full_name: bool,

    /// Name of the added column containing detected lang.
    #[arg(long, default_value = "lang")]
    lang_column: String,

    /// Default value to use when lang cannot be detected.
    #[arg(long, default_value = "")]
    default: String,

    /// Path to output file. Will write to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,

    /// Path to output file. Will write to stdout if not given or if path is "-".
    #[arg(long)]
    filter_ease: bool,

    #[command(flatten)]
    parallelization: ParallelizationArgs,

    #[command(flatten)]
    common: CommonArgs,
}

impl LangArgs {
    fn process_record(&self, record: &mut ByteRecord, column_index: usize, word_tokenizer: Option<&WordTokenizer>) -> CLIResult<()> {
        // let text = if let Some(tokenizer) = word_tokenizer {
        //     &(tokenizer
        //         .tokenize(from_utf8(&record[column_index])?)
        //         .map(|token| { if !token.is_junk() {token.text} else {""}})
        //         .collect::<Vec<_>>().join(" "))
        let text = if let Some(tokenizer) = word_tokenizer {
            &(tokenizer
                .tokenize(from_utf8(&record[column_index])?)
                .filter_map(|token| { 
                    let (tok, tok_kind) = token.to_pair();
                    (!token.is_junk() && tok_kind == WordTokenKind::Word).then_some(tok)})
                .collect::<Vec<_>>()
                .join(" "))
        } else {
            from_utf8(&record[column_index])?
        };

        let lang_opt = detect_language(text.as_ref());
        
        let cell = if let Some(lang) = lang_opt {
            if self.full_name {
                lang.eng_name()
            } else {
                lang.three_letter_code()
            }
        } else {
            &self.default
        };

        record.push_field(cell.as_bytes());

        Ok(())
    }
}

pub fn action(args: LangArgs) -> CLIResult<()> {
    let mut reader = Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    let mut headers = reader.byte_headers()?.clone();
    let column_index = reader.select_one(&args.column)?;

    let mut writer = Output::new(&args.output).csv_writer()?;

    if reader.has_headers() {
        headers.push_field(args.lang_column.as_bytes());
        writer.write_byte_record(&headers)?;
    }

    let tokenizer = args.filter_ease.then(WordTokenizer::new);

    if let Some(t) = args.parallelization.threads() {
        for result in reader.into_byte_records().chunks(64).parallel_map_custom(
            |o| o.threads(t),
            move |results| -> CLIResult<Vec<ByteRecord>> {
                results
                    .into_iter()
                    .map(|result| -> CLIResult<ByteRecord> {
                        let mut record = result?;
                        args.process_record(&mut record, column_index, tokenizer.as_ref())?;
                        Ok(record)
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        ) {
            for record in result? {
                writer.write_byte_record(&record)?;
            }
        }
    } else {
        let mut record = ByteRecord::new();

        while reader.read_byte_record(&mut record)? {
            args.process_record(&mut record, column_index, tokenizer.as_ref())?;
            writer.write_byte_record(&record)?;
        }
    }

    Ok(writer.flush()?)
}
