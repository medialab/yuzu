use std::str::from_utf8;

use clap::Args;
use simd_csv::ByteRecord;
use whichlang::detect_language;

use crate::utils::io::{Input, Output};
use crate::utils::select::SelectedColumns;
use crate::{CLIResult, CommonArgs};

#[derive(Args, Debug)]
pub struct LangArgs {
    /// Column containing text to classify
    column: SelectedColumns,
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
    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: LangArgs) -> CLIResult<()> {
    let mut reader = Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .csv_reader()?;

    let mut headers = reader.byte_headers()?.clone();
    let column_index = args.column.single_selection(&headers, true)?;

    let mut writer = Output::new(&args.output).csv_writer()?;

    headers.push_field(args.lang_column.as_bytes());

    writer.write_byte_record(&headers)?;

    let mut record = ByteRecord::new();

    while reader.read_byte_record(&mut record)? {
        let text = &record[column_index];
        let lang_opt = detect_language(from_utf8(text)?);

        let cell = if let Some(lang) = lang_opt {
            if args.full_name {
                lang.eng_name()
            } else {
                lang.three_letter_code()
            }
        } else {
            &args.default
        };

        record.push_field(cell.as_bytes());
        writer.write_byte_record(&record)?;
    }

    Ok(())
}
