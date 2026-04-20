use clap::Args;
use simd_csv::ByteRecord;
use whichlang::detect_language;

use crate::utils::io::{Input, Output};
use crate::utils::select::SelectedColumns;
use crate::{CLIResult, CommonArgs};

#[derive(Args, Debug)]
pub struct LangArgs {
    /// Column containing the text to classify
    column: SelectedColumns,
    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,
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

    headers.push_field(b"lang");

    writer.write_byte_record(&headers)?;

    let mut record = ByteRecord::new();

    while reader.read_byte_record(&mut record)? {
        let text = &record[column_index];
        let lang = detect_language(std::str::from_utf8(text).expect("Could not decode utf-8!"));

        record.push_field(lang.three_letter_code().as_bytes());
        writer.write_byte_record(&record)?;
    }

    Ok(())
}
