use clap::Args;
use simd_csv::ByteRecord;

use crate::utils::io::CSVInput;
use crate::utils::select::SelectedColumns;
use crate::{CLIResult, CommonArgs};

#[derive(Args, Debug)]
pub struct LangArgs {
    /// Column containing the text to classify
    column: SelectedColumns,
    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,
    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: LangArgs) -> CLIResult<()> {
    let mut reader = CSVInput::new(&args.input).csv_reader()?;
    let headers = reader.byte_headers()?;
    let column_index = args.column.single_selection(headers, true)?;

    let mut record = ByteRecord::new();

    while reader.read_byte_record(&mut record)? {
        let text = &record[column_index];

        dbg!("{}", text);
    }

    Ok(())
}
