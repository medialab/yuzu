use clap::Args;
use simd_csv::{ByteRecord, Selection, Selector};
use phylacter::TopKHeapMap;
use ordered_float::OrderedFloat;

use crate::utils::hf::{EmbeddingModel, print_models_list};
use crate::utils::io;
use crate::{CLIResult, CommonArgs, ParallelizationArgs};


#[derive(Args, Debug)]
#[command(
    help_template = concat!("{usage-heading} yuzu search [OPTIONS] <QUERY> [INPUT]\n",
                "       yuzu search --list-models\n\n",
                "{all-args}{after-help}")
)]
pub struct SearchArgs {
    /// Query in natural language.
    query: String,

    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,

    /// If given, print a list of supported models then exit.
    #[arg(long, exclusive = true)]
    list_models: bool,

    /// CSV column containing the embedded text.
    #[arg(short = 's', long, default_value = "dim_*", value_name = "COLS")]
    embedded_column: Selector,

    /// Name of the column containing the cosine similarity.
    #[arg(long, default_value = "similarity_column", value_name = "NAME")]
    similarity_column: String,

    /// Number of most cosine-similar rows to return.
    #[arg(short = 'l', long, default_value = "10", value_name = "NUMBER")]
    limit: usize,

    /// Id of the model on HuggingFace.
    #[arg(short, default_value = "ibm-granite/granite-embedding-107m-multilingual", long)]
    model: EmbeddingModel,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,

    /// Whether to drop the embedded column.
    #[arg(short = 'D', long)]
    drop: bool,

    #[command(flatten)]
    parallelization: ParallelizationArgs,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: SearchArgs) -> CLIResult<()> {

    if args.list_models {
        print_models_list();
        return Ok(());
    }

    let threads = args.parallelization.build_rayon_global_thread_pool();

    let model = args.model;
    let mut embedder = model.embedder(threads)?;

    let query = embedder.embed(vec!(args.query))?
                                  .pop().unwrap();
    let norm_query = query.iter().fold(0 as f32, |s, &x| s+x*x);

    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    let headers: ByteRecord = reader.byte_headers()?.clone();

    let embedded_columns = reader.select(&args.embedded_column)?;

    let out_sel = if args.drop {
        embedded_columns.inverse()
    } else {
        Selection::full(headers.len())
    };

    let mut writer = io::Output::new(&args.output).csv_writer()?;

    if reader.has_headers() {
        let out_headers = [args.similarity_column.as_bytes()].into_iter().chain(out_sel.select(&headers)).collect::<ByteRecord>();
        writer.write_byte_record(&out_headers)?;
    }

    let mut record = ByteRecord::new();
    let mut heap = TopKHeapMap::<OrderedFloat<f32>, ByteRecord>::new(args.limit);

    let mut row = 0;
    while reader.read_byte_record(&mut record)? {
        row += 1;
        let embedded = embedded_columns.select(&record).enumerate().map(|(i,e)| fast_float::parse(e).unwrap_or_else({
            |err| {println!("Erreur: Unvalid float parse on row {row} for dimension {i}.\n {err}"); 0 as f32}
        })).collect::<Vec<f32>>();
        let (dot_prod, norm_embed ) = query.iter().zip(embedded.iter()).fold(
            (0 as f32, 0 as f32), |(dot, norm), (q, e)| (dot+q*e, norm+e*e)
        );
        let cos_sim = dot_prod / (norm_embed.sqrt() * norm_query.sqrt());
        heap.push_with( OrderedFloat(cos_sim), || {record.clone()});
    }

    for (cos_sim, record) in heap.into_sorted_vec() {
        let out_row = [cos_sim.to_string().as_bytes()].into_iter().chain(out_sel.select(&record)).collect::<ByteRecord>();
        writer.write_byte_record(&out_row)?;
    }

    Ok(())
}