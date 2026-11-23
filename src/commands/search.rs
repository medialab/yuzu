use clap::Args;
use ordered_float::OrderedFloat;
use phylacter::TopKHeapMap;
use simd_csv::{ByteRecord, Selection, Selector};

use crate::utils::hf::{DEFAULT_MODEL, EmbeddingModel, print_models_list};
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
    #[arg(short, long, default_value = "dim_*", value_name = "cols")]
    select: Selector,

    /// Name of the column containing the cosine similarity.
    #[arg(long, default_value = "similarity_column", value_name = "name")]
    similarity_column: String,

    /// Number of most cosine-similar rows to return.
    #[arg(short = 'l', long, default_value = "10", value_name = "n")]
    limit: usize,

    /// Id of the model on HuggingFace.
    #[arg(
        short, long,
        default_value = DEFAULT_MODEL,
        value_name = "model"
    )]
    model: EmbeddingModel,

    /// Path to output file. Will write to stdout if not given or if path is "-".
    #[arg(short, long, value_name = "path")]
    output: Option<String>,

    /// Whether to drop the embedded column.
    #[arg(short, long)]
    keep: bool,

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

    let query = embedder.embed(vec![args.query])?.pop().unwrap();
    let norm_query = query.iter().fold(0.0_f32, |s, &x| s + x * x).sqrt();

    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    let headers: ByteRecord = reader.byte_headers()?.clone();

    let embedded_columns = reader.select(&args.select)?;

    let out_sel = if args.keep {
        Selection::full(headers.len())
    } else {
        embedded_columns.inverse()
    };

    let mut writer = io::Output::new(&args.output).csv_writer()?;

    if reader.has_headers() {
        let out_headers = [args.similarity_column.as_bytes()]
            .into_iter()
            .chain(out_sel.select(&headers))
            .collect::<ByteRecord>();
        writer.write_byte_record(&out_headers)?;
    }

    let mut record = ByteRecord::new();
    let mut heap = TopKHeapMap::<OrderedFloat<f32>, ByteRecord>::new(args.limit);

    let mut row = 0;
    let mut embedded: Vec<f32> = Vec::with_capacity(embedded_columns.len());
    while reader.read_byte_record(&mut record)? {
        for (i, e) in embedded_columns.select(&record).enumerate() {
            embedded.push(fast_float::parse(e).map_err(|err| {
                format!("Erreur: Invalid float parse on row {row} for dimension {i}.\n {err}")
            })?);
        }
        let (dot_prod, squared_norm_embed) = query
            .iter()
            .zip(embedded.iter())
            .fold((0 as f32, 0 as f32), |(dot, norm), (q, e)| {
                (dot + q * e, norm + e * e)
            });
        let cos_sim = dot_prod / squared_norm_embed.sqrt();
        heap.push_with(OrderedFloat(cos_sim), || {
            out_sel.select(&record).collect::<ByteRecord>()
        });
        embedded.clear();
        row += 1;
    }

    for (cos_sim, record) in heap.into_sorted_vec() {
        let out_row = [(cos_sim / norm_query).to_string().as_bytes()]
            .into_iter()
            .chain(&record)
            .collect::<ByteRecord>();
        writer.write_byte_record(&out_row)?;
    }

    Ok(())
}
