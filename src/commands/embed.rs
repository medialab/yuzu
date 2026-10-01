use std::num::NonZeroUsize;
use std::path::Path;
use std::time::SystemTime;

use clap::Args;

use rayon::prelude::*;
use simd_csv::{ByteRecord, Selector};

use crate::utils::hf::{EmbeddingModel, print_models_list};
use crate::utils::io::{DynamicUsize, Input, Output};
use crate::utils::iter::IteratorExt;
use crate::utils::readers::ReaderExt;
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

#[derive(Args, Debug)]
pub struct EmbedArgs {
    /// CSV column containing the text to embed
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

    /// Chunk size in number of rows. Rows in the same chunk are encoded simultaneously.
    #[arg(long, default_value = "16")]
    chunk_size: NonZeroUsize,

    /// Batch size in number of rows. Rows in the same batch are loaded in memory together and sorted on text length.
    #[arg(long, default_value = "2048", allow_hyphen_values = true)]
    batch_size: DynamicUsize,

    /// Whether to resume from an aborted run. Requires -o/--output to be given.
    #[arg(long, requires = "output")]
    resume: bool,

    /// Whether to print information about the embedding process in stderr
    #[arg(short, long)]
    verbose: bool,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,

    #[command(flatten)]
    parallelization: ParallelizationArgs,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: EmbedArgs) -> CLIResult<()> {
    if args.list_models {
        print_models_list();
        return Ok(());
    }

    if let DynamicUsize::Limited(size) = args.batch_size {
        if args.chunk_size > size {
            Err("--chunk-size should be smaller than --batch-size")?;
        }
    }

    let threads = args.parallelization.build_rayon_global_thread_pool();

    let rows_to_skip_when_resuming = if args.resume {
        let output_path = args.output.clone().unwrap();

        if Path::new(&output_path).is_file() {
            Some(
                Input::new(&Some(output_path))
                    .csv_splitter()?
                    .count_records()?,
            )
        } else {
            None
        }
    } else {
        None
    };

    let mut reader = Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    if let Some(skip) = rows_to_skip_when_resuming {
        reader.skip(skip)?;
    }

    let text_column_index = reader.select_one(args.text_column.as_ref().unwrap())?;
    let output = Output::maybe_resume(&args.output, args.resume)?;
    let model = args.model.unwrap_or_default();
    let mut writer = output.vector_writer(model.dim)?;

    let mut embedder = model.embedder(threads)?;

    if !output.can_resume && reader.has_headers() {
        writer.write_headers(reader.byte_headers()?, model.dim, "dim_")?;
    }

    let default_batch_len = args.batch_size.as_usize().unwrap_or(1024);

    let mut input_batch: Vec<String> = Vec::with_capacity(default_batch_len);
    let mut records: Vec<ByteRecord> = Vec::with_capacity(default_batch_len);
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(default_batch_len);

    for batch in reader.into_byte_records().chunks_or_total(args.batch_size) {
        input_batch.clear();
        records.clear();
        embeddings.clear();

        for row in batch.into_iter() {
            let record = row?;
            let string = String::from_utf8(record[text_column_index].to_vec())?;
            input_batch.push(string);
            records.push(record);
            embeddings.push(Vec::new());
        }

        let mut sorted_indices = (0..input_batch.len()).collect::<Vec<_>>();

        if threads > 1 {
            sorted_indices.par_sort_unstable_by_key(|&i| input_batch[i].len());
        } else {
            sorted_indices.sort_unstable_by_key(|&i| input_batch[i].len());
        }

        for idx_chunk in sorted_indices.chunks(args.chunk_size.get()) {
            let timer_opt = args.verbose.then(SystemTime::now);

            let input: Vec<&str> = idx_chunk.iter().map(|&i| input_batch[i].as_str()).collect();
            let mut embedding = embedder.embed(input)?;

            for (&i, e) in idx_chunk.iter().zip(embedding.iter_mut()) {
                std::mem::swap(&mut embeddings[i], e);
            }

            if let Some(timer) = timer_opt {
                eprintln!(
                    "Batch ({}) took {:?}",
                    args.chunk_size,
                    timer.elapsed().unwrap()
                );
            }
        }

        for (record, embedding) in records.iter_mut().zip(&embeddings) {
            writer.write_vector(record, embedding)?;
        }

        writer.flush()?;
    }

    writer.finish()?;

    Ok(())
}
