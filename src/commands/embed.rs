use std::num::NonZeroUsize;
use std::path::Path;
use std::time::SystemTime;

use clap::Args;
use ndarray::{ArrayView1, Axis};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use rayon::prelude::*;
use simd_csv::{ByteRecord, Selector};
use std::fs::File;
use std::iter::zip;
use tokenizers::Tokenizer;

use crate::utils::hf::{EmbeddingModel, print_models_list};
use crate::utils::io;
use crate::utils::io::DynamicUsize;
use crate::utils::iter::IteratorExt;
use crate::utils::readers::ReaderExt;
use crate::{CLIResult, CommonArgs, ParallelizationArgs};

fn l2_normalize(vec: ArrayView1<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

fn encode(
    input: Vec<&str>,
    session: &mut Session,
    tokenizer: &Tokenizer,
    model: &EmbeddingModel,
    model_type: Option<&str>,
) -> Vec<Vec<f32>> {
    let encodings = tokenizer.encode_batch(input.clone(), true).unwrap();
    let padded_token_length = encodings
        .iter()
        .map(|encoding| encoding.len())
        .max()
        .unwrap();

    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
        .collect();

    let mask: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
        .collect();

    let position_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|_| (0..padded_token_length as i64))
        .collect();

    let type_ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
        .collect();

    let a_ids = TensorRef::from_array_view(([input.len(), padded_token_length], &*ids)).unwrap();
    let a_mask = TensorRef::from_array_view(([input.len(), padded_token_length], &*mask)).unwrap();
    let a_position_ids =
        TensorRef::from_array_view(([input.len(), padded_token_length], &*position_ids)).unwrap();
    let a_type_ids =
        TensorRef::from_array_view(([input.len(), padded_token_length], &*type_ids)).unwrap();

    let session_input = match model_type {
        Some("qwen3") => Vec::from(ort::inputs![a_ids, a_mask.clone(), a_position_ids]),
        Some("bert") => Vec::from(ort::inputs![a_ids, a_mask.clone(), a_type_ids]),
        _ => Vec::from(ort::inputs![a_ids, a_mask.clone()]),
    };

    let session_output: ort::session::SessionOutputs<'_> =
        session.run(session_input.as_slice()).unwrap();

    let last_hidden_state = session_output[0].try_extract_array::<f32>().unwrap();

    // TODO: What if attention_mask is not needed? in pooling.apply?
    let attention_mask = a_mask.try_extract_array::<i64>().unwrap();
    let pooled_embeddings = model
        .pooling
        .apply(&last_hidden_state, Some(&attention_mask));

    let normalized: Vec<Vec<f32>> = pooled_embeddings
        .axis_iter(Axis(0))
        .map(l2_normalize)
        .collect();
    normalized
}

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
                io::Input::new(&Some(output_path))
                    .csv_splitter()?
                    .count_records()?,
            )
        } else {
            None
        }
    } else {
        None
    };

    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;

    if let Some(skip) = rows_to_skip_when_resuming {
        reader.skip(skip)?;
    }

    // TODO: --resume must open output with append, and write headers only if file already exists
    // TODO: this require Output to have a method to set append mode & to return whether it already exist
    // TODO: this means it should be opened before to avoid repeating the condition in line 147

    let text_column_index = reader.select_one(args.text_column.as_ref().unwrap())?;
    let output = io::Output::new(&args.output);
    let model = args.model.unwrap_or_default();
    let mut writer = output.vector_writer(model.dim)?;

    let model_files = model.paths()?;

    let config = File::open(model_files.config).expect("file should open read only");
    let json: serde_json::Value =
        serde_json::from_reader(config).expect("file should be proper JSON");
    let model_type = json
        .get("model_type")
        .expect("file should have model_type key")
        .as_str();

    let tokenizer = model.tokenizer(&model_files.tokenizer)?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .with_intra_threads(threads)?
        .commit_from_file(model_files.onnx)?;

    if reader.has_headers() {
        writer.write_headers(reader.byte_headers()?, model.dim, "dim_")?;
    }

    for batch in reader.into_byte_records().chunks_or_total(args.batch_size) {
        let mut input_batch: Vec<String> = Vec::with_capacity(batch.len());
        let mut records: Vec<ByteRecord> = Vec::with_capacity(batch.len());
        for row in batch.into_iter() {
            let record = row?;
            let string = String::from_utf8(record[text_column_index].to_vec()).unwrap();
            input_batch.push(string);
            records.push(record);
        }

        let mut sort_indices = (0..input_batch.len()).collect::<Vec<_>>();

        if threads > 1 {
            sort_indices.par_sort_unstable_by_key(|&i| input_batch[i].len());
        } else {
            sort_indices.sort_unstable_by_key(|&i| input_batch[i].len());
        }

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(input_batch.len());

        for idx_chunk in sort_indices.chunks(args.chunk_size.get()) {
            let timer_opt = args.verbose.then(SystemTime::now);

            let input: Vec<&str> = idx_chunk.iter().map(|&i| input_batch[i].as_str()).collect();
            let embedding = encode(input, &mut session, &tokenizer, &model, model_type);

            embeddings.extend(embedding);

            if let Some(timer) = timer_opt {
                eprintln!(
                    "Batch ({}) took {:?}",
                    args.chunk_size,
                    timer.elapsed().unwrap()
                );
            }
        }

        for (i, mut record) in zip(&sort_indices, records) {
            writer.write_vector(&mut record, &embeddings[*i])?;
        }
    }
    writer.finish()?;

    Ok(())
}
