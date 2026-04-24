use clap::Args;
use ndarray::{ArrayView1, Axis};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use simd_csv::ByteRecord;
use std::fs::File;
use std::iter::zip;
use tokenizers::Tokenizer;

use crate::utils::hf::EmbeddingModel;
use crate::utils::hf::get_model_files;
use crate::utils::io;
use crate::utils::iter::IteratorExt;
use crate::utils::select::SelectedColumns;
use crate::{CLIResult, CommonArgs};

fn l2_normalize(vec: ArrayView1<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

fn encode(
    input: Vec<String>,
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
    column: SelectedColumns,

    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,

    #[arg(short, long)]
    model: Option<EmbeddingModel>,

    #[command(flatten)]
    common: CommonArgs,

    /// Path to output file. Will infer the format (CSV or numpy) depending on the extension (.csv or .npy)
    /// Will write in CSV to stdout if not given or if path is "-".
    #[arg(short, long)]
    output: Option<String>,
}

pub fn action(args: EmbedArgs) -> CLIResult<()> {
    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .no_headers(args.common.no_headers)
        .csv_reader()?;
    let column_index = args.column.single_selection(reader.byte_headers()?, true)?;
    let output = io::Output::new(&args.output);
    let model = args.model.unwrap_or_default();
    let mut writer = output.vector_writer(model.dim)?;

    let padding = tokenizers::PaddingParams {
        direction: model.padding_direction,
        ..Default::default()
    };

    let truncation = tokenizers::TruncationParams {
        max_length: model.max_length,
        ..Default::default()
    };

    let model_files = get_model_files(&model);

    let config = File::open(model_files.config).expect("file should open read only");
    let json: serde_json::Value =
        serde_json::from_reader(config).expect("file should be proper JSON");
    let model_type = json
        .get("model_type")
        .expect("file should have model_type key")
        .as_str();

    let mut tokenizer = Tokenizer::from_file(model_files.tokenizer).unwrap();
    tokenizer.with_padding(Some(padding));
    tokenizer.with_truncation(Some(truncation)).unwrap();

    let mut session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_file(model_files.onnx)
        .unwrap();

    if reader.has_headers() {
        writer.write_headers(reader.byte_headers()?, model.dim, "dim_")?;
    }

    for chunk in reader.into_byte_records().chunks(32) {
        let mut input: Vec<String> = Vec::new();
        let mut records: Vec<ByteRecord> = Vec::new();
        for result in chunk.into_iter() {
            let record = result?;
            let string = String::from_utf8(record[column_index].to_vec()).unwrap();
            input.push(string);
            records.push(record);
        }
        let embedding = encode(input, &mut session, &tokenizer, &model, model_type);
        for (i, mut record) in zip(&embedding, records) {
            writer.write_vector(&mut record, i)?;
        }
    }
    writer.finish()?;

    Ok(())
}
