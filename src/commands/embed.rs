use clap::Args;
use ndarray::{ArrayView1, Axis};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use simd_csv::ByteRecord;
use std::fs::File;
use tokenizers::Tokenizer;

use crate::utils::hf::EmbeddingModel;
use crate::utils::hf::get_model_files;
use crate::utils::io;
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

#[derive(Args, Debug)]
pub struct EmbedArgs {
    column: SelectedColumns,

    /// Path to CSV file containing text to classify (will use stdin if not given or if path is "-").
    input: Option<String>,

    #[arg(short, long)]
    model: Option<EmbeddingModel>,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: EmbedArgs) -> CLIResult<()> {
    let mut reader = io::Input::new(&args.input)
        .delimiter(args.common.delimiter)
        .csv_reader()?;
    let headers = reader.byte_headers()?;
    let column_index = args.column.single_selection(headers, true)?;

    let mut record = ByteRecord::new();

    let mut input: Vec<String> = Vec::new();

    while reader.read_byte_record(&mut record)? {
        let string = String::from_utf8(record[column_index].to_vec()).unwrap();
        input.push(string);
    }

    let model = args.model.unwrap_or_default();

    let padding = tokenizers::PaddingParams {
        direction: model.padding_direction,
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

    let encodings = tokenizer.encode_batch(input.clone(), true).unwrap();
    let padded_token_length = encodings[0].len();

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

    let a_ids = TensorRef::from_array_view(([input.len(), padded_token_length], &*ids)).unwrap();
    let a_mask = TensorRef::from_array_view(([input.len(), padded_token_length], &*mask)).unwrap();
    let a_position_ids =
        TensorRef::from_array_view(([input.len(), padded_token_length], &*position_ids)).unwrap();

    let session_input = match model_type {
        Some("qwen3") => Vec::from(ort::inputs![a_ids, a_mask.clone(), a_position_ids]),
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

    let mut writer = io::Output::new(&None).csv_writer()?;
    for i in &normalized {
        let mut record = ByteRecord::new();
        for f in i {
            record.push_field(f.to_string().as_bytes());
        }
        writer.write_byte_record(&record)?;
    }
    Ok(())
}
