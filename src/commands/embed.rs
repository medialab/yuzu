use clap::Args;
use hf_hub::api::sync::Api;
use ndarray::{Array2, ArrayView1, Axis};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use simd_csv::{ByteRecord, Reader};
use std::fs::File;
use std::path::PathBuf;
use tokenizers::{PaddingDirection, Tokenizer};

use crate::utils::pooling;
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
    path: PathBuf,

    #[command(flatten)]
    common: CommonArgs,
}

pub fn action(args: EmbedArgs) -> CLIResult<()> {
    let mut reader = Reader::from_reader(File::open(args.path)?);
    let mut record = ByteRecord::new();

    let mut input: Vec<String> = Vec::new();

    while reader.read_byte_record(&mut record)? {
        let string = String::from_utf8(record[0].to_vec()).unwrap();
        input.push(string);
    }

    let api = Api::new().unwrap();
    let repo = api.model("medialab-sciencespo/Qwen3-Embedding-0.6B-ONNX".to_string());
    let onnx_file = repo.get("onnx/model.onnx").unwrap();
    let config_file = repo.get("config.json").unwrap();
    let vectorizer_file = repo.get("tokenizer.json").unwrap();
    let _data_file = repo.get("onnx/model.onnx_data");

    let config = File::open(config_file).expect("file should open read only");
    let json: serde_json::Value =
        serde_json::from_reader(config).expect("file should be proper JSON");
    let model_type = json
        .get("model_type")
        .expect("file should have model_type key")
        .as_str();

    let padding = tokenizers::PaddingParams {
        direction: match model_type {
            Some("qwen3") => PaddingDirection::Left,
            _ => PaddingDirection::Right,
        },
        ..Default::default()
    };
    let mut tokenizer = Tokenizer::from_file(vectorizer_file).unwrap();
    tokenizer.with_padding(Some(padding));

    let mut session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_file(onnx_file)
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

    let attention_mask = a_mask.try_extract_array::<i64>().unwrap();
    let pooled_embeddings: Array2<f32> = match model_type {
        Some("qwen3") => pooling::last_token(&last_hidden_state),
        _ => pooling::mean_pooling(&last_hidden_state, &attention_mask),
    };

    let normalized: Vec<Vec<f32>> = pooled_embeddings
        .axis_iter(Axis(0))
        .map(l2_normalize)
        .collect();

    dbg!(normalized);

    Ok(())
}
