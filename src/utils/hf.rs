use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokenizers::PaddingDirection;

use crate::utils::pooling;

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    model_id: String,
    pub dim: u64,
    pub padding_direction: PaddingDirection,
    pub pooling: pooling::Pooling,
    pub max_length: usize,
    onnx_file: String,
    config_file: String,
    tokenizer_file: String,
    onnx_data_file: Option<String>,
    local: bool,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self {
            model_id: String::from("ibm-granite/granite-embedding-107m-multilingual"),
            dim: 384,
            padding_direction: PaddingDirection::Right,
            pooling: pooling::Pooling::Cls,
            max_length: 512,
            onnx_file: String::from("model.onnx"),
            config_file: String::from("config.json"),
            tokenizer_file: String::from("tokenizer.json"),
            onnx_data_file: None,
            local: false,
        }
    }
}

impl FromStr for EmbeddingModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ibm-granite/granite-embedding-107m-multilingual" => Ok(Default::default()),
            "Qwen/Qwen3-Embedding-0.6B" => Ok(EmbeddingModel {
                model_id: String::from("medialab-sciencespo/Qwen3-Embedding-0.6B-ONNX"),
                dim: 1024,
                padding_direction: PaddingDirection::Left,
                pooling: pooling::Pooling::LastToken,
                max_length: 8192,
                onnx_file: String::from("onnx/model.onnx"),
                onnx_data_file: Some(String::from("onnx/model.onnx_data")),
                ..Default::default()
            }),
            "sentence-transformers/all-MiniLM-L6-v2" => Ok(EmbeddingModel {
                model_id: String::from("sentence-transformers/all-MiniLM-L6-v2"),
                dim: 384,
                pooling: pooling::Pooling::Mean,
                max_length: 256,
                onnx_file: String::from("onnx/model.onnx"),
                ..Default::default()
            }),
            // #[cfg(test)]
            "test-model" => Ok(EmbeddingModel {
                model_id: String::from("local"),
                dim: 384,
                pooling: pooling::Pooling::Cls,
                onnx_file: String::from("onnx/model_qint8_arm64.onnx"),
                max_length: 256,
                local: true,
                ..Default::default()
            }),
            _ => {
                let msg = format!("Model {} not supported", value);
                Err(msg)
            }
        }
    }
}

pub struct ModelPaths {
    pub onnx: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

pub fn get_model_files(model: &EmbeddingModel) -> ModelPaths {
    let (onnx_file, config_file, tokenizer_file) = match model.local {
        true => (
            Path::new(&model.model_id).join(&model.onnx_file),
            Path::new(&model.model_id).join(&model.config_file),
            Path::new(&model.model_id).join(&model.tokenizer_file),
        ),
        false => {
            let api = Api::new().unwrap();
            let repo = api.model(model.model_id.clone());
            if let Some(data_file) = model.onnx_data_file.clone() {
                let _data_file = repo.get(&data_file);
            }
            (
                repo.get(&model.onnx_file).unwrap(),
                repo.get(&model.config_file).unwrap(),
                repo.get(&model.tokenizer_file).unwrap(),
            )
        }
    };
    ModelPaths {
        onnx: onnx_file,
        config: config_file,
        tokenizer: tokenizer_file,
    }
}
