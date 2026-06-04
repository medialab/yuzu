use hf_hub::api::sync::{Api, ApiError, ApiRepo};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokenizers::{
    Error as TokenizerError, PaddingDirection, PaddingParams, Tokenizer, TruncationParams,
};

use crate::utils::pooling;

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    model_id: String,
    pub dim: u64,
    pub padding_direction: PaddingDirection,
    pub pooling: pooling::Pooling,
    pub max_length: usize,
    pub disk_size: String,
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
            disk_size: String::from("417M"),
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
                disk_size: String::from("1.2G"),
                onnx_file: String::from("onnx/model.onnx"),
                onnx_data_file: Some(String::from("onnx/model.onnx_data")),
                ..Default::default()
            }),
            "sentence-transformers/all-MiniLM-L6-v2" => Ok(EmbeddingModel {
                model_id: String::from("sentence-transformers/all-MiniLM-L6-v2"),
                dim: 384,
                pooling: pooling::Pooling::Mean,
                max_length: 256,
                disk_size: String::from("174M"),
                onnx_file: String::from("onnx/model.onnx"),
                ..Default::default()
            }),
            "Lajavaness/sentence-camembert-large" => Ok(EmbeddingModel {
                model_id: String::from("Lajavaness/sentence-camembert-large"),
                dim: 1024,
                pooling: pooling::Pooling::Mean,
                max_length: 256,
                disk_size: String::from("1.3G"),
                onnx_file: String::from("onnx/model_O2.onnx"),
                ..Default::default()
            }),
            // #[cfg(test)]
            "test-model" => Ok(EmbeddingModel {
                model_id: String::from("local"),
                pooling: pooling::Pooling::Mean,
                onnx_file: String::from("onnx/model.onnx"),
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

pub static SUPPORTED_MODELS: [&str; 4] = [
    "ibm-granite/granite-embedding-107m-multilingual",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
    "Lajavaness/sentence-camembert-large",
];

pub fn print_models_list() {
    use colored::Colorize;

    for model_name in SUPPORTED_MODELS {
        let model: EmbeddingModel = model_name.parse().unwrap();

        println!("{}", model.model_id.cyan());
        println!("url: {}", model.url().blue());
        println!("dimensions: {}", model.dim.to_string().red());
        println!("size on disk: {}", model.disk_size.purple());
        println!("context window: {}", model.max_length.to_string().red());
        println!("pooling: {}", model.pooling.as_str().green());
        println!();
    }
}

pub struct ModelPaths {
    pub onnx: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl EmbeddingModel {
    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}", self.model_id)
    }

    fn repo(&self) -> Result<ApiRepo, ApiError> {
        let api = Api::new()?;
        Ok(api.model(self.model_id.clone()))
    }

    pub fn paths(&self) -> Result<ModelPaths, ApiError> {
        let (onnx_file, config_file, tokenizer_file) = if self.local {
            (
                Path::new(&self.model_id).join(&self.onnx_file),
                Path::new(&self.model_id).join(&self.config_file),
                Path::new(&self.model_id).join(&self.tokenizer_file),
            )
        } else {
            let repo = self.repo()?;

            if let Some(data_file) = &self.onnx_data_file {
                repo.get(data_file)?;
            }

            (
                repo.get(&self.onnx_file)?,
                repo.get(&self.config_file)?,
                repo.get(&self.tokenizer_file)?,
            )
        };

        Ok(ModelPaths {
            onnx: onnx_file,
            config: config_file,
            tokenizer: tokenizer_file,
        })
    }

    pub fn tokenizer_path(&self) -> Result<PathBuf, ApiError> {
        if self.local {
            Ok(Path::new(&self.model_id).join(&self.tokenizer_file))
        } else {
            let repo = self.repo()?;
            repo.get(&self.tokenizer_file)
        }
    }

    pub fn tokenizer(&self, path: impl AsRef<Path>) -> Result<Tokenizer, TokenizerError> {
        let padding = PaddingParams {
            direction: self.padding_direction,
            ..Default::default()
        };

        let truncation = TruncationParams {
            max_length: self.max_length,
            ..Default::default()
        };

        let mut tokenizer = Tokenizer::from_file(path)?;
        tokenizer.with_padding(Some(padding));
        tokenizer.with_truncation(Some(truncation))?;

        Ok(tokenizer)
    }
}
