use std::convert::Infallible;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use hf_hub::api::sync::{Api, ApiError, ApiRepo};
use ndarray::{ArrayView1, Axis};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokenizers::{
    EncodeInput, Error as TokenizerError, PaddingDirection, PaddingParams, Tokenizer,
    TruncationParams,
};

use crate::CLIResult;
use crate::utils::pooling::Pooling;

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Qwen3,
    Bert,
    Other,
}

impl FromStr for ModelType {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "qwen3" => Self::Qwen3,
            "bert" => Self::Bert,
            _ => Self::Other,
        })
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    model_id: &'static str,
    alias: Option<&'static str>,
    pub dim: u64,
    pub padding_direction: PaddingDirection,
    pub pooling: Pooling,
    pub max_length: usize,
    pub disk_size: &'static str,
    pub preferred_language: &'static str,
    onnx_file: &'static str,
    config_file: &'static str,
    tokenizer_file: &'static str,
    onnx_data_file: Option<&'static str>,
    local: bool,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        "granite".parse().unwrap()
    }
}

impl FromStr for EmbeddingModel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "granite" | "ibm-granite/granite-embedding-107m-multilingual" => Self {
                model_id: "ibm-granite/granite-embedding-107m-multilingual",
                alias: Some("granite"),
                dim: 384,
                padding_direction: PaddingDirection::Right,
                pooling: Pooling::Cls,
                max_length: 512,
                disk_size: "417M",
                preferred_language: "multilingual",
                onnx_file: "model.onnx",
                config_file: "config.json",
                tokenizer_file: "tokenizer.json",
                onnx_data_file: None,
                local: false,
            },
            "qwen" | "Qwen/Qwen3-Embedding-0.6B" => Self {
                model_id: "medialab-sciencespo/Qwen3-Embedding-0.6B-ONNX",
                alias: Some("qwen"),
                dim: 1024,
                padding_direction: PaddingDirection::Left,
                pooling: Pooling::LastToken,
                max_length: 8192,
                disk_size: "1.2G",
                preferred_language: "multilingual",
                onnx_file: "onnx/model.onnx",
                onnx_data_file: Some("onnx/model.onnx_data"),
                ..Default::default()
            },
            "mini" | "sentence-transformers/all-MiniLM-L6-v2" => Self {
                model_id: "sentence-transformers/all-MiniLM-L6-v2",
                alias: Some("mini"),
                dim: 384,
                pooling: Pooling::Mean,
                max_length: 256,
                disk_size: "174M",
                preferred_language: "english",
                onnx_file: "onnx/model.onnx",
                ..Default::default()
            },
            "camembert" | "Lajavaness/sentence-camembert-large" => Self {
                model_id: "Lajavaness/sentence-camembert-large",
                alias: Some("camembert"),
                dim: 1024,
                pooling: Pooling::Mean,
                max_length: 256,
                disk_size: "1.3G",
                preferred_language: "french",
                onnx_file: "onnx/model_O2.onnx",
                ..Default::default()
            },
            // #[cfg(test)]
            "test-model" => Self {
                model_id: "local",
                pooling: Pooling::Mean,
                onnx_file: "onnx/model.onnx",
                max_length: 256,
                local: true,
                ..Default::default()
            },
            _ => {
                let msg = format!("Model {value} not supported");
                return Err(msg);
            }
        })
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

        println!(
            "{}{}",
            model.model_id.cyan(),
            if let Some(name) = model.alias {
                format!(" ({name})")
            } else {
                "".to_string()
            }
        );
        println!("url: {}", model.url().blue());
        println!("dimensions: {}", model.dim.to_string().red());
        println!("size on disk: {}", model.disk_size.purple());
        println!("context window: {}", model.max_length.to_string().red());
        println!("pooling: {}", model.pooling.as_str().green());
        println!("preferred language: {}", model.preferred_language.green());
        println!();
    }
}

pub struct ModelPaths {
    pub onnx: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    pub fn model_type(&self) -> Result<ModelType, &'static str> {
        let config = File::open(&self.config).map_err(|_| "could not open model config file")?;
        let json: serde_json::Value =
            serde_json::from_reader(config).map_err(|_| "model config is not valid JSON")?;
        let model_type_str = json
            .get("model_type")
            .ok_or("config file should have a model_type")?
            .as_str()
            .unwrap();

        Ok(model_type_str.parse().unwrap())
    }
}

impl EmbeddingModel {
    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}", self.model_id)
    }

    fn repo(&self) -> Result<ApiRepo, ApiError> {
        let api = Api::new()?;
        Ok(api.model(self.model_id.to_string()))
    }

    pub fn paths(&self) -> Result<ModelPaths, ApiError> {
        let (onnx_file, config_file, tokenizer_file) = if self.local {
            (
                Path::new(&self.model_id).join(self.onnx_file),
                Path::new(&self.model_id).join(self.config_file),
                Path::new(&self.model_id).join(self.tokenizer_file),
            )
        } else {
            let repo = self.repo()?;

            if let Some(data_file) = &self.onnx_data_file {
                repo.get(data_file)?;
            }

            (
                repo.get(self.onnx_file)?,
                repo.get(self.config_file)?,
                repo.get(self.tokenizer_file)?,
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
            Ok(Path::new(&self.model_id).join(self.tokenizer_file))
        } else {
            let repo = self.repo()?;
            repo.get(self.tokenizer_file)
        }
    }

    pub fn tokenizer_from_path(&self, path: impl AsRef<Path>) -> Result<Tokenizer, TokenizerError> {
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

    #[inline]
    pub fn tokenizer(&self) -> Result<Tokenizer, TokenizerError> {
        self.tokenizer_from_path(self.tokenizer_path()?)
    }

    pub fn embedder(&self, threads: usize) -> CLIResult<Embedder> {
        let paths = self.paths()?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([ep::CPU::default().build()])?
            .with_intra_threads(threads)?
            .commit_from_file(&paths.onnx)?;

        let tokenizer = self.tokenizer_from_path(&paths.tokenizer)?;

        let model_type = paths.model_type()?;

        Ok(Embedder {
            tokenizer,
            session,
            model_type,
            pooling: self.pooling,
        })
    }
}

fn l2_normalize(vec: ArrayView1<f32>) -> Vec<f32> {
    let norm = vec.dot(&vec).sqrt();

    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

pub struct Embedder {
    tokenizer: Tokenizer,
    session: Session,
    model_type: ModelType,
    pooling: Pooling,
}

impl Embedder {
    pub fn embed<'s, E>(&mut self, input: Vec<E>) -> CLIResult<Vec<Vec<f32>>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let input_len = input.len();

        debug_assert!(input_len > 0);

        let encodings = self.tokenizer.encode_batch(input, true)?;
        let padded_token_len = encodings
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
            .flat_map(|_| (0..padded_token_len as i64))
            .collect();

        let type_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_type_ids().iter().map(|i| *i as i64))
            .collect();

        let a_ids = TensorRef::from_array_view(([input_len, padded_token_len], &*ids))?;
        let a_mask = TensorRef::from_array_view(([input_len, padded_token_len], &*mask))?;
        let a_position_ids =
            TensorRef::from_array_view(([input_len, padded_token_len], &*position_ids))?;
        let a_type_ids = TensorRef::from_array_view(([input_len, padded_token_len], &*type_ids))?;

        let session_input = match self.model_type {
            ModelType::Qwen3 => Vec::from(ort::inputs![a_ids, a_mask.clone(), a_position_ids]),
            ModelType::Bert => Vec::from(ort::inputs![a_ids, a_mask.clone(), a_type_ids]),
            ModelType::Other => Vec::from(ort::inputs![a_ids, a_mask.clone()]),
        };

        let session_output: ort::session::SessionOutputs<'_> =
            self.session.run(session_input.as_slice())?;

        let last_hidden_state = session_output[0].try_extract_array::<f32>()?;

        // TODO: What if attention_mask is not needed? in pooling.apply?
        let attention_mask = a_mask.try_extract_array::<i64>()?;
        let pooled_embeddings = self
            .pooling
            .apply(&last_hidden_state, Some(&attention_mask));

        Ok(pooled_embeddings
            .axis_iter(Axis(0))
            .map(l2_normalize)
            .collect())
    }
}
