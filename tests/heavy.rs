mod utils;
use utils::cmd;
#[test]
fn embed() {
    let sentence_transformers_matrix: Vec<Vec<f32>> = vec![
        vec![-0.037464, 0.070467, -0.008327, -0.003479, 0.029578],
        vec![-0.009637, 0.050861, -0.002397, -0.000858, 0.008585],
    ];

    cmd()
        .arg("embed")
        .arg("sentence")
        .args(["--model", "medialab-sciencespo/Qwen3-Embedding-0.6B-ONNX"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin."],
            &["Le yuzu est un agrume acide originaire de l'est de l'Asie."],
        ])
        .assert_csv_matrix(sentence_transformers_matrix);
}
