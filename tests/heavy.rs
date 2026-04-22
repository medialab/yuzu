mod utils;
use utils::cmd;

#[test]
fn embed() {
    let sentence_transformers_matrix: Vec<Vec<f32>> = vec![
        vec![-0.03746, 0.07047, -0.00833, -0.00348, 0.02958],
        vec![-0.00964, 0.05086, -0.0024, -0.00086, 0.00859],
        vec![0.06019, -0.03181, -0.00608, 0.02732, 0.0169],
        vec![0.02755, 0.01381, -0.00894, -0.00323, -0.00154],
        vec![0.05729, -0.0534, -0.00716, 0.05947, 0.04664],
        vec![0.05684, -0.04139, -0.00813, 0.03242, 0.06028],
    ];

    cmd()
        .arg("embed")
        .arg("sentence")
        .args(["--model", "medialab-sciencespo/Qwen3-Embedding-0.6B-ONNX"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin."],
            &["Le yuzu est un agrume acide originaire de l'est de l'Asie."],
            &["Yuzukoshō (柚子胡椒 / ゆずこしょう?, aussi yuzugoshō) est un type d'assaisonnement nippon."],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Le Cochon est un documentaire français de moyen métrage coréalisé par Jean-Michel Barjol et Jean Eustache, sorti en 1970."],
            &["Le Cochon ('The Pig') is a fifty-minute featurette co-directed by Jean Eustache and Jean-Michel Barjol in 1970."],
        ])
        .assert_csv_matrix(sentence_transformers_matrix);
}
