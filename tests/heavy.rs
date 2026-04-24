mod utils;
use utils::cmd;

#[test]
fn embed_bert() {
    let sentence_transformers_matrix: Vec<Vec<f32>> = vec![
        vec![-0.06034, 0.00864, -0.00112, 0.02872, 0.03097],
        vec![-0.08879, -0.05624, -0.0177, -0.01385, 0.0494],
        vec![-0.09631, 0.04346, 0.02765, -0.01094, -0.03659],
        vec![-0.09802, -0.01892, 0.02692, 0.0107, 0.04899],
        vec![-0.01697, 0.07074, 0.00851, -0.06991, -0.05831],
        vec![0.0174, 0.05749, 0.02785, -0.09676, -0.05744],
    ];

    cmd()
        .arg("embed")
        .arg("sentence")
        .args(["--model", "sentence-transformers/all-MiniLM-L6-v2"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin."],
            &["Le yuzu est un agrume acide originaire de l'est de l'Asie."],
            &["Yuzukoshō (柚子胡椒 / ゆずこしょう?, aussi yuzugoshō) est un type d'assaisonnement nippon."],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Le Cochon est un documentaire français de moyen métrage coréalisé par Jean-Michel Barjol et Jean Eustache, sorti en 1970."],
            &["Le Cochon ('The Pig') is a fifty-minute featurette co-directed by Jean Eustache and Jean-Michel Barjol in 1970."],
        ])
        .assert_csv_matrix(sentence_transformers_matrix, 1);
}

#[test]
fn embed_granite() {
    let sentence_transformers_matrix: Vec<Vec<f32>> = vec![
        vec![0.01843, 0.03861, 0.06082, 0.00106, 0.0487],
        vec![-0.02896, 0.04126, 0.04314, 0.01148, 0.02963],
        vec![0.00232, -0.01301, 0.0637, 0.00497, 0.09661],
        vec![-0.01044, 0.0518, 0.02995, 0.01297, 0.06155],
        vec![0.01252, 0.10484, 0.04357, -0.00614, 0.10592],
        vec![0.03436, 0.07677, 0.04541, 0.01161, 0.09566],
    ];

    cmd()
        .arg("embed")
        .arg("sentence")
        .args(["--model", "ibm-granite/granite-embedding-107m-multilingual"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin."],
            &["Le yuzu est un agrume acide originaire de l'est de l'Asie."],
            &["Yuzukoshō (柚子胡椒 / ゆずこしょう?, aussi yuzugoshō) est un type d'assaisonnement nippon."],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Le Cochon est un documentaire français de moyen métrage coréalisé par Jean-Michel Barjol et Jean Eustache, sorti en 1970."],
            &["Le Cochon ('The Pig') is a fifty-minute featurette co-directed by Jean Eustache and Jean-Michel Barjol in 1970."],
        ])
        .assert_csv_matrix(sentence_transformers_matrix, 1);
}

#[test]
fn embed_qwen3() {
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
        .args(["--model", "Qwen/Qwen3-Embedding-0.6B"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Yuzu is a citrus fruit and plant in the family Rutaceae of Chinese origin."],
            &["Le yuzu est un agrume acide originaire de l'est de l'Asie."],
            &["Yuzukoshō (柚子胡椒 / ゆずこしょう?, aussi yuzugoshō) est un type d'assaisonnement nippon."],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Le Cochon est un documentaire français de moyen métrage coréalisé par Jean-Michel Barjol et Jean Eustache, sorti en 1970."],
            &["Le Cochon ('The Pig') is a fifty-minute featurette co-directed by Jean Eustache and Jean-Michel Barjol in 1970."],
        ])
        .assert_csv_matrix(sentence_transformers_matrix, 1);
}
