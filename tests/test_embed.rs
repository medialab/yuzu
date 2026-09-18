use crate::utils::cmd;

#[test]
fn embed() {
    cmd()
        .arg("embed")
        .arg("text")
        .args(["--model", "test-model"])
        .write_csv_stdin(&[
            &["text"],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Yuzu is a citrus fruit"],
        ])
        .approx_assert_csv_matrix(
            vec![
                vec![-0.09802, -0.01892, 0.02692, 0.0107, 0.04899],
                vec![-0.03927, 0.00132, 0.00888, 0.02248, 0.02282],
            ],
            1,
        );
}

#[test]
fn embed_with_chunks() {
    cmd()
        .arg("embed")
        .arg("text")
        .args([
            "--model",
            "test-model",
            "--batch-size",
            "2",
            "--chunk-size",
            "1",
        ])
        .write_csv_stdin(&[
            &["text"],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Yuzu is a citrus fruit"],
        ])
        .approx_assert_csv_matrix(
            vec![
                vec![-0.09802, -0.01892, 0.02692, 0.0107, 0.04899],
                vec![-0.03927, 0.00132, 0.00888, 0.02248, 0.02282],
            ],
            1,
        );
}

#[test]
fn embed_total() {
    cmd()
        .arg("embed")
        .arg("text")
        .args(["--model", "test-model", "--batch-size", "-1"])
        .write_csv_stdin(&[
            &["text"],
            &["Yuzu koshō (柚子胡椒; also yuzu goshō) is a type of Japanese seasoning."],
            &["Yuzu is a citrus fruit"],
        ])
        .approx_assert_csv_matrix(
            vec![
                vec![-0.09802, -0.01892, 0.02692, 0.0107, 0.04899],
                vec![-0.03927, 0.00132, 0.00888, 0.02248, 0.02282],
            ],
            1,
        );
}

#[test]
fn embed_argsort_consistency() {
    let expected = vec![
        vec![
            0.03272301,
            0.05405777,
            0.039685573,
            -0.020051328,
            -0.14868496,
        ],
        vec![
            -0.049536016,
            0.0014692778,
            0.0014806606,
            -0.047504283,
            -0.050163805,
        ],
        vec![
            -0.044822857,
            0.008428363,
            0.0014387168,
            -0.06648014,
            -0.0467383,
        ],
        vec![
            -0.06250424,
            -0.0065791286,
            0.0613406,
            -0.03893918,
            0.031070044,
        ],
    ];

    cmd()
        .arg("embed")
        .arg("text")
        .args([
            "--model",
            "test-model",
            "--batch-size",
            "4",
            "--chunk-size",
            "4",
        ])
        .write_csv_stdin(&[
            &["text"],
            &["cacatoes ok?"],
            &["le chat mange la souris"],
            &["le chat mange"],
            &["le chat"],
        ])
        .approx_assert_csv_matrix(expected.clone(), 1);

    cmd()
        .arg("embed")
        .arg("text")
        .args([
            "--model",
            "test-model",
            "--batch-size",
            "1",
            "--chunk-size",
            "1",
        ])
        .write_csv_stdin(&[
            &["text"],
            &["cacatoes ok?"],
            &["le chat mange la souris"],
            &["le chat mange"],
            &["le chat"],
        ])
        .approx_assert_csv_matrix(expected.clone(), 1);
}
