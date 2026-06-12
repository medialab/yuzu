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
