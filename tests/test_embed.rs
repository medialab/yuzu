use crate::utils::cmd;

#[test]
fn embed() {
    cmd()
        .arg("embed")
        .arg("text")
        .args(["--model", "test-model"])
        .write_csv_stdin(&[&["text"], &["Yuzu is a citrus fruit"]])
        .approx_assert_csv_matrix(vec![vec![-0.03927, 0.00132, 0.00888, 0.02248, 0.02282]], 1);
}
