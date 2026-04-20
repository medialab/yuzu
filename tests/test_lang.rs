use crate::cmd;

#[test]
fn lang() {
    cmd()
        .arg("lang")
        .arg("sentence")
        .write_csv_stdin(&[
            &["sentence"],
            &["this is an English sentence"],
            &["cette phrase est en français"],
        ])
        .assert_csv(&[
            &["sentence", "lang"],
            &["this is an English sentence", "eng"],
            &["cette phrase est en français", "fra"],
        ]);
}

#[test]
fn lang_full_name() {
    cmd()
        .arg("lang")
        .arg("sentence")
        .arg("--full-name")
        .write_csv_stdin(&[
            &["sentence"],
            &["this is an English sentence"],
            &["cette phrase est en français"],
        ])
        .assert_csv(&[
            &["sentence", "lang"],
            &["this is an English sentence", "english"],
            &["cette phrase est en français", "french"],
        ]);
}

#[test]
fn lang_lang_column() {
    cmd()
        .arg("lang")
        .arg("sentence")
        .args(["--lang-column", "language"])
        .write_csv_stdin(&[
            &["sentence"],
            &["this is an English sentence"],
            &["cette phrase est en français"],
        ])
        .assert_csv(&[
            &["sentence", "language"],
            &["this is an English sentence", "eng"],
            &["cette phrase est en français", "fra"],
        ]);
}
