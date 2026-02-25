use crate::cmd;

#[test]
fn lang() {
    cmd()
        .arg("lang")
        .arg("sentence")
        .write_csv_stdin(&[&["sentence"], &["this is an English sentence"]])
        .assert_csv(&[
            &["sentence", "lang"],
            &["this is an English sentence", "eng"],
        ]);
}
