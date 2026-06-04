use crate::utils::cmd;

#[test]
fn tokenize() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .args(["--model", "test-model"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
        ])
        .assert_csv(&[
            &["tokens"],
            &["[CLS] say hello to my little friend ! [SEP]"],
            &["[CLS] beatrice aim ##e la ba ##b ##ka . [SEP]"],
        ]);
}

#[test]
fn tokenize_keep() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .arg("--keep")
        .args(["--model", "test-model"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
        ])
        .assert_csv(&[
            &["sentence", "tokens"],
            &[
                "Say hello to my little friend!",
                "[CLS] say hello to my little friend ! [SEP]",
            ],
            &[
                "Béatrice aime la babka.",
                "[CLS] beatrice aim ##e la ba ##b ##ka . [SEP]",
            ],
        ]);
}

#[test]
fn tokenize_count() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .args(["--model", "test-model"])
        .arg("--count")
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
        ])
        .assert_csv(&[&["token_count"], &["9"], &["10"]]);
}

#[test]
fn tokenize_column() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .args(["--model", "test-model"])
        .arg("--count")
        .args(["-c", "count"])
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
        ])
        .assert_csv(&[&["count"], &["9"], &["10"]]);
}

#[test]
fn tokenize_explode() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .args(["--model", "test-model"])
        .arg("--explode")
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
        ])
        .assert_csv(&[
            &["token"],
            &["[CLS]"],
            &["say"],
            &["hello"],
            &["to"],
            &["my"],
            &["little"],
            &["friend"],
            &["!"],
            &["[SEP]"],
            &["[CLS]"],
            &["beatrice"],
            &["aim"],
            &["##e"],
            &["la"],
            &["ba"],
            &["##b"],
            &["##ka"],
            &["."],
            &["[SEP]"],
        ]);
}
