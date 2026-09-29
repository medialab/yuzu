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
fn tokenize_fits() {
    cmd()
        .arg("tokenize")
        .arg("sentence")
        .args(["--model", "test-model"])
        .arg("--fits")
        .write_csv_stdin(&[
            &["sentence"],
            &["Say hello to my little friend!"],
            &["Béatrice aime la babka."],
            &["--------------------------------------------------------------------------------------------------
            Par contre Benjamin n'aime pas le chocolat, ce qui implique qu'il ne mange pas de chocolat, \
            ni même de pains au chocolat. Vous préférez chocolatine ? Malheureusement ce n'est pas l'esprit \
            de Benjamin. Benjamin aime qu'il y ait le mot 'pain' dans 'pain au chocolat'. Déjà parce que ça \
            rime avec son prénom, mais surtout parce qu'il adore ses copains. Et vous savez ce qu'on dit : \
            'Pas de pain? Pas de copains'.
            --------------------------------------------------------------------------------------------------"],
        ])
        .assert_csv(&[&["fits"], &["1"], &["1"], &["0"]]);
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
