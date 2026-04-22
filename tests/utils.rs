use std::ffi;
use std::io::Cursor;

use assert_cmd::{Command, cargo_bin_cmd};
use simd_csv::{ByteRecord, Writer};

pub struct YuzuCommand(pub Command);

impl YuzuCommand {
    pub fn new() -> Self {
        Self(cargo_bin_cmd!())
    }

    #[inline]
    pub fn arg<S: AsRef<ffi::OsStr>>(&mut self, arg: S) -> &mut Self {
        self.0.arg(arg);
        self
    }

    #[inline]
    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<ffi::OsStr>,
    {
        self.0.args(args);
        self
    }

    pub fn write_csv_stdin(&mut self, data: &[&[&str]]) -> &mut Self {
        let mut buffer = Vec::<u8>::new();

        let mut writer = Writer::from_writer(&mut buffer);
        let mut record = ByteRecord::new();

        for row in data {
            record.clear();

            for cell in row.iter() {
                record.push_field(cell.as_bytes());
            }

            writer.write_byte_record(&record).unwrap();
        }

        std::mem::drop(writer);

        self.0.write_stdin(buffer);

        self
    }

    #[allow(dead_code)]
    pub fn assert_csv_matrix(&mut self, data: Vec<Vec<f32>>) {
        let assert = self.0.assert().success();

        let reader = simd_csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(Cursor::new(&assert.get_output().stdout));

        let records: Vec<Vec<f32>> = reader
            .into_byte_records()
            .map(|record| {
                record
                    .unwrap()
                    .into_iter()
                    .take(5)
                    .map(|cell| {
                        let x = std::str::from_utf8(cell).unwrap().parse::<f32>().unwrap();
                        (x * 1_000_000.0).round() / 1_000_000.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(records, data);
    }

    #[allow(dead_code)]
    pub fn assert_csv(&mut self, data: &[&[&str]]) {
        let assert: assert_cmd::assert::Assert = self.0.assert().success();

        let reader = simd_csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(Cursor::new(&assert.get_output().stdout));

        let records = reader
            .into_byte_records()
            .map(|record| {
                record
                    .unwrap()
                    .into_iter()
                    .map(|cell| std::str::from_utf8(cell).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(records, data);
    }
}

pub fn cmd() -> YuzuCommand {
    YuzuCommand::new()
}
