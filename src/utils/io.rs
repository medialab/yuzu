use std::fs::File;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::str::FromStr;

pub type BoxedReader = Box<dyn Read + Send + 'static>;
pub type BoxedWriter = Box<dyn Write + Send + 'static>;

#[derive(Clone, Copy, Debug)]
pub struct Delimiter(u8);

impl Delimiter {
    pub fn as_byte(self) -> u8 {
        self.0
    }
}

impl FromStr for Delimiter {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            r"\t" => Ok(Delimiter(b'\t')),
            s => {
                if s.len() != 1 {
                    let msg = format!(
                        "Could not convert '{}' to a single \
                                       ASCII character.",
                        s
                    );
                    return Err(msg);
                }
                let c = s.chars().next().unwrap();
                if c.is_ascii() {
                    Ok(Delimiter(c as u8))
                } else {
                    let msg = format!(
                        "Could not convert '{}' \
                                       to ASCII delimiter.",
                        c
                    );
                    Err(msg)
                }
            }
        }
    }
}

pub struct CSVInput {
    path: Option<PathBuf>,
    delimiter: u8,
}

impl Default for CSVInput {
    fn default() -> Self {
        Self {
            path: None,
            delimiter: b',',
        }
    }
}

impl CSVInput {
    pub fn new(path_opt: &Option<String>) -> Self {
        let mut input = Self::default();

        if let Some(path) = path_opt {
            if path != "-" {
                input.path = Some(PathBuf::from(path))
            }
        }

        input
    }

    pub fn delimiter(mut self, delimiter: Option<Delimiter>) -> Self {
        self.delimiter = match delimiter {
            None => b',',
            Some(d) => d.as_byte(),
        };
        self
    }

    pub fn reader(&self) -> io::Result<BoxedReader> {
        match &self.path {
            None => {
                if io::stdin().is_terminal() {
                    Err(io::Error::other(
                        "failed to read CSV data from stdin! Did you forget to give a path to your file?",
                    ))
                } else {
                    Ok(Box::new(io::stdin()))
                }
            }
            Some(path) => Ok(Box::new(File::open(path)?)),
        }
    }

    fn csv_reader_builder(&self) -> simd_csv::ReaderBuilder {
        let mut builder = simd_csv::ReaderBuilder::new();

        builder.delimiter(self.delimiter);

        builder
    }

    fn csv_reader_from_reader<R: Read>(&self, reader: R) -> simd_csv::Reader<R> {
        self.csv_reader_builder().from_reader(reader)
    }

    pub fn csv_reader(&self) -> io::Result<simd_csv::Reader<BoxedReader>> {
        Ok(self.csv_reader_from_reader(self.reader()?))
    }
}

pub struct CSVOutput {
    path: Option<PathBuf>,
    delimiter: u8,
}

impl Default for CSVOutput {
    fn default() -> Self {
        Self {
            path: None,
            delimiter: b',',
        }
    }
}

impl CSVOutput {
    pub fn new(path_opt: &Option<String>) -> Self {
        let mut input = Self::default();

        if let Some(path) = path_opt {
            if path != "-" {
                input.path = Some(PathBuf::from(path))
            }
        }

        input
    }

    pub fn writer(&self) -> io::Result<BoxedWriter> {
        match &self.path {
            None => {
                if io::stdout().is_terminal() {
                    Err(io::Error::other(
                        "failed to read CSV data from stdout! Did you forget to give a path to your file?",
                    ))
                } else {
                    Ok(Box::new(io::stdout()))
                }
            }
            Some(path) => Ok(Box::new(File::open(path)?)),
        }
    }

    fn csv_writer_builder(&self) -> simd_csv::WriterBuilder {
        let mut builder = simd_csv::WriterBuilder::with_capacity(32 * (1 << 10));

        builder.delimiter(self.delimiter);

        builder
    }

    fn csv_writer_from_writer<R: Write>(&self, writer: R) -> simd_csv::Writer<R> {
        self.csv_writer_builder().from_writer(writer)
    }

    pub fn csv_writer(&self) -> io::Result<simd_csv::Writer<BoxedWriter>> {
        Ok(self.csv_writer_from_writer(self.writer()?))
    }
}
