use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, IsTerminal, Read, Seek, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use npyz::WriterBuilder;
extern crate npyz;

use crate::utils::writers;

const DEFAULT_BUFFERED_WRITER_CAPACITY: usize = 32 * (1 << 10);

pub trait SeekWrite: Seek + Write {}
impl<T: Seek + Write> SeekWrite for T {}

pub type BoxedReader = Box<dyn Read + Send + 'static>;
pub type BoxedWriter = Box<dyn Write + Send + 'static>;
pub type BoxedSeekableWriter = Box<dyn SeekWrite + Send + 'static>;

#[derive(Clone, Copy, Debug)]
pub struct Delimiter(u8);

#[derive(Debug, Clone, Copy)]
pub enum DynamicUsize {
    Unlimited,
    Limited(NonZeroUsize),
}

impl FromStr for DynamicUsize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            r"-1" => Ok(DynamicUsize::Unlimited),
            v => match NonZeroUsize::from_str(v) {
                Ok(i) => Ok(DynamicUsize::Limited(i)),
                Err(_) => {
                    let msg = "--batch-size should be strictly positive or equal to -1".to_string();
                    Err(msg)
                }
            },
        }
    }
}

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

#[derive(Debug, Clone, Copy)]
pub enum Compression {
    Gzip,
    Zstd,
}

impl Compression {
    fn infer_from_path(path: &str) -> Option<Self> {
        if path.ends_with(".gz") {
            Some(Self::Gzip)
        } else if path.ends_with(".zst") {
            Some(Self::Zstd)
        } else {
            None
        }
    }
}

pub struct Input {
    path: Option<PathBuf>,
    delimiter: u8,
    compression: Option<Compression>,
    no_headers: bool,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            path: None,
            delimiter: b',',
            compression: None,
            no_headers: false,
        }
    }
}

impl Input {
    pub fn new(path_opt: &Option<String>) -> Self {
        let mut input = Self::default();

        if let Some(path) = path_opt {
            if path != "-" {
                input.path = Some(PathBuf::from(path));
                input.compression = Compression::infer_from_path(path);
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

    pub fn no_headers(mut self, yes: bool) -> Self {
        self.no_headers = yes;
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
            Some(path) => {
                let file = File::open(path)?;

                match self.compression {
                    None => Ok(Box::new(file)),
                    Some(compression) => match compression {
                        Compression::Gzip => Ok(Box::new(flate2::read::MultiGzDecoder::new(file))),
                        Compression::Zstd => Ok(Box::new(zstd::Decoder::new(file)?)),
                    },
                }
            }
        }
    }

    fn csv_reader_builder(&self) -> simd_csv::ReaderBuilder {
        let mut builder = simd_csv::ReaderBuilder::new();

        builder
            .delimiter(self.delimiter)
            .has_headers(!self.no_headers);

        builder
    }

    fn csv_reader_from_reader<R: Read>(&self, reader: R) -> simd_csv::Reader<R> {
        self.csv_reader_builder().from_reader(reader)
    }

    pub fn csv_reader(&self) -> io::Result<simd_csv::Reader<BoxedReader>> {
        Ok(self.csv_reader_from_reader(self.reader()?))
    }

    fn csv_splitter_builder(&self) -> simd_csv::SplitterBuilder {
        let mut builder = simd_csv::SplitterBuilder::new();

        builder
            .delimiter(self.delimiter)
            .has_headers(!self.no_headers);

        builder
    }

    fn csv_splitter_from_reader<R: Read>(&self, reader: R) -> simd_csv::Splitter<R> {
        self.csv_splitter_builder().from_reader(reader)
    }

    pub fn csv_splitter(&self) -> io::Result<simd_csv::Splitter<BoxedReader>> {
        Ok(self.csv_splitter_from_reader(self.reader()?))
    }
}

pub enum FileFormat {
    Csv,
    Npy,
}

impl FileFormat {
    fn infer_from_path(path: &Option<String>) -> Self {
        match path {
            Some(filename) => {
                if filename.ends_with(".npy") {
                    Self::Npy
                } else {
                    Self::Csv
                }
            }
            None => Self::Csv,
        }
    }

    fn is_csv(&self) -> bool {
        matches!(self, Self::Csv)
    }
}

pub struct Output {
    path: Option<PathBuf>,
    delimiter: u8,
    pub format: FileFormat,
    pub can_resume: bool,
}

impl Default for Output {
    fn default() -> Self {
        Self {
            path: None,
            delimiter: b',',
            format: FileFormat::Csv,
            can_resume: false,
        }
    }
}

impl Output {
    pub fn new(path_opt: &Option<String>) -> Self {
        let mut input = Self {
            format: FileFormat::infer_from_path(path_opt),
            ..Default::default()
        };

        if let Some(path) = path_opt {
            if path != "-" {
                input.path = Some(PathBuf::from(path));
            }
        }

        input
    }

    pub fn maybe_resume(path_opt: &Option<String>, resume: bool) -> io::Result<Self> {
        let mut output = Self::new(path_opt);

        if resume {
            if !output.format.is_csv() {
                return Err(io::Error::other("can only resume CSV output!"));
            }

            output.can_resume = output.exists_and_nonempty()?;
        }

        Ok(output)
    }

    fn exists_and_nonempty(&self) -> io::Result<bool> {
        match &self.path {
            None => Ok(false),
            Some(path) => Ok(path.is_file() && path.metadata()?.len() > 0),
        }
    }

    fn open_file(&self, path: impl AsRef<Path>) -> io::Result<File> {
        if self.can_resume {
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .append(true)
                .open(path)
        } else {
            File::create(path)
        }
    }

    pub fn writer(&self) -> io::Result<BoxedWriter> {
        match &self.path {
            None => Ok(Box::new(io::stdout())),
            Some(path) => Ok(Box::new(self.open_file(path)?)),
        }
    }

    pub fn seekable_writer(&self) -> io::Result<BoxedSeekableWriter> {
        match &self.path {
            None => Err(io::Error::other("cannot seek output")),
            Some(path) => Ok(Box::new(self.open_file(path)?)),
        }
    }

    pub fn buf_writer(&self) -> io::Result<BufWriter<BoxedWriter>> {
        Ok(BufWriter::with_capacity(
            DEFAULT_BUFFERED_WRITER_CAPACITY,
            self.writer()?,
        ))
    }

    pub fn seekable_buf_writer(&self) -> io::Result<BufWriter<BoxedSeekableWriter>> {
        Ok(BufWriter::with_capacity(
            DEFAULT_BUFFERED_WRITER_CAPACITY,
            self.seekable_writer()?,
        ))
    }

    fn csv_writer_builder(&self) -> simd_csv::WriterBuilder {
        let mut builder = simd_csv::WriterBuilder::with_capacity(DEFAULT_BUFFERED_WRITER_CAPACITY);

        builder.delimiter(self.delimiter);

        builder
    }

    fn csv_writer_from_writer<R: Write>(&self, writer: R) -> simd_csv::Writer<R> {
        self.csv_writer_builder().from_writer(writer)
    }

    pub fn csv_writer(&self) -> io::Result<simd_csv::Writer<BoxedWriter>> {
        Ok(self.csv_writer_from_writer(self.writer()?))
    }

    pub fn npy_writer<T: npyz::AutoSerialize>(
        &self,
        dimensions: u64,
    ) -> io::Result<npyz::NpyWriter<T, BufWriter<BoxedSeekableWriter>>> {
        let buf_writer = self.seekable_buf_writer()?;

        npyz::WriteOptions::new()
            .default_dtype()
            .writer(buf_writer)
            .begin_2d(dimensions)
    }

    pub fn vector_writer<T: npyz::AutoSerialize>(
        &self,
        dimensions: u64,
    ) -> io::Result<writers::VectorWriter<T, BoxedWriter, BufWriter<BoxedSeekableWriter>>> {
        Ok(match self.format {
            FileFormat::Csv => writers::VectorWriter::from(self.csv_writer()?),
            FileFormat::Npy => writers::VectorWriter::from(self.npy_writer(dimensions)?),
        })
    }
}
