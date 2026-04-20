use std::io::{self, Write};

use npyz::NpyWriter;
use simd_csv::{ByteRecord, Writer};

pub enum VectorWriter<T: npyz::Serialize, W: Write> {
    Csv(Box<Writer<W>>),
    Npy(NpyWriter<T, W>),
}

impl<T: npyz::Serialize + ToString, W: Write> VectorWriter<T, W> {
    pub fn write_headers(
        &mut self,
        headers: &ByteRecord,
        dimensions: u64,
        prefix: &str,
    ) -> io::Result<()> {
        if let Self::Csv(writer) = self {
            let mut headers = headers.clone();

            for i in 0..dimensions {
                headers.push_field(format!("{}{}", prefix, i).as_bytes());
            }

            writer.write_byte_record(&headers)?;
        }

        Ok(())
    }

    pub fn write_vector(
        &mut self,
        record: &mut ByteRecord,
        vector: impl Iterator<Item = T>,
    ) -> io::Result<()> {
        match self {
            Self::Csv(writer) => {
                for x in vector {
                    record.push_field(x.to_string().as_bytes());
                }

                writer.write_byte_record(&record)?;

                Ok(())
            }
            Self::Npy(writer) => {
                for x in vector {
                    writer.push(&x)?;
                }

                Ok(())
            }
        }
    }

    pub fn finish(self) -> io::Result<()> {
        match self {
            Self::Csv(mut writer) => writer.flush(),
            Self::Npy(writer) => writer.finish(),
        }
    }
}

impl<T: npyz::Serialize, W: Write> From<Writer<W>> for VectorWriter<T, W> {
    fn from(value: Writer<W>) -> Self {
        Self::Csv(Box::new(value))
    }
}

impl<T: npyz::Serialize, W: Write> From<NpyWriter<T, W>> for VectorWriter<T, W> {
    fn from(value: NpyWriter<T, W>) -> Self {
        Self::Npy(value)
    }
}
