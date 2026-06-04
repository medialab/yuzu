use std::io::Read;

use simd_csv::{ByteRecord, Reader};

pub trait ReaderExt {
    fn skip(&mut self, count: u64) -> simd_csv::Result<()>;
}

impl<R: Read> ReaderExt for Reader<R> {
    fn skip(&mut self, count: u64) -> simd_csv::Result<()> {
        let mut skipped: u64 = 0;
        let mut record = ByteRecord::new();

        while self.read_byte_record(&mut record)? {
            skipped += 1;

            if skipped == count {
                break;
            }
        }

        Ok(())
    }
}
