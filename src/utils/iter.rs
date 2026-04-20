pub struct Chunks<I> {
    size: usize,
    inner: I,
}

impl<I> Iterator for Chunks<I>
where
    I: Iterator,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk: Vec<I::Item> = Vec::with_capacity(self.size);

        while chunk.len() < self.size {
            match self.inner.next() {
                None => {
                    if chunk.is_empty() {
                        return None;
                    }

                    return Some(chunk);
                }
                Some(item) => {
                    chunk.push(item);
                }
            }
        }

        Some(chunk)
    }
}

pub trait IteratorExt: Sized {
    fn chunks(self, size: usize) -> Chunks<Self>;
}

impl<T: Iterator> IteratorExt for T {
    fn chunks(self, size: usize) -> Chunks<Self> {
        debug_assert!(size != 0);

        Chunks { size, inner: self }
    }
}
