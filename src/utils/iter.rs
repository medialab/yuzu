use crate::utils::io::DynamicUsize;

pub struct Chunks<I> {
    size: Option<usize>,
    inner: I,
}

impl<I> Iterator for Chunks<I>
where
    I: Iterator,
{
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.size {
            Some(size) => {
                let mut chunk: Vec<I::Item> = Vec::with_capacity(size);

                while chunk.len() < size {
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
            None => {
                let mut total: Vec<I::Item> = Vec::new();

                for item in self.inner.by_ref() {
                    total.push(item);
                }

                if total.is_empty() { None } else { Some(total) }
            }
        }
    }
}

pub trait IteratorExt: Sized {
    fn chunks(self, size: usize) -> Chunks<Self>;

    fn chunks_or_total(self, size: DynamicUsize) -> Chunks<Self>;
}

impl<T: Iterator> IteratorExt for T {
    fn chunks(self, size: usize) -> Chunks<Self> {
        debug_assert!(size != 0);

        Chunks {
            size: Some(size),
            inner: self,
        }
    }

    fn chunks_or_total(self, size: DynamicUsize) -> Chunks<Self> {
        match size {
            DynamicUsize::Limited(size) => Chunks {
                size: Some(size.into()),
                inner: self,
            },
            DynamicUsize::Unlimited => Chunks {
                size: None,
                inner: self,
            },
        }
    }
}
