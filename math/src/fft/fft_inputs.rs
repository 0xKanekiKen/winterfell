use crate::FieldElement;
use core::{cmp, mem::transmute, slice::ChunksMut};

// FFTINPUTS TRAIT
// ================================================================================================

/// Defines a set of inputs to be used in FFT computations.
pub trait FftInputs<E: FieldElement>: Sized {
    /// An iterator over mutable chunks of this fftinputs.
    type ChunksMut: Iterator<Item = Self>;

    /// Returns the number of elements in this input.
    fn size(&self) -> usize;

    /// Combines the result of smaller discrete fourier transforms into a larger DFT.
    fn butterfly(&mut self, offset: usize, stride: usize);

    /// Combines the result of smaller discrete fourier transforms multiplied with a
    /// twiddle factor into a larger DFT.
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize);

    /// Swaps the element at index i with the element at index j.
    fn swap_elements(&mut self, i: usize, j: usize);

    /// Multiplies every element in this input by the product of `init_offset` with
    /// `offset_factor` raise to power i, where i is the index at which the element
    /// is present in fftindex. Specifically:
    ///
    /// elem_{i} = elem_{i} * init_offset * offset_factor^i
    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField);

    /// Multiplies every element in this input by `offset`. Specifically:
    ///
    /// elem_{i} = elem_{i} * offset
    fn shift_by(&mut self, offset: E::BaseField);

    /// Copies the elements from `source` into this input, multiplying each element by
    /// the product of `init_offset` with `offset_factor` raise to power i, where i is
    /// the index at which the element is present in fftindex. Specifically:
    ///
    /// elem_{i} = source_{i} * init_offset * offset_factor^i
    fn clone_and_shift_by(
        &mut self,
        source: &Self,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    );

    /// Returns an iterator over mutable chunks of this fftinputs.
    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut;
}

// FFTINPUTS IMPLEMENTATIONS
// ================================================================================================

/// An implementation of `FftInputs` for slices of field elements.
impl<'a, E: FieldElement> FftInputs<E> for &'a mut [E] {
    type ChunksMut = std::slice::ChunksMut<'a, E>;

    fn size(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;
        let temp = self[i];
        self[i] = temp + self[j];
        self[j] = temp - self[j];
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;
        let temp = self[i];
        self[j] = self[j].mul_base(twiddle);
        self[i] = temp + self[j];
        self[j] = temp - self[j];
    }

    fn swap_elements(&mut self, i: usize, j: usize) {
        self.swap(i, j)
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField) {
        let mut offset = E::from(offset);
        let increment = E::from(increment);
        for d in self.iter_mut() {
            *d *= offset;
            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        let offset = E::from(offset);
        for d in self.iter_mut() {
            *d *= offset;
        }
    }

    fn clone_and_shift_by(
        &mut self,
        source: &Self,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) {
        let mut init_offset = init_offset;
        for (d, c) in self.iter_mut().zip(source.iter()) {
            *d = (*c).mul_base(init_offset);
            init_offset *= offset_factor;
        }
    }

    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut {
        let chunks = <[E]>::chunks_mut(self, chunk_size);
        // Safety: this conversion is safe because `&'a mut [T]: 'a`; the only way to express that
        // is to leak the lifetime of `FftInputs`, and that is undesirable since it will create
        // unnecessary complexity for the users of the trait.
        //
        // We are using transmute to guarantee this conversion is noop.
        unsafe { transmute::<ChunksMut<'_, E>, ChunksMut<'a, E>>(chunks) }
    }
}

/// An iterator over structs that implement the `FftInputs` trait in mutable chunks of
/// these structs.
pub struct MatrixChunks<'a, E: FieldElement> {
    data: &'a mut [E],
    row_width: usize,
}

impl<'a, E: FieldElement> MatrixChunks<'a, E> {
    /// Creates a new instance of `MatrixChunks` from the given data and row width.
    pub fn new(data: &'a mut [E], row_width: usize) -> Self {
        Self { data, row_width }
    }

    /// Splits the struct into two mutable struct at the given split point. Data of first
    /// chunk will contain elements at indices [0, split_point), and the second chunk
    /// will contain elements at indices [split_point, size).
    fn split_at_mut(&'a mut self, split_point: usize) -> (Self, Self) {
        let (left, right) = self.data.split_at_mut(split_point * self.row_width);
        (
            Self::new(left, self.row_width),
            Self::new(right, self.row_width),
        )
    }

    /// Returns an iterator over mutable chunks of this struct.
    pub fn chunks_mut(&'a mut self, chunk_size: usize) -> MatrixChunksMut<'a, E> {
        assert_ne!(chunk_size, 0, "chunks cannot have a size of zero");
        MatrixChunksMut::new(self, chunk_size)
    }
}

impl<'a, E: FieldElement> FftInputs<E> for &'a mut MatrixChunks<'a, E> {
    type ChunksMut = MatrixChunksMut<'a, E>;

    fn size(&self) -> usize {
        self.data.len() / self.row_width
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;

        for col_idx in 0..self.row_width {
            let temp = self.data[self.row_width * i + col_idx];
            self.data[self.row_width * i + col_idx] =
                temp + self.data[self.row_width * j + col_idx];
            self.data[self.row_width * j + col_idx] =
                temp - self.data[self.row_width * j + col_idx];
        }
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;

        for col_idx in 0..self.row_width {
            let temp = self.data[self.row_width * i + col_idx];
            self.data[self.row_width * j + col_idx] =
                self.data[self.row_width * j + col_idx].mul_base(twiddle);
            self.data[self.row_width * i + col_idx] =
                temp + self.data[self.row_width * j + col_idx];
            self.data[self.row_width * j + col_idx] =
                temp - self.data[self.row_width * j + col_idx];
        }
    }

    fn swap_elements(&mut self, i: usize, j: usize) {
        for col_idx in 0..self.row_width {
            self.data
                .swap(self.row_width * i + col_idx, self.row_width * j + col_idx);
        }
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField) {
        let increment = E::from(increment);
        for d in 0..self.size() {
            let mut offset = E::from(offset);
            for i in 0..self.row_width {
                self.data[d * self.row_width + i] *= offset
            }
            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        let offset = E::from(offset);
        for d in self.data.iter_mut() {
            *d *= offset;
        }
    }

    fn clone_and_shift_by(
        &mut self,
        source: &Self,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) {
        let increment = E::from(offset_factor);
        for d in 0..self.size() {
            let mut offset = E::from(init_offset);
            for i in 0..self.row_width {
                self.data[d * self.row_width + i] =
                    source.data[d * self.row_width + i].mul_base(init_offset)
            }
            offset *= increment;
        }
    }

    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut {
        let chunks = <MatrixChunks<E>>::chunks_mut(self, chunk_size);
        // Safety: this conversion is safe because `&'a mut [T]: 'a`; the only way to express that
        // is to leak the lifetime of `FftInputs`, and that is undesirable since it will create
        // unnecessary complexity for the users of the trait.
        //
        // We are using transmute to guarantee this conversion is noop.
        unsafe { transmute::<MatrixChunksMut<'_, E>, MatrixChunksMut<'a, E>>(chunks) }
    }
}

/// A mutable iterator over chunks of a mutable FftInputs. This struct is created
///  by the `chunks_mut` method on `FftInputs`.
pub struct MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    data: &'a mut MatrixChunks<'a, E>,
    chunk_size: usize,
}

impl<'a, E> MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    /// Creates a new `MutChunks` iterator.
    pub fn new(inputs: &'a mut MatrixChunks<E>, chunk_size: usize) -> Self {
        MatrixChunksMut {
            data: inputs,
            chunk_size,
        }
    }
}

impl<'a, E> ExactSizeIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    fn len(&self) -> usize {
        self.data.size()
    }
}

impl<'a, E> Iterator for MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    type Item = &'a mut MatrixChunks<'a, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        } else {
            let at = cmp::min(self.chunk_size, self.len());
            let (mut head, mut tail) = self.data.split_at_mut(at);
            self.data = &mut tail;
            Some(&mut head)
        }
    }
}
