use crate::FieldElement;
use core::slice::{ChunksMut, Iter};

// FFTINPUTS TRAIT
// ================================================================================================

/// Defines a set of inputs to be used in FFT computations.
pub trait FftInputs<E: FieldElement> {
    /// A chunk of this fftinputs.
    type ChunkItem<'b>: FftInputs<E>
    where
        Self: 'b,
        E: 'b;

    /// An iterator over mutable chunks of this fftinputs.
    type ChunksMut<'c>: Iterator<Item = Self::ChunkItem<'c>>
    where
        Self: 'c,
        E: 'c;

    /// An iterator over elements of this fftinputs.
    type ElementIter<'i>: Iterator<Item = &'i E>
    where
        Self: 'i,
        E: 'i;

    /// Returns the number of elements in this input.
    fn size(&self) -> usize;

    /// Returns an iterator over elements of this fftinputs.
    fn iter(&self) -> Self::ElementIter<'_>;

    /// Returns a reference to the element at index `idx`.
    fn get(&self, idx: usize) -> &E;

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
    fn clone_and_shift_by<S>(
        &mut self,
        source: &S,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) where
        S: FftInputs<E> + ?Sized;

    /// Returns an iterator over mutable chunks of this fftinputs.
    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut<'_>;
}

// FFTINPUTS IMPLEMENTATIONS
// ================================================================================================

/// An implementation of `FftInputs` for slices of field elements.
impl<E> FftInputs<E> for [E]
where
    E: FieldElement,
{
    type ChunkItem<'b> = &'b mut [E] where E: 'b;
    type ChunksMut<'a> = ChunksMut<'a, E> where Self: 'a;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;

    fn size(&self) -> usize {
        self.len()
    }

    fn iter(&self) -> Self::ElementIter<'_> {
        self.iter()
    }

    fn get(&self, idx: usize) -> &E {
        &self[idx]
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

    fn clone_and_shift_by<S>(
        &mut self,
        source: &S,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) where
        S: FftInputs<E> + ?Sized,
    {
        let mut init_offset = init_offset;
        for (d, c) in self.iter_mut().zip(source.iter()) {
            *d = (*c).mul_base(init_offset);
            init_offset *= offset_factor;
        }
    }

    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut<'_> {
        self.chunks_mut(chunk_size)
    }
}

/// An implementation of `FftInputs` for mutable references to slices of field elements.
impl<'a, E> FftInputs<E> for &'a mut [E]
where
    E: FieldElement,
{
    type ChunkItem<'b> = &'b mut [E] where Self: 'b;
    type ChunksMut<'c> = ChunksMut<'c, E> where Self: 'c;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;

    fn size(&self) -> usize {
        <[E] as FftInputs<E>>::size(self)
    }

    fn iter(&self) -> Self::ElementIter<'_> {
        <[E] as FftInputs<E>>::iter(self)
    }

    fn get(&self, idx: usize) -> &E {
        <[E] as FftInputs<E>>::get(self, idx)
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        <[E] as FftInputs<E>>::butterfly(self, offset, stride)
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        <[E] as FftInputs<E>>::butterfly_twiddle(self, twiddle, offset, stride)
    }

    fn swap_elements(&mut self, i: usize, j: usize) {
        <[E] as FftInputs<E>>::swap_elements(self, i, j)
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField) {
        <[E] as FftInputs<E>>::shift_by_series(self, offset, increment)
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        <[E] as FftInputs<E>>::shift_by(self, offset)
    }

    fn clone_and_shift_by<S>(
        &mut self,
        source: &S,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) where
        S: FftInputs<E> + ?Sized,
    {
        <[E] as FftInputs<E>>::clone_and_shift_by(self, source, init_offset, offset_factor)
    }

    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut<'_> {
        <[E] as FftInputs<E>>::mut_chunks(self, chunk_size)
    }
}

/// An iterator over structs that implement the `FftInputs` trait in mutable chunks of
/// these structs.
pub struct RowMajor<'a, E: FieldElement> {
    data: &'a mut [E],
    row_width: usize,
}

/// An implementation of RowMajor for mutable references to slices of field elements.
impl<'a, E> RowMajor<'a, E>
where
    E: FieldElement,
{
    /// Creates a new instance of `RowMajor` from the given data and row width.
    pub fn new(data: &'a mut [E], row_width: usize) -> Self {
        debug_assert_ne!(0, row_width);
        Self { data, row_width }
    }

    /// Returns the number of elements in the underlying slice.
    fn len(&self) -> usize {
        self.data.len()
    }

    /// Safe mutable slice cast to avoid unnecessary lifetime complexity.
    fn as_mut_slice(&mut self) -> &'a mut [E] {
        let ptr = self.data as *mut [E];
        // Safety: we still hold the mutable reference to the slice so no ownership rule is
        // violated.
        unsafe { ptr.as_mut().expect("the initial reference was not valid.") }
    }

    /// Splits the struct into two mutable struct at the given split point. Data of first
    /// chunk will contain elements at indices [0, split_point), and the second chunk
    /// will contain elements at indices [split_point, size).
    fn split_at_mut(&mut self, split_point: usize) -> (Self, Self) {
        let at = split_point * self.row_width;
        let (left, right) = self.as_mut_slice().split_at_mut(at);
        let left = Self::new(left, self.row_width);
        let right = Self::new(right, self.row_width);
        (left, right)
    }
}

/// Implementation of `FftInputs` for `RowMajor`.
impl<'a, E: FieldElement> FftInputs<E> for RowMajor<'a, E> {
    type ChunkItem<'b> = RowMajor<'b, E> where Self: 'b;
    type ChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;

    fn size(&self) -> usize {
        self.data.len() / self.row_width
    }

    fn iter(&self) -> Self::ElementIter<'_> {
        self.data.iter()
    }

    fn get(&self, idx: usize) -> &E {
        &self.data[idx]
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

    fn clone_and_shift_by<S>(
        &mut self,
        source: &S,
        init_offset: E::BaseField,
        offset_factor: E::BaseField,
    ) where
        S: FftInputs<E> + ?Sized,
    {
        let increment = E::from(offset_factor);
        for d in 0..self.size() {
            let mut offset = E::from(init_offset);
            for i in 0..self.row_width {
                self.data[d * self.row_width + i] =
                    source.get(d * self.row_width + i).mul_base(init_offset)
            }
            offset *= increment;
        }
    }

    fn mut_chunks(&mut self, chunk_size: usize) -> Self::ChunksMut<'_> {
        MatrixChunksMut {
            data: RowMajor {
                data: self.as_mut_slice(),
                row_width: self.row_width,
            },
            chunk_size,
        }
    }
}

/// A mutable iterator over chunks of a mutable FftInputs. This struct is created
///  by the `chunks_mut` method on `FftInputs`.
pub struct MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    data: RowMajor<'a, E>,
    chunk_size: usize,
}

impl<'a, E> ExactSizeIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a, E: FieldElement> Iterator for MatrixChunksMut<'a, E> {
    type Item = RowMajor<'a, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }
        let at = self.chunk_size.min(self.len());
        let (head, tail) = self.data.split_at_mut(at);
        self.data = tail;
        Some(head)
    }
}
