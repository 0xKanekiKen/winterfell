use crate::FieldElement;
use core::{
    cmp,
    fmt::{self, Debug, Formatter},
    slice::{ChunksMut, Iter},
};

// #[cfg(feature = "concurrent")]
use rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    prelude::*,
};

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

    // #[cfg(feature = "concurrent")]
    /// A parallel iterator over mutable chunks of this fftinputs.
    type ParChunksMut<'c>: IndexedParallelIterator<Item = Self::ChunkItem<'c>>
    where
        Self: 'c,
        E: 'c;

    /// An immutable chunk of this fftinputs.
    type ImChunkItem<'b>: FftInputs<E>
    where
        Self: 'b,
        E: 'b;

    // #[cfg(feature = "concurrent")]
    /// A parallel iterator over immutable chunks of this fftinputs.
    type ParChunks<'c>: IndexedParallelIterator<Item = Self::ImChunkItem<'c>>
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
    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize);

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

    // CONCURRENT METHODS
    // --------------------------------------------------------------------------------------------

    // Need to implement these methods in the implementation of FftInputs for slices, mutable reference to slices
    // and RowMatrix.

    // #[cfg(feature = "concurrent")]
    /// Returns an iterator over chunks of size `chunk_size` of this fftinputs.
    fn par_chunks(&self, chunk_size: usize) -> Self::ParChunks<'_>;

    // #[cfg(feature = "concurrent")]
    /// Returns an iterator over mutable chunks of size `chunk_size` of this fftinputs.
    fn par_mut_chunks(&mut self, chunk_size: usize) -> Self::ParChunksMut<'_>;

    // #[cfg(feature = "concurrent")]
    /// Parallelizes the chone-and-shift-by operation.
    fn par_clone_and_shift_by<S>(&mut self, source: &S, offset: E::BaseField)
    where
        S: FftInputs<E> + ?Sized;
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
    type ParChunksMut<'c> = rayon::slice::ChunksMut<'c, E> where Self: 'c;
    type ImChunkItem<'b> = &'b [E] where E: 'b;
    type ParChunks<'c> = rayon::slice::Chunks<'c, E> where Self: 'c;
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

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let mut offset = E::from(offset);
        let increment = E::from(increment);
        for d in self.iter_mut().skip(num_skip) {
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

    // CONCURRENT METHODS
    // --------------------------------------------------------------------------------------------

    fn par_chunks(&self, chunk_size: usize) -> Self::ParChunks<'_> {
        rayon::prelude::ParallelSlice::par_chunks(self, chunk_size)
    }

    fn par_mut_chunks(&mut self, chunk_size: usize) -> Self::ParChunksMut<'_> {
        self.par_chunks_mut(chunk_size)
    }

    fn par_clone_and_shift_by<S>(&mut self, source: &S, offset: E::BaseField)
    where
        S: FftInputs<E> + ?Sized,
    {
        let batch_size = source.size() / rayon::current_num_threads().next_power_of_two();

        source
            .par_chunks(batch_size)
            .zip(self.par_mut_chunks(batch_size))
            .enumerate()
            .for_each(|(i, (src, dest))| {
                let factor = offset.exp(((i * batch_size) as u64).into());
                dest.clone_and_shift_by(&src, factor, offset);
            });
    }
}

/// An implementation of `FFtInputs` for references to slices of field elements.
impl<'a, E> FftInputs<E> for &'a [E]
where
    E: FieldElement,
{
    type ChunkItem<'b> = &'b mut [E] where Self: 'b;
    type ImChunkItem<'b> = &'b [E] where Self: 'b;
    type ChunksMut<'c> = ChunksMut<'c, E> where Self: 'c;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;
    type ParChunksMut<'c> = rayon::slice::ChunksMut<'c, E> where Self: 'c;
    type ParChunks<'c> = rayon::slice::Chunks<'c, E> where Self: 'c;

    fn size(&self) -> usize {
        <[E] as FftInputs<E>>::size(self)
    }

    fn iter(&self) -> Self::ElementIter<'_> {
        <[E] as FftInputs<E>>::iter(self)
    }

    fn get(&self, idx: usize) -> &E {
        <[E] as FftInputs<E>>::get(self, idx)
    }

    fn butterfly(&mut self, _offset: usize, _stride: usize) {
        unimplemented!()
    }

    fn butterfly_twiddle(
        &mut self,
        _twiddle: <E as FieldElement>::BaseField,
        _offset: usize,
        _stride: usize,
    ) {
        unimplemented!()
    }

    fn swap_elements(&mut self, _i: usize, _j: usize) {
        unimplemented!()
    }

    fn shift_by_series(
        &mut self,
        _offset: <E as FieldElement>::BaseField,
        _increment: <E as FieldElement>::BaseField,
        _num_skip: usize,
    ) {
        unimplemented!()
    }

    fn shift_by(&mut self, _offset: <E as FieldElement>::BaseField) {
        unimplemented!()
    }

    fn clone_and_shift_by<S>(
        &mut self,
        _source: &S,
        _init_offset: <E as FieldElement>::BaseField,
        _offset_factor: <E as FieldElement>::BaseField,
    ) where
        S: FftInputs<E> + ?Sized,
    {
        unimplemented!()
    }

    fn mut_chunks(&mut self, _chunk_size: usize) -> Self::ChunksMut<'_> {
        unimplemented!()
    }

    // CONCURRENT METHODS
    // --------------------------------------------------------------------------------------------

    fn par_chunks(&self, chunk_size: usize) -> Self::ParChunks<'_> {
        <[E] as FftInputs<E>>::par_chunks(self, chunk_size)
    }

    fn par_mut_chunks(&mut self, _chunk_size: usize) -> Self::ParChunksMut<'_> {
        unimplemented!()
    }

    fn par_clone_and_shift_by<S>(&mut self, _source: &S, _offset: <E as FieldElement>::BaseField)
    where
        S: FftInputs<E> + ?Sized,
    {
        unimplemented!()
    }
}

/// An implementation of `FftInputs` for mutable references to slices of field elements.
impl<'a, E> FftInputs<E> for &'a mut [E]
where
    E: FieldElement,
{
    type ChunkItem<'b> = &'b mut [E] where Self: 'b;
    type ImChunkItem<'b> = &'b [E] where Self: 'b;
    type ChunksMut<'c> = ChunksMut<'c, E> where Self: 'c;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;
    type ParChunks<'c> = rayon::slice::Chunks<'c, E> where Self: 'c;
    type ParChunksMut<'c> = rayon::slice::ChunksMut<'c, E> where Self: 'c;

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

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        <[E] as FftInputs<E>>::shift_by_series(self, offset, increment, num_skip)
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

    // CONCURRENT METHODS
    // --------------------------------------------------------------------------------------------

    fn par_chunks(&self, chunk_size: usize) -> Self::ParChunks<'_> {
        <[E] as FftInputs<E>>::par_chunks(self, chunk_size)
    }

    fn par_mut_chunks(&mut self, chunk_size: usize) -> Self::ParChunksMut<'_> {
        <[E] as FftInputs<E>>::par_mut_chunks(self, chunk_size)
    }

    fn par_clone_and_shift_by<S>(&mut self, source: &S, offset: E::BaseField)
    where
        S: FftInputs<E> + ?Sized,
    {
        <[E] as FftInputs<E>>::par_clone_and_shift_by(self, source, offset)
    }
}

/// An iterator over structs that implement the `FftInputs` trait in mutable chunks of
/// these structs.
pub struct RowMajor<'a, E: FieldElement> {
    data: &'a mut [E],
    row_width: usize,
}

/// Debug implementation for `RowMajor`.
impl<'a, E> Debug for RowMajor<'a, E>
where
    E: FieldElement,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowMajor")
            .field("data", &self.data)
            .field("row_width", &self.row_width)
            .finish()
    }
}

/// An implementation of RowMajor for mutable references to slices of field elements.
impl<'a, E> RowMajor<'a, E>
where
    E: FieldElement,
{
    // CONSTRUCTOR
    // ================================================================================================

    /// Creates a new instance of `RowMajor` from the given data and row width.
    pub fn new(data: &'a mut [E], row_width: usize) -> Self {
        debug_assert_ne!(0, row_width);
        Self { data, row_width }
    }

    // PUBLIC ACCESSORS
    // ================================================================================================

    /// Returns the underlying slice of data.
    pub fn get_data(&self) -> &[E] {
        self.data
    }

    /// Returns the number of elements in the underlying slice.
    pub fn len(&self) -> usize {
        self.data.len() / self.row_width
    }

    /// Returns if the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
    type ImChunkItem<'b> = RowMajor<'b, E> where Self: 'b;
    type ChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;
    type ElementIter<'i> = Iter<'i, E> where Self: 'i;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;
    type ParChunks<'c> = MatrixChunksMut<'c, E> where Self: 'c;

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

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);
        for d in num_skip..self.size() {
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

    // CONCURRENT METHODS
    // --------------------------------------------------------------------------------------------

    fn par_chunks(&self, chunk_size: usize) -> Self::ParChunks<'_> {
        // SAFETY: convert a reference to a mutable reference. This is safe because
        // we are not actually mutating the data, we are just pretending to be able to mutate it.
        // This is necessary because the `par_chunks` method is defined on `FftInputs` which is
        // implemented for both `&RowMajor` and `&mut RowMajor`. We need to be able to call `par_chunks` on
        // both references, but we can only implement `par_chunks` on
        // `&mut RowMajor` because it returns a mutable iterator.
        // This is a hack to get around that.
        MatrixChunksMut {
            data: RowMajor {
                data: unsafe {
                    std::slice::from_raw_parts_mut(self.data.as_ptr() as *mut E, self.data.len())
                },
                row_width: self.row_width,
            },
            chunk_size,
        }
    }

    fn par_mut_chunks(&mut self, chunk_size: usize) -> Self::ParChunksMut<'_> {
        MatrixChunksMut {
            data: RowMajor {
                data: self.as_mut_slice(),
                row_width: self.row_width,
            },
            chunk_size,
        }
    }

    fn par_clone_and_shift_by<S>(&mut self, source: &S, offset: E::BaseField)
    where
        S: FftInputs<E> + ?Sized,
    {
        let batch_size = source.size() / rayon::current_num_threads().next_power_of_two();

        source
            .par_chunks(batch_size)
            .zip(self.par_mut_chunks(batch_size))
            .enumerate()
            .for_each(|(i, (src, mut dest))| {
                let factor = offset.exp(((i * batch_size) as u64).into());
                dest.clone_and_shift_by(&src, factor, offset);
            });
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
        self.data.size()
    }
}

impl<'a, E> DoubleEndedIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            return None;
        }
        let at = self.chunk_size.min(self.len());
        let (head, tail) = self.data.split_at_mut(at);
        self.data = head;
        Some(tail)
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

// pub struct ParMatrixChunksMut<'a, E>
// where
//     E: FieldElement,
// {
//     data: RowMajor<'a, E>,
//     chunk_size: usize,
// }

/// Implement a parallel iterator for MatrixChunksMut. This is a parallel version
/// of the MatrixChunksMut iterator.
impl<'a, E> ParallelIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement + Send,
{
    type Item = RowMajor<'a, E>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(rayon::iter::IndexedParallelIterator::len(self))
    }
}

impl<'a, E> IndexedParallelIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement + Send,
{
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = ChunksMutProducer {
            chunk_size: self.chunk_size,
            data: self.data,
        };
        callback.callback(producer)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        div_round_up(self.data.size(), self.chunk_size)
    }
}

struct ChunksMutProducer<'a, E>
where
    E: FieldElement,
{
    chunk_size: usize,
    data: RowMajor<'a, E>,
}

impl<'a, E> Producer for ChunksMutProducer<'a, E>
where
    E: FieldElement,
{
    type Item = RowMajor<'a, E>;
    type IntoIter = MatrixChunksMut<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixChunksMut {
            data: self.data,
            chunk_size: self.chunk_size,
        }
    }

    fn split_at(mut self, index: usize) -> (Self, Self) {
        let elem_index = cmp::min(index * self.chunk_size, self.data.size());
        let (left, right) = self.data.split_at_mut(elem_index);
        (
            ChunksMutProducer {
                chunk_size: self.chunk_size,
                data: left,
            },
            ChunksMutProducer {
                chunk_size: self.chunk_size,
                data: right,
            },
        )
    }
}

#[inline]
pub fn div_round_up(n: usize, divisor: usize) -> usize {
    debug_assert!(divisor != 0, "Division by zero!");
    if n == 0 {
        0
    } else {
        (n - 1) / divisor + 1
    }
}
