// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use core::cmp;

use crate::Matrix;

use super::{ColumnIter, ColumnIterMut, StarkDomain};
use crypto::{ElementHasher, MerkleTree};
use math::{
    fft::{self, fft_inputs::FftInputs, get_twiddles, permute_index, MIN_CONCURRENT_SIZE},
    log2, polynom, FieldElement, StarkField,
};
use utils::{collections::Vec, uninit_vector};

#[cfg(feature = "concurrent")]
use utils::rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    prelude::*,
};

use rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    prelude::*,
};

#[macro_export]
macro_rules! iter {
    ($e: expr) => {{
        // #[cfg(feature = "concurrent")]
        // let result = $e.par_iter();

        // #[cfg(not(feature = "concurrent"))]
        let result = $e.iter();

        result
    }};
}

// ROWMAJOR MATRIX
// ================================================================================================

pub struct RowMatrix<E>
where
    E: FieldElement,
{
    data: Vec<E>,
    row_width: usize,
}

impl<E> RowMatrix<E>
where
    E: FieldElement,
{
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------
    /// Returns a new [RowMatrix] instantiated with the data from the specified columns.
    pub fn new(data: Vec<E>, row_width: usize) -> Self {
        Self { data, row_width }
    }

    pub fn from_polys(polys: &Matrix<E>, blowup_factor: usize) -> Self {
        let row_width = polys.num_cols();
        let num_rows = polys.num_rows();

        let twiddles = fft::get_twiddles::<E::BaseField>(polys.num_rows() * blowup_factor);
        let domain_size = num_rows * blowup_factor;
        let g = E::BaseField::get_root_of_unity(log2(domain_size));
        let domain_offset = E::BaseField::GENERATOR;

        let mut result_vec_of_arrays =
            unsafe { uninit_vector(num_rows * row_width * blowup_factor) };

        let offsets = (0..num_rows)
            .map(|_i| {
                let idx = permute_index(blowup_factor, 0) as u64;
                g.exp(idx.into()) * domain_offset
            })
            .collect::<Vec<_>>();

        iter!(polys.columns).enumerate().for_each(|(i, row)| {
            let mut factor = E::BaseField::ONE;
            iter!(row).enumerate().for_each(|(j, elem)| {
                result_vec_of_arrays[j * row_width + i] = elem.mul_base(factor);
                factor *= offsets[j];
            })
        });

        let mut row_matrix = RowMatrix::new(result_vec_of_arrays, row_width);

        if cfg!(feature = "concurrent") && polys.num_rows() >= MIN_CONCURRENT_SIZE {
            {
                // #[cfg(feature = "concurrent")]
                // Self::evaluate_poly_with_offset_concurrent(
                //     &matrix,
                //     &twiddles,
                //     E::BaseField::GENERATOR,
                //     blowup_factor,
                // );
            }
        } else {
            Self::evaluate_poly_with_offset(&mut row_matrix, &twiddles);
        }

        row_matrix
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Returns the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.row_width
    }

    /// Returns the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.data.len() / self.row_width
    }

    /// Returns the data stored in this matrix.
    pub fn get_data(&self) -> &[E] {
        &self.data
    }

    // POLYNOMIAL EVALUATION
    // ================================================================================================

    /// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
    /// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
    pub fn evaluate_poly_with_offset(&mut self, twiddles: &[E::BaseField]) {
        self.fft_in_place(twiddles);
        self.permute();
    }

    // CONCURRENT EVALUATION
    // ================================================================================================

    // #[cfg(feature = "concurrent")]
    /// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
    /// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
    ///
    /// This function is only available when the `concurrent` feature is enabled.
    pub fn evaluate_poly_with_offset_concurrent(
        &self,
        twiddles: &[E::BaseField],
        domain_offset: E::BaseField,
        blowup_factor: usize,
    ) -> RowMatrix<E>
    where
        E: FieldElement,
    {
        let domain_size = self.len() * blowup_factor;
        let g = E::BaseField::get_root_of_unity(log2(domain_size));
        let mut result = unsafe { uninit_vector(domain_size * self.row_width) };

        let batch_size = self.len()
            / rayon::current_num_threads()
                .next_power_of_two()
                .min(self.len());

        result
            .par_chunks_mut(self.len() * self.row_width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let idx = permute_index(blowup_factor, i) as u64;
                let offset = g.exp(idx.into()) * domain_offset;

                self.data
                    .par_chunks(batch_size * self.row_width)
                    .zip(chunk.par_chunks_mut(batch_size * self.row_width))
                    .enumerate()
                    .for_each(|(i, (src, dest))| {
                        let mut factor = offset.exp(((i * batch_size) as u64).into());
                        let src_len = src.len() / self.row_width;
                        for d in 0..src_len {
                            for i in 0..self.row_width {
                                dest[d * self.row_width + i] =
                                    src[d * self.row_width + i].mul_base(factor)
                            }
                            factor *= offset;
                        }
                    });
                let mut matrix_chunk = RowMatrixRef {
                    data: chunk,
                    row_width: self.row_width,
                };
                matrix_chunk.split_radix_fft(twiddles);
            });

        let mut matrix_result = RowMatrix {
            data: result,
            row_width: self.row_width,
        };

        FftInputs::permute_concurrent(&mut matrix_result);
        matrix_result
    }
}

/// Implementation of `FftInputs` for `RowMatrix`.
impl<E> FftInputs<E> for RowMatrix<E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixRef<'b, E> where Self: 'b;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;

    fn len(&self) -> usize {
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

    fn swap(&mut self, i: usize, j: usize) {
        let i = i * self.row_width;
        let j = j * self.row_width;

        let (first_row, second_row) = self.data.split_at_mut(j);
        let (first_row, second_row) = (
            &mut first_row[i..i + self.row_width],
            &mut second_row[0..self.row_width],
        );

        // Swap the two rows.
        first_row.swap_with_slice(second_row);
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);
        for d in num_skip..self.len() {
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

    // #[cfg(feature = "concurrent")]
    fn par_mut_chunks(&mut self, chunk_size: usize) -> MatrixChunksMut<'_, E> {
        MatrixChunksMut {
            data: RowMatrixRef {
                data: self.data.as_mut_slice(),
                row_width: self.row_width,
            },
            chunk_size,
        }
    }
}

pub struct RowMatrixRef<'a, E>
where
    E: FieldElement,
{
    data: &'a mut [E],
    row_width: usize,
}

impl<'a, E> RowMatrixRef<'a, E>
where
    E: FieldElement,
{
    pub fn new(data: &'a mut [E], row_width: usize) -> Self {
        Self { data, row_width }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.row_width
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

/// Implementation of `FftInputs` for `RowMatrix`. This is used to perform FFT on the
/// rows of the matrix.
impl<'a, E> FftInputs<E> for RowMatrixRef<'a, E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixRef<'b, E> where Self: 'b;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;

    fn len(&self) -> usize {
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

    fn swap(&mut self, i: usize, j: usize) {
        let i = i * self.row_width;
        let j = j * self.row_width;

        let (first_row, second_row) = self.data.split_at_mut(j);
        let (first_row, second_row) = (
            &mut first_row[i..i + self.row_width],
            &mut second_row[0..self.row_width],
        );

        // Swap the two rows.
        first_row.swap_with_slice(second_row);
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);
        for d in num_skip..self.len() {
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

    // #[cfg(feature = "concurrent")]
    fn par_mut_chunks(&mut self, chunk_size: usize) -> MatrixChunksMut<'_, E> {
        MatrixChunksMut {
            data: RowMatrixRef {
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
    data: RowMatrixRef<'a, E>,
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
    type Item = RowMatrixRef<'a, E>;

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

// #[cfg(feature = "concurrent")]
/// Implement a parallel iterator for MatrixChunksMut. This is a parallel version
/// of the MatrixChunksMut iterator.
impl<'a, E> ParallelIterator for MatrixChunksMut<'a, E>
where
    E: FieldElement + Send,
{
    type Item = RowMatrixRef<'a, E>;

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

// #[cfg(feature = "concurrent")]
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
        self.data.len() / self.chunk_size
    }
}

// #[cfg(feature = "concurrent")]
struct ChunksMutProducer<'a, E>
where
    E: FieldElement,
{
    chunk_size: usize,
    data: RowMatrixRef<'a, E>,
}

// #[cfg(feature = "concurrent")]
impl<'a, E> Producer for ChunksMutProducer<'a, E>
where
    E: FieldElement,
{
    type Item = RowMatrixRef<'a, E>;
    type IntoIter = MatrixChunksMut<'a, E>;

    fn into_iter(self) -> Self::IntoIter {
        MatrixChunksMut {
            data: self.data,
            chunk_size: self.chunk_size,
        }
    }

    fn split_at(mut self, index: usize) -> (Self, Self) {
        let elem_index = cmp::min(index * self.chunk_size, self.data.len());
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

// HELPER FUNCTIONS
// ================================================================================================

/// Transposes a matrix stored in a column major format to a row major format.
fn transpose<E: FieldElement>(matrix: Matrix<E>) -> Vec<Vec<E>> {
    let num_rows = matrix.num_rows();
    let num_cols = matrix.num_cols();
    let mut result = vec![vec![E::ZERO; num_cols]; num_rows];
    for row in 0..num_rows {
        for col in 0..num_cols {
            result[row][col] = matrix.get(col, row);
        }
    }
    result
}
