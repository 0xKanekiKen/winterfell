// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::Matrix;
use core::cmp;
use math::{
    fft::{self, fft_inputs::FftInputs, permute_index, MIN_CONCURRENT_SIZE},
    log2, FieldElement, StarkField,
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

// CONSTANTS
// ================================================================================================

pub const ARR_SIZE: usize = 8;

// RowMatrix MATRIX
// ================================================================================================

#[derive(Debug, Clone)]
pub struct RowMatrix<E>
where
    E: FieldElement,
{
    data: Vec<[E; ARR_SIZE]>,
    row_width: usize,
}

impl<E> RowMatrix<E>
where
    E: FieldElement,
{
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------

    pub fn new(data: Vec<[E; ARR_SIZE]>, row_width: usize) -> Self {
        Self { data, row_width }
    }

    pub fn from_polys(polys: &Matrix<E>, blowup_factor: usize) -> Self {
        let row_width = polys.num_cols();
        let num_rows = polys.num_rows();

        let twiddles = fft::get_twiddles::<E::BaseField>(polys.num_rows() * blowup_factor);
        let domain_size = num_rows * blowup_factor;
        let g = E::BaseField::get_root_of_unity(log2(domain_size));
        let domain_offset = E::BaseField::GENERATOR;

        let mut result_vec_of_arrays = unsafe {
            uninit_vector::<[E; ARR_SIZE]>(num_rows * row_width * blowup_factor / ARR_SIZE)
        };

        let offsets = (0..num_rows)
            .map(|_i| {
                let idx = permute_index(blowup_factor, 0) as u64;
                g.exp(idx.into()) * domain_offset
            })
            .collect::<Vec<_>>();

        let time = std::time::Instant::now();

        iter!(polys.columns).enumerate().for_each(|(i, row)| {
            let mut factor = E::BaseField::ONE;
            iter!(row).enumerate().for_each(|(j, elem)| {
                result_vec_of_arrays[(j * row_width + i) / ARR_SIZE][i % ARR_SIZE] =
                    elem.mul_base(factor);
                factor *= offsets[j];
            })
        });

        let transpose_time = time.elapsed().as_millis();
        println!("Time to transpose: {:?}", transpose_time);

        let mut row_matrix = RowMatrix::new(result_vec_of_arrays, row_width);

        if cfg!(feature = "concurrent") && polys.num_rows() >= MIN_CONCURRENT_SIZE {
            {
                // #[cfg(feature = "concurrent")]
                evaluate_poly_with_offset_concurrent(
                    &row_matrix,
                    &twiddles,
                    E::BaseField::GENERATOR,
                    blowup_factor,
                );
            }
        } else {
            evaluate_poly_with_offset(&mut row_matrix, &twiddles);
        }
        println!("time so far: {:?}", time.elapsed().as_millis());
        println!(
            "time taken to evaluate: {:?}",
            time.elapsed().as_millis() - transpose_time
        );

        row_matrix
    }

    // PUBLIC ACCESSORS
    // --------------------------------------------------------------------------------------------

    /// Ruturns the chunks size of the data.
    pub fn arr_size(&self) -> usize {
        ARR_SIZE
    }

    /// Returns the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.row_width
    }

    /// Returns the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.data.len() * self.data[0].len() / self.row_width
    }

    /// Returns the data in this matrix as a slice of arrays.
    pub fn as_data(&self) -> &[[E; ARR_SIZE]] {
        &self.data
    }
}

/// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
/// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
pub fn evaluate_poly_with_offset<E>(row_matrix: &mut RowMatrix<E>, twiddles: &[E::BaseField])
where
    E: FieldElement,
{
    let seg_in_row = row_matrix.row_width / ARR_SIZE;

    let mut matrix_segment = RowMatrixSegment {
        data: row_matrix.data.as_mut_slice(),
        row_width: row_matrix.row_width,
        init_col: 0,
    };

    for seg_idx in 0..seg_in_row {
        matrix_segment.update_segment(seg_idx);
        matrix_segment.fft_in_place(twiddles);
    }

    row_matrix.permute();
}

// #[cfg(feature = "concurrent")]
/// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
/// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
///
/// This function is only available when the `concurrent` feature is enabled.
pub fn evaluate_poly_with_offset_concurrent<E>(
    p: &RowMatrix<E>,
    twiddles: &[E::BaseField],
    domain_offset: E::BaseField,
    blowup_factor: usize,
) -> RowMatrix<E>
where
    E: FieldElement,
{
    let domain_size = p.len() * blowup_factor;
    let g = E::BaseField::get_root_of_unity(log2(domain_size));

    let mut result_vec_of_arrays =
        unsafe { uninit_vector::<[E; ARR_SIZE]>(domain_size * p.row_width / ARR_SIZE) };

    let batch_size = p.len()
        / rayon::current_num_threads()
            .next_power_of_two()
            .min(p.len());

    result_vec_of_arrays
        .par_chunks_mut(p.len() * p.row_width / ARR_SIZE)
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = permute_index(blowup_factor, i) as u64;
            let offset = g.exp(idx.into()) * domain_offset;

            p.data
                .par_chunks(batch_size * p.row_width / ARR_SIZE)
                .zip(chunk.par_chunks_mut(batch_size * p.row_width / ARR_SIZE))
                .enumerate()
                .for_each(|(i, (src, dest))| {
                    let dest_len = dest.len() * ARR_SIZE / p.row_width;
                    let seg_in_row = p.row_width / ARR_SIZE;
                    for d in 0..seg_in_row {
                        let mut factor = offset.exp(((i * batch_size) as u64).into());
                        for row in 0..dest_len {
                            let row_idx = (row * p.row_width / ARR_SIZE) + d;
                            dest[row_idx][0] = src[row_idx][0].mul_base(factor);
                            dest[row_idx][1] = src[row_idx][1].mul_base(factor);
                            dest[row_idx][2] = src[row_idx][2].mul_base(factor);
                            dest[row_idx][3] = src[row_idx][3].mul_base(factor);
                            dest[row_idx][4] = src[row_idx][4].mul_base(factor);
                            dest[row_idx][5] = src[row_idx][5].mul_base(factor);
                            dest[row_idx][6] = src[row_idx][6].mul_base(factor);
                            dest[row_idx][7] = src[row_idx][7].mul_base(factor);

                            factor *= offset;
                        }
                    }
                });
            for d in 0..p.row_width / ARR_SIZE {
                let mut row_matrix_segment_i = RowMatrixSegment::new(chunk, p.row_width, d);
                row_matrix_segment_i.split_radix_fft(twiddles);
            }
        });

    let mut matrix_result = RowMatrix {
        data: result_vec_of_arrays,
        row_width: p.row_width,
    };

    matrix_result.permute_concurrent();
    matrix_result
}

/// Implementation of `FftInputs` for `RowMatrix`.
impl<E> FftInputs<E> for RowMatrix<E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixSegment<'b, E> where Self: 'b;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;

    fn len(&self) -> usize {
        self.data.len() * self.arr_size() / self.row_width
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;

        let arr_in_row = self.row_width / ARR_SIZE;

        let i_index = i * arr_in_row;
        let j_index = j * arr_in_row;

        for vec_idx in 0..arr_in_row {
            let i_vec_idx = i_index + vec_idx;
            let j_vec_idx = j_index + vec_idx;
            let temp = self.data[i_vec_idx];

            //  apply on 1st element of the array.
            self.data[i_vec_idx][0] = temp[0] + self.data[j_vec_idx][0];
            self.data[j_vec_idx][0] = temp[0] - self.data[j_vec_idx][0];

            // apply on 2nd element of the array.
            self.data[i_vec_idx][1] = temp[1] + self.data[j_vec_idx][1];
            self.data[j_vec_idx][1] = temp[1] - self.data[j_vec_idx][1];

            // apply on 3rd element of the array.
            self.data[i_vec_idx][2] = temp[2] + self.data[j_vec_idx][2];
            self.data[j_vec_idx][2] = temp[2] - self.data[j_vec_idx][2];

            // apply on 4th element of the array.
            self.data[i_vec_idx][3] = temp[3] + self.data[j_vec_idx][3];
            self.data[j_vec_idx][3] = temp[3] - self.data[j_vec_idx][3];

            // apply on 5th element of the array.
            self.data[i_vec_idx][4] = temp[4] + self.data[j_vec_idx][4];
            self.data[j_vec_idx][4] = temp[4] - self.data[j_vec_idx][4];

            // apply on 6th element of the array.
            self.data[i_vec_idx][5] = temp[5] + self.data[j_vec_idx][5];
            self.data[j_vec_idx][5] = temp[5] - self.data[j_vec_idx][5];

            // apply on 7th element of the array.
            self.data[i_vec_idx][6] = temp[6] + self.data[j_vec_idx][6];
            self.data[j_vec_idx][6] = temp[6] - self.data[j_vec_idx][6];

            // apply on 8th element of the array.
            self.data[i_vec_idx][7] = temp[7] + self.data[j_vec_idx][7];
            self.data[j_vec_idx][7] = temp[7] - self.data[j_vec_idx][7];
        }
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;

        let twiddle = E::from(twiddle);
        let arr_in_row = self.row_width / ARR_SIZE;

        let i_index = i * arr_in_row;
        let j_index = j * arr_in_row;

        for vec_idx in 0..arr_in_row {
            let i_vec_idx = i_index + vec_idx;
            let j_vec_idx = j_index + vec_idx;
            let temp = self.data[i_vec_idx];

            // apply of index 0 of twiddle.
            self.data[j_vec_idx][0] *= twiddle;
            self.data[i_vec_idx][0] = temp[0] + self.data[j_vec_idx][0];
            self.data[j_vec_idx][0] = temp[0] - self.data[j_vec_idx][0];

            // apply of index 1 of twiddle.
            self.data[j_vec_idx][1] *= twiddle;
            self.data[i_vec_idx][1] = temp[1] + self.data[j_vec_idx][1];
            self.data[j_vec_idx][1] = temp[1] - self.data[j_vec_idx][1];

            // apply of index 2 of twiddle.
            self.data[j_vec_idx][2] *= twiddle;
            self.data[i_vec_idx][2] = temp[2] + self.data[j_vec_idx][2];
            self.data[j_vec_idx][2] = temp[2] - self.data[j_vec_idx][2];

            // apply of index 3 of twiddle.
            self.data[j_vec_idx][3] *= twiddle;
            self.data[i_vec_idx][3] = temp[3] + self.data[j_vec_idx][3];
            self.data[j_vec_idx][3] = temp[3] - self.data[j_vec_idx][3];

            // apply of index 4 of twiddle.
            self.data[j_vec_idx][4] *= twiddle;
            self.data[i_vec_idx][4] = temp[4] + self.data[j_vec_idx][4];
            self.data[j_vec_idx][4] = temp[4] - self.data[j_vec_idx][4];

            // apply of index 5 of twiddle.
            self.data[j_vec_idx][5] *= twiddle;
            self.data[i_vec_idx][5] = temp[5] + self.data[j_vec_idx][5];
            self.data[j_vec_idx][5] = temp[5] - self.data[j_vec_idx][5];

            // apply of index 6 of twiddle.
            self.data[j_vec_idx][6] *= twiddle;
            self.data[i_vec_idx][6] = temp[6] + self.data[j_vec_idx][6];
            self.data[j_vec_idx][6] = temp[6] - self.data[j_vec_idx][6];

            // apply of index 7 of twiddle.
            self.data[j_vec_idx][7] *= twiddle;
            self.data[i_vec_idx][7] = temp[7] + self.data[j_vec_idx][7];
            self.data[j_vec_idx][7] = temp[7] - self.data[j_vec_idx][7];
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        let i = i * self.row_width / ARR_SIZE;
        let j = j * self.row_width / ARR_SIZE;

        let arr_in_row = self.row_width / ARR_SIZE;

        let (first_row, second_row) = self.data.split_at_mut(j);
        let (first_row, second_row) = (
            &mut first_row[i..i + arr_in_row],
            &mut second_row[0..arr_in_row],
        );

        // Swap the two rows.
        first_row.swap_with_slice(second_row);
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);

        let arr_in_row = self.row_width / ARR_SIZE;

        for d in num_skip..self.len() {
            let row_start = d * arr_in_row;
            for idx in 0..arr_in_row {
                // apply on index 0.
                self.data[row_start + idx][0] *= offset;

                // apply on index 1.
                self.data[row_start + idx][1] *= offset;

                // apply on index 2.
                self.data[row_start + idx][2] *= offset;

                // apply on index 3.
                self.data[row_start + idx][3] *= offset;

                // apply on index 4.
                self.data[row_start + idx][4] *= offset;

                // apply on index 5.
                self.data[row_start + idx][5] *= offset;

                // apply on index 6.
                self.data[row_start + idx][6] *= offset;

                // apply on index 7.
                self.data[row_start + idx][7] *= offset;
            }
            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        let offset = E::from(offset);

        let arr_in_row = self.row_width / ARR_SIZE;

        for d in 0..self.len() {
            let row_start = d * arr_in_row;
            for idx in 0..arr_in_row {
                // apply on index 0.
                self.data[row_start + idx][0] *= offset;

                // apply on index 1.
                self.data[row_start + idx][1] *= offset;

                // apply on index 2.
                self.data[row_start + idx][2] *= offset;

                // apply on index 3.
                self.data[row_start + idx][3] *= offset;

                // apply on index 4.
                self.data[row_start + idx][4] *= offset;

                // apply on index 5.
                self.data[row_start + idx][5] *= offset;

                // apply on index 6.
                self.data[row_start + idx][6] *= offset;

                // apply on index 7.
                self.data[row_start + idx][7] *= offset;
            }
        }
    }
    // #[cfg(feature = "concurrent")]
    fn par_mut_chunks(&mut self, _chunk_size: usize) -> MatrixChunksMut<'_, E> {
        unimplemented!("parallelism is not supported in this version of the library")
    }
}

pub struct RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    data: &'a mut [[E; ARR_SIZE]],
    row_width: usize,
    init_col: usize,
}

impl<'a, E> RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    /// Creates a new RowMatrixSegment from a mutable reference to a slice of arrays.
    pub fn new(data: &'a mut [[E; ARR_SIZE]], row_width: usize, init_col: usize) -> Self {
        Self {
            data,
            row_width,
            init_col,
        }
    }

    fn len(&self) -> usize {
        self.data.len() * ARR_SIZE / self.row_width
    }

    fn update_segment(&mut self, seg_idx: usize) {
        assert!(seg_idx < self.row_width / ARR_SIZE);
        self.init_col = seg_idx;
    }

    /// Safe mutable slice cast to avoid unnecessary lifetime complexity.
    fn as_mut_slice(&mut self) -> &'a mut [[E; ARR_SIZE]] {
        let ptr = self.data as *mut [[E; ARR_SIZE]];
        // Safety: we still hold the mutable reference to the slice so no ownership rule is
        // violated.
        unsafe { ptr.as_mut().expect("the initial reference was not valid.") }
    }

    /// Splits the struct into two mutable struct at the given split point. Data of first
    /// chunk will contain elements at indices [0, split_point), and the second chunk
    /// will contain elements at indices [split_point, size).
    fn split_at_mut(&mut self, split_point: usize) -> (Self, Self) {
        let at = split_point * self.row_width / ARR_SIZE;
        let (left, right) = self.as_mut_slice().split_at_mut(at);
        let left = Self::new(left, self.row_width, self.init_col);
        let right = Self::new(right, self.row_width, self.init_col);
        (left, right)
    }
}

/// Implementation of `FftInputs` for `RowMatrix`.
impl<'a, E> FftInputs<E> for RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixSegment<'b, E> where Self: 'b;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;

    fn len(&self) -> usize {
        self.data.len() * ARR_SIZE / self.row_width
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = (offset * self.row_width) / ARR_SIZE + self.init_col;
        let j = ((stride + offset) * self.row_width) / ARR_SIZE + self.init_col;

        let temp = self.data[i];
        let mut data_at_i = self.data[i];
        let mut data_at_j = self.data[j];

        // apply on index 0.
        data_at_i[0] = temp[0] + data_at_j[0];
        data_at_j[0] = temp[0] - data_at_j[0];

        // apply on index 1.
        data_at_i[1] = temp[1] + data_at_j[1];
        data_at_j[1] = temp[1] - data_at_j[1];

        // apply on index 2.
        data_at_i[2] = temp[2] + data_at_j[2];
        data_at_j[2] = temp[2] - data_at_j[2];

        // apply on index 3.
        data_at_i[3] = temp[3] + data_at_j[3];
        data_at_j[3] = temp[3] - data_at_j[3];

        // apply on index 4.
        data_at_i[4] = temp[4] + data_at_j[4];
        data_at_j[4] = temp[4] - data_at_j[4];

        // apply on index 5.
        data_at_i[5] = temp[5] + data_at_j[5];
        data_at_j[5] = temp[5] - data_at_j[5];

        // apply on index 6.
        data_at_i[6] = temp[6] + data_at_j[6];
        data_at_j[6] = temp[6] - data_at_j[6];

        // apply on index 7.
        data_at_i[7] = temp[7] + data_at_j[7];
        data_at_j[7] = temp[7] - data_at_j[7];

        self.data[i] = data_at_i;
        self.data[j] = data_at_j;
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        let i = (offset * self.row_width) / ARR_SIZE + self.init_col;
        let j = ((stride + offset) * self.row_width) / ARR_SIZE + self.init_col;

        let temp = self.data[i];
        let mut data_at_i = self.data[i];
        let mut data_at_j = self.data[j];

        // apply on index 0.
        data_at_i[0] = temp[0] + data_at_j[0].mul_base(twiddle);
        data_at_j[0] = temp[0] - data_at_j[0].mul_base(twiddle);

        // apply on index 1.
        data_at_i[1] = temp[1] + data_at_j[1].mul_base(twiddle);
        data_at_j[1] = temp[1] - data_at_j[1].mul_base(twiddle);

        // apply on index 2.
        data_at_i[2] = temp[2] + data_at_j[2].mul_base(twiddle);
        data_at_j[2] = temp[2] - data_at_j[2].mul_base(twiddle);

        // apply on index 3.
        // data_at_j[3] = data_at_j[3].mul_base(twiddle);
        data_at_i[3] = temp[3] + data_at_j[3].mul_base(twiddle);
        data_at_j[3] = temp[3] - data_at_j[3].mul_base(twiddle);

        // apply on index 4.
        // data_at_j[4] = data_at_j[4].mul_base(twiddle);
        data_at_i[4] = temp[4] + data_at_j[4].mul_base(twiddle);
        data_at_j[4] = temp[4] - data_at_j[4].mul_base(twiddle);

        // apply on index 5.
        // data_at_j[5] = data_at_j[5].mul_base(twiddle);
        data_at_i[5] = temp[5] + data_at_j[5].mul_base(twiddle);
        data_at_j[5] = temp[5] - data_at_j[5].mul_base(twiddle);

        // apply on index 6.
        // data_at_j[6] = data_at_j[6].mul_base(twiddle);
        data_at_i[6] = temp[6] + data_at_j[6].mul_base(twiddle);
        data_at_j[6] = temp[6] - data_at_j[6].mul_base(twiddle);

        // apply on index 7.
        // data_at_j[7] = data_at_j[7].mul_base(twiddle);
        data_at_i[7] = temp[7] + data_at_j[7].mul_base(twiddle);
        data_at_j[7] = temp[7] - data_at_j[7].mul_base(twiddle);

        self.data[i] = data_at_i;
        self.data[j] = data_at_j;
    }

    fn swap(&mut self, i: usize, j: usize) {
        let i = i * self.row_width / ARR_SIZE + self.init_col;
        let j = j * self.row_width / ARR_SIZE + self.init_col;

        self.data.swap(i, j);
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);

        for row in num_skip..self.len() {
            let mut array_at_row = self.data[(row * self.row_width / ARR_SIZE) + self.init_col];

            // apply on index 0.
            array_at_row[0] *= offset;

            // apply on index 1.
            array_at_row[1] *= offset;

            // apply on index 2.
            array_at_row[2] *= offset;

            // apply on index 3.
            array_at_row[3] *= offset;

            // apply on index 4.
            array_at_row[4] *= offset;

            // apply on index 5.
            array_at_row[5] *= offset;

            // apply on index 6.
            array_at_row[6] *= offset;

            // apply on index 7.
            array_at_row[7] *= offset;

            // update the array.
            self.data[(row * self.row_width / ARR_SIZE) + self.init_col] = array_at_row;

            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        let offset = E::from(offset);

        for row in 0..self.len() {
            let mut array_at_row = self.data[(row * self.row_width / ARR_SIZE) + self.init_col];

            // apply on index 0.
            array_at_row[0] *= offset;

            // apply on index 1.
            array_at_row[1] *= offset;

            // apply on index 2.
            array_at_row[2] *= offset;

            // apply on index 3.
            array_at_row[3] *= offset;

            // apply on index 4.
            array_at_row[4] *= offset;

            // apply on index 5.
            array_at_row[5] *= offset;

            // apply on index 6.
            array_at_row[6] *= offset;

            // apply on index 7.
            array_at_row[7] *= offset;

            self.data[(row * self.row_width / ARR_SIZE) + self.init_col] = array_at_row;
        }
    }

    // #[cfg(feature = "concurrent")]
    fn par_mut_chunks(&mut self, chunk_size: usize) -> MatrixChunksMut<'_, E> {
        MatrixChunksMut {
            data: RowMatrixSegment {
                data: self.as_mut_slice(),
                row_width: self.row_width,
                init_col: self.init_col,
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
    data: RowMatrixSegment<'a, E>,
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
    type Item = RowMatrixSegment<'a, E>;

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
    type Item = RowMatrixSegment<'a, E>;

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
    data: RowMatrixSegment<'a, E>,
}

// #[cfg(feature = "concurrent")]
impl<'a, E> Producer for ChunksMutProducer<'a, E>
where
    E: FieldElement,
{
    type Item = RowMatrixSegment<'a, E>;
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

/// Transposes a matrix stored in a column major format into RowMatrix.
pub fn transpose<E>(matrix: &Matrix<E>) -> RowMatrix<E>
where
    E: FieldElement,
{
    let mut result =
        unsafe { uninit_vector::<[E; ARR_SIZE]>(matrix.num_rows() * matrix.num_cols() / ARR_SIZE) };

    let num_rows = matrix.num_cols();

    iter!(matrix.columns).enumerate().for_each(|(i, row)| {
        iter!(row).enumerate().for_each(|(j, elem)| {
            result[(j * num_rows + i) / ARR_SIZE][i % ARR_SIZE] = *elem;
        })
    });

    RowMatrix::new(result, matrix.num_cols())
}
