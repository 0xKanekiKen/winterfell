// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use core::cmp;

use super::{ColumnIter, ColumnIterMut, StarkDomain};
use crypto::{ElementHasher, MerkleTree};
use math::{
    fft::{self, fft_inputs::FftInputs, permute_index, MIN_CONCURRENT_SIZE},
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
    /// Returns a new [RowMatrix] instantiated with the data from the specified columns.
    ///
    /// # Panics
    /// Panics if:
    /// * The provided rows of data is empty.
    /// * The remainder of the length of the data and the row width is not zero.
    /// * Number of rows is smaller than or equal to 1.
    /// * Number of rows is not a power of two.
    pub fn new(data: Vec<[E; ARR_SIZE]>, row_width: usize) -> Self {
        // assert!(
        //     !data.is_empty(),
        //     "a matrix must contain at least one column"
        // );
        // assert!(
        //     data.len() % row_width == 0,
        //     "the length of the data should be a multiple of the row width"
        // );
        // assert!(
        //     data.len() / row_width > 1,
        //     "number of rows should be greater than 1"
        // );
        // assert!(
        //     (data.len() / row_width) & (data.len() / row_width - 1) == 0,
        //     "number of rows should be a power of 2"
        // );

        Self { data, row_width }
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

    /// Returns the data in this matrix as a mutable slice of arrays.
    pub fn as_data_mut(&mut self) -> &mut [[E; ARR_SIZE]] {
        &mut self.data
    }

    /// Returns the data in this matrix as a slice of arrays.
    pub fn as_data(&self) -> &[[E; ARR_SIZE]] {
        &self.data
    }

    /// Returns the element located at the specified column and row indexes in this matrix.
    ///
    /// # Panics
    /// Panics if either `col_idx` or `row_idx` are out of bounds for this matrix.
    pub fn get(&self, col_idx: usize, row_idx: usize) -> E {
        self.data[row_idx * self.row_width / ARR_SIZE + col_idx / ARR_SIZE][col_idx % ARR_SIZE]
    }

    /// Set the cell in this matrix at the specified column and row indexes to the provided value.
    ///
    /// # Panics
    /// Panics if either `col_idx` or `row_idx` are out of bounds for this matrix.
    pub fn set(&mut self, col_idx: usize, row_idx: usize, value: E) {
        self.data[row_idx * self.row_width / ARR_SIZE + col_idx / ARR_SIZE][col_idx % ARR_SIZE] =
            value;
    }

    /// Returns a reference to the column at the specified index.
    pub fn get_column(&self, _col_idx: usize) -> &[E] {
        unimplemented!("not implemented yet")
    }

    /// Returns a reference to the column at the specified index.
    pub fn get_column_mut(&mut self, _col_idx: usize) -> &mut [E] {
        unimplemented!("not implemented yet")
    }

    /// Copies values of all columns at the specified row into the specified row slice.
    ///
    /// # Panics
    /// Panics if `row_idx` is out of bounds for this matrix.
    pub fn read_row_into(&self, row_idx: usize, row: &mut [E]) {
        row.iter_mut().enumerate().for_each(|(i, x)| {
            *x = self.data[row_idx * self.row_width / self.arr_size() + i / self.arr_size()]
                [i % self.arr_size()]
        });
    }

    /// Updates a row in this matrix at the specified index to the provided data.
    ///
    /// # Panics
    /// Panics if `row_idx` is out of bounds for this matrix.
    pub fn update_row(&mut self, row_idx: usize, row: &[E]) {
        let row_start = row_idx * self.row_width;
        let vector_width = self.arr_size();

        row.iter()
            .enumerate()
            .for_each(|(i, x)| self.data[(row_start + i) / vector_width][i % vector_width] = *x);
    }

    /// Returns the columns of this matrix as a list of vectors.
    pub fn into_columns(&self) -> Vec<Vec<E>> {
        let mut columns = vec![Vec::new(); self.row_width];
        let vector_width = self.arr_size();

        for i in 0..self.data.len() {
            for j in 0..self.row_width {
                columns[j].push(self.data[i][j % vector_width]);
            }
        }
        columns
    }

    /// Returns the column at the specified index.
    pub fn into_column(&self, index: usize) -> Vec<E> {
        let mut column = Vec::new();
        let vector_width = self.arr_size();

        for i in 0..self.data.len() {
            column.push(self.data[i][index % vector_width]);
        }
        column
    }

    // PUBLIC ACCESSORS
    // ================================================================================================

    /// Returns the underlying slice of data.
    pub fn get_data(&self) -> &Vec<[E; ARR_SIZE]> {
        &self.data
    }

    /// Returns the number of elements in the underlying slice.
    pub fn len(&self) -> usize {
        self.data.len() * ARR_SIZE / self.row_width
    }

    /// Returns if the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // ITERATION
    // --------------------------------------------------------------------------------------------

    /// Returns an iterator over the columns of this matrix.
    pub fn columns(&self) -> ColumnIter<E> {
        unimplemented!("not implemented yet")
    }

    /// Returns a mutable iterator over the columns of this matrix.
    pub fn columns_mut(&mut self) -> ColumnIterMut<E> {
        unimplemented!("not implemented yet")
    }

    // POLYNOMIAL METHODS
    // --------------------------------------------------------------------------------------------

    /// Interpolates columns of the matrix into polynomials in coefficient form and returns the
    /// result.
    ///
    /// The interpolation is performed as follows:
    /// * Each column of the matrix is interpreted as evaluations of degree `num_rows - 1`
    ///   polynomial over a subgroup of size `num_rows`.
    /// * Then each column is interpolated using iFFT algorithm into a polynomial in coefficient
    ///   form.
    /// * The resulting polynomials are returned as a single matrix where each column contains
    ///   coefficients of a degree `num_rows - 1` polynomial.
    pub fn interpolate_columns(&mut self) -> Self {
        unimplemented!("not implemented yet")
    }

    /// Interpolates columns of the matrix into polynomials in coefficient form and returns the
    /// result. The input matrix is consumed in the process.
    ///
    /// The interpolation is performed as follows:
    /// * Each column of the matrix is interpreted as evaluations of degree `num_rows - 1`
    ///   polynomial over a subgroup of size `num_rows`.
    /// * Then each column is interpolated (in place) using iFFT algorithm into a polynomial in
    ///   coefficient form.
    /// * The resulting polynomials are returned as a single matrix where each column contains
    ///   coefficients of a degree `num_rows - 1` polynomial.
    pub fn interpolate_columns_into(mut self) -> Self {
        let inv_twiddles = fft::get_inv_twiddles::<E::BaseField>(self.num_rows());
        assert_eq!(
            self.len(),
            inv_twiddles.len() * 2,
            "invalid number of twiddles: expected {} but received {}",
            self.len() / 2,
            inv_twiddles.len()
        );
        assert!(
            log2(self.len()) <= E::BaseField::TWO_ADICITY,
            "multiplicative subgroup of size {} does not exist in the specified base field",
            self.len()
        );

        // when `concurrent` feature is enabled, run the concurrent version of interpolate_poly;
        // unless the number of evaluations is small, then don't bother with the concurrent version
        if cfg!(feature = "concurrent") && self.len() >= MIN_CONCURRENT_SIZE {
            #[cfg(feature = "concurrent")]
            interpolate_poly_concurrent(&mut self, inv_twiddles);
        } else {
            Self::interpolate_poly(&mut self, &inv_twiddles);
        }

        self
    }

    /// Evaluates polynomials contained in the columns of this matrix over the specified domain
    /// and returns the result.
    ///
    /// The evaluation is done as follows:
    /// * Each column of the matrix is interpreted as coefficients of degree `num_rows - 1`
    ///   polynomial.
    /// * These polynomials are evaluated over the LDE domain defined by the specified
    ///   [StarkDomain] using FFT algorithm. The domain specification includes the size of the
    ///   subgroup as well as the domain offset (to define a coset).
    /// * The resulting evaluations are returned in a new Matrix.
    pub fn evaluate_columns_over(&self, domain: &StarkDomain<E::BaseField>) -> RowMatrix<E> {
        let blowup_factor = domain.trace_to_lde_blowup();
        let domain_offset = domain.offset();
        let twiddles = domain.trace_twiddles();

        assert!(
            self.len().is_power_of_two(),
            "number of coefficients must be a power of 2"
        );
        assert!(
            blowup_factor.is_power_of_two(),
            "blowup factor must be a power of 2"
        );
        assert_eq!(
            self.len(),
            twiddles.len() * 2,
            "invalid number of twiddles: expected {} but received {}",
            self.len() / 2,
            twiddles.len()
        );
        assert!(
            log2(self.len() * blowup_factor) <= E::BaseField::TWO_ADICITY,
            "multiplicative subgroup of size {} does not exist in the specified base field",
            self.len() * blowup_factor
        );
        assert_ne!(
            domain_offset,
            E::BaseField::ZERO,
            "domain offset cannot be zero"
        );

        let mut result = RowMatrix::new(Vec::new(), self.row_width);

        // when `concurrent` feature is enabled, run the concurrent version of the function; unless
        // the polynomial is small, then don't bother with the concurrent version
        //
        // Cant return RowMatrix because result was created in this function. So we are returning a
        // Vec<E> instead. The caller of this function should convert it to a RowMatrix.
        if cfg!(feature = "concurrent") && self.len() >= MIN_CONCURRENT_SIZE {
            {
                // #[cfg(feature = "concurrent")]
                evaluate_poly_with_offset_concurrent(self, twiddles, domain_offset, blowup_factor);
            }
        } else {
            result = evaluate_poly_with_offset(self, twiddles, domain_offset, blowup_factor);
        }

        result
    }

    /// Evaluates polynomials contained in the columns of this matrix at a single point `x`.
    pub fn evaluate_columns_at<F>(&self, x: F) -> Vec<F>
    where
        F: FieldElement + From<E>,
    {
        self.into_columns()
            .iter()
            .map(|p| polynom::eval(p, x))
            .collect()
    }

    // COMMITMENTS
    // --------------------------------------------------------------------------------------------

    /// Returns a commitment to this matrix.
    ///
    /// The commitment is built as follows:
    /// * Each row of the matrix is hashed into a single digest of the specified hash function.
    /// * The resulting values are used to built a binary Merkle tree such that each row digest
    ///   becomes a leaf in the tree. Thus, the number of leaves in the tree is equal to the
    ///   number of rows in the matrix.
    /// * The resulting Merkle tree is return as the commitment to the entire matrix.
    pub fn commit_to_rows<H>(&self) -> MerkleTree<H>
    where
        H: ElementHasher<BaseField = E::BaseField>,
    {
        // // allocate vector to store row hashes
        // let mut row_hashes = unsafe { uninit_vector::<H::Digest>(self.num_rows()) };

        // // iterate though matrix rows, hashing each row; the hashing is done by first copying a
        // // row into row_buf to avoid heap allocations, and then by applying the hash function to
        // // the buffer.
        // batch_iter_mut!(
        //     &mut row_hashes,
        //     128, // min batch size
        //     |batch: &mut [H::Digest], batch_offset: usize| {
        //         let mut row_buf = vec![E::ZERO; self.num_cols()];
        //         for (i, row_hash) in batch.iter_mut().enumerate() {
        //             self.read_row_into(i + batch_offset, &mut row_buf);
        //             *row_hash = H::hash_elements(&row_buf);
        //         }
        //     }
        // );

        // // build Merkle tree out of hashed rows
        // MerkleTree::new(row_hashes).expect("failed to construct trace Merkle tree")
        unimplemented!("TODO")
    }

    // POLYNOMIAL EVALUATION
    // ================================================================================================

    /// Evaluates polynomial `p` in-place over the domain of length `p.len()` in the field specified
    /// by `B` using the FFT algorithm.
    pub fn evaluate_poly(p: &mut RowMatrix<E>, twiddles: &[E::BaseField]) {
        for i in 0..p.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(p.data.as_mut_slice(), p.row_width, i);
            row_matrix_segment_i.fft_in_place(twiddles);
        }
        p.permute();
    }

    // POLYNOMIAL INTERPOLATION
    // ================================================================================================

    /// Interpolates `evaluations` over a domain of length `evaluations.len()` in the field specified
    /// `B` into a polynomial in coefficient form using the FFT algorithm.
    pub fn interpolate_poly(evaluations: &mut RowMatrix<E>, inv_twiddles: &[E::BaseField]) {
        let inv_length = E::BaseField::inv((evaluations.len() as u64).into());
        for i in 0..evaluations.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(evaluations.data.as_mut_slice(), evaluations.row_width, i);
            row_matrix_segment_i.fft_in_place(inv_twiddles);
        }
        evaluations.shift_by(inv_length);
        evaluations.permute();
    }

    /// Interpolates `evaluations` over a domain of length `evaluations.len()` and shifted by
    /// `domain_offset` in the field specified by `B` into a polynomial in coefficient form using
    /// the FFT algorithm.
    pub fn interpolate_poly_with_offset(
        evaluations: &mut RowMatrix<E>,
        inv_twiddles: &[E::BaseField],
        domain_offset: E::BaseField,
    ) {
        let domain_offset = E::BaseField::inv(domain_offset);
        let offset = E::BaseField::inv((evaluations.len() as u64).into());

        for i in 0..evaluations.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(evaluations.data.as_mut_slice(), evaluations.row_width, i);
            row_matrix_segment_i.fft_in_place(inv_twiddles);
        }
        evaluations.permute();

        // Use fftinputs's shift_by_series on evaluations.
        evaluations.shift_by_series(offset, domain_offset, 0);
    }

    // CONCURRENT EVALUATION
    // ================================================================================================

    // #[cfg(feature = "concurrent")]
    /// Evaluates polynomial `p` over the domain of length `p.len()` in the field specified `B` using
    /// the FFT algorithm and returns the result.
    ///
    /// This function is only available when the `concurrent` feature is enabled.
    pub fn evaluate_poly_concurrent(p: &mut RowMatrix<E>, twiddles: &[E::BaseField]) {
        for i in 0..p.row_width / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(p.data.as_mut_slice(), p.row_width, i);
            row_matrix_segment_i.split_radix_fft(twiddles);
        }
        p.permute_concurrent();
    }

    // CONCURRENT INTERPOLATION
    // ================================================================================================

    // #[cfg(feature = "concurrent")]
    /// Interpolates `evaluations` over a domain of length `evaluations.len()` in the field specified
    /// `B` into a polynomial in coefficient form using the FFT algorithm.
    ///
    /// This function is only available when the `concurrent` feature is enabled.
    pub fn interpolate_poly_concurrent(
        evaluations: &mut RowMatrix<E>,
        inv_twiddles: &[E::BaseField],
    ) {
        let inv_length = E::BaseField::inv((evaluations.len() as u64).into());
        let batch_size = evaluations.len()
            / rayon::current_num_threads()
                .next_power_of_two()
                .min(evaluations.len());

        for i in 0..evaluations.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(evaluations.data.as_mut_slice(), evaluations.row_width, i);
            row_matrix_segment_i.split_radix_fft(inv_twiddles);

            rayon::iter::IndexedParallelIterator::enumerate(
                row_matrix_segment_i.par_mut_chunks(batch_size),
            )
            .for_each(|(_i, mut batch)| batch.shift_by(inv_length));
            row_matrix_segment_i.permute_concurrent();
        }
    }

    // #[cfg(feature = "concurrent")]
    /// Interpolates `evaluations` over a domain of length `evaluations.len()` and shifted by
    /// `domain_offset` in the field specified by `B` into a polynomial in coefficient form using
    /// the FFT algorithm.
    ///
    /// This function is only available when the `concurrent` feature is enabled.
    pub fn interpolate_poly_with_offset_concurrent(
        evaluations: &mut RowMatrix<E>,
        inv_twiddles: &[E::BaseField],
        domain_offset: E::BaseField,
    ) {
        let domain_offset = E::BaseField::inv(domain_offset);
        let inv_length = E::BaseField::inv((evaluations.len() as u64).into());

        let batch_size = evaluations.len()
            / rayon::current_num_threads()
                .next_power_of_two()
                .min(evaluations.len());

        for i in 0..evaluations.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i =
                RowMatrixSegment::new(evaluations.data.as_mut_slice(), evaluations.row_width, i);
            row_matrix_segment_i.split_radix_fft(inv_twiddles);
            row_matrix_segment_i.permute_concurrent();
            rayon::iter::IndexedParallelIterator::enumerate(
                row_matrix_segment_i.par_mut_chunks(batch_size),
            )
            .for_each(|(i, mut batch)| {
                let offset = domain_offset.exp(((i * batch_size) as u64).into()) * inv_length;
                batch.shift_by_series(offset, domain_offset, 0);
            });
        }
    }

    // fn par_column_iter_mut(&'a mut self) -> impl ParallelIterator<Item = &'a mut E> {
    //     (0..self.row_width)
    //         .into_par_iter()
    //         .map(move |col_idx| {
    //             (0..self.row_width)
    //                 .map(move |row_idx| &mut self.data[row_idx * self.row_width + col_idx])
    //         })
    //         .flatten_mut()
    // }
}

/// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
/// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
pub fn evaluate_poly_with_offset<E>(
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

    result_vec_of_arrays
        .chunks_mut(p.len() * p.row_width / ARR_SIZE)
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = fft::permute_index(blowup_factor, i) as u64;
            let offset = g.exp(idx.into()) * domain_offset;

            let chunk_len = chunk.len() * ARR_SIZE / p.row_width;
            let seg_in_row = p.row_width / ARR_SIZE;
            for d in 0..seg_in_row {
                let mut factor = E::BaseField::ONE;
                for row in 0..chunk_len {
                    let row_idx = (row * p.row_width / ARR_SIZE) + d;
                    chunk[row_idx][0] = p.data[row_idx][0].mul_base(factor);
                    chunk[row_idx][1] = p.data[row_idx][1].mul_base(factor);
                    chunk[row_idx][2] = p.data[row_idx][2].mul_base(factor);
                    chunk[row_idx][3] = p.data[row_idx][3].mul_base(factor);
                    chunk[row_idx][4] = p.data[row_idx][4].mul_base(factor);
                    chunk[row_idx][5] = p.data[row_idx][5].mul_base(factor);
                    chunk[row_idx][6] = p.data[row_idx][6].mul_base(factor);
                    chunk[row_idx][7] = p.data[row_idx][7].mul_base(factor);

                    factor *= offset;
                }
                let mut matrix_chunk = RowMatrixSegment {
                    data: chunk,
                    row_width: p.row_width,
                    init_col: d,
                };
                FftInputs::fft_in_place(&mut matrix_chunk, twiddles);
            }
        });

    let mut matrix_result = RowMatrix {
        data: result_vec_of_arrays,
        row_width: p.row_width,
    };

    FftInputs::permute(&mut matrix_result);
    matrix_result
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

                // // apply on index 4.
                // self.data[row_start + idx][4] *= offset;

                // // apply on index 5.
                // self.data[row_start + idx][5] *= offset;

                // // apply on index 6.
                // self.data[row_start + idx][6] *= offset;

                // // apply on index 7.
                // self.data[row_start + idx][7] *= offset;
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
        let sum_0 = temp[0] + data_at_j[0];
        let diff_0 = temp[0] - data_at_j[0];
        data_at_i[0] = sum_0;
        data_at_j[0] = diff_0;

        // apply on index 1.
        let sum_1 = temp[1] + data_at_j[1];
        let diff_1 = temp[1] - data_at_j[1];
        data_at_i[1] = sum_1;
        data_at_j[1] = diff_1;

        // apply on index 2.
        let sum_2 = temp[2] + data_at_j[2];
        let diff_2 = temp[2] - data_at_j[2];
        data_at_i[2] = sum_2;
        data_at_j[2] = diff_2;

        // apply on index 3.
        let sum_3 = temp[3] + data_at_j[3];
        let diff_3 = temp[3] - data_at_j[3];
        data_at_i[3] = sum_3;
        data_at_j[3] = diff_3;

        // apply on index 4.
        let sum_4 = temp[4] + data_at_j[4];
        let diff_4 = temp[4] - data_at_j[4];
        data_at_i[4] = sum_4;
        data_at_j[4] = diff_4;

        // apply on index 5.
        let sum_5 = temp[5] + data_at_j[5];
        let diff_5 = temp[5] - data_at_j[5];
        data_at_i[5] = sum_5;
        data_at_j[5] = diff_5;

        // apply on index 6.
        let sum_6 = temp[6] + data_at_j[6];
        let diff_6 = temp[6] - data_at_j[6];
        data_at_i[6] = sum_6;
        data_at_j[6] = diff_6;

        // apply on index 7.
        let sum_7 = temp[7] + data_at_j[7];
        let diff_7 = temp[7] - data_at_j[7];
        data_at_i[7] = sum_7;
        data_at_j[7] = diff_7;

        // // apply on index 0.
        // data_at_i[0] = temp[0] + data_at_j[0];
        // data_at_j[0] = temp[0] - data_at_j[0];

        // // apply on index 1.
        // data_at_i[1] = temp[1] + data_at_j[1];
        // data_at_j[1] = temp[1] - data_at_j[1];

        // // apply on index 2.
        // data_at_i[2] = temp[2] + data_at_j[2];
        // data_at_j[2] = temp[2] - data_at_j[2];

        // // apply on index 3.
        // data_at_i[3] = temp[3] + data_at_j[3];
        // data_at_j[3] = temp[3] - data_at_j[3];

        // // apply on index 4.
        // data_at_i[4] = temp[4] + data_at_j[4];
        // data_at_j[4] = temp[4] - data_at_j[4];

        // // apply on index 5.
        // data_at_i[5] = temp[5] + data_at_j[5];
        // data_at_j[5] = temp[5] - data_at_j[5];

        // // apply on index 6.
        // data_at_i[6] = temp[6] + data_at_j[6];
        // data_at_j[6] = temp[6] - data_at_j[6];

        // // apply on index 7.
        // data_at_i[7] = temp[7] + data_at_j[7];
        // data_at_j[7] = temp[7] - data_at_j[7];

        // self.data[i] = data_at_i;
        // self.data[j] = data_at_j;

        // // apply on index 0.
        // self.data[i][0] = temp[0] + self.data[j][0];
        // self.data[j][0] = temp[0] - self.data[j][0];

        // // apply on index 1.
        // self.data[i][1] = temp[1] + self.data[j][1];
        // self.data[j][1] = temp[1] - self.data[j][1];

        // // apply on index 2.
        // self.data[i][2] = temp[2] + self.data[j][2];
        // self.data[j][2] = temp[2] - self.data[j][2];

        // // apply on index 3.
        // self.data[i][3] = temp[3] + self.data[j][3];
        // self.data[j][3] = temp[3] - self.data[j][3];

        // // apply on index 4.
        // self.data[i][4] = temp[4] + self.data[j][4];
        // self.data[j][4] = temp[4] - self.data[j][4];

        // // apply on index 5.
        // self.data[i][5] = temp[5] + self.data[j][5];
        // self.data[j][5] = temp[5] - self.data[j][5];

        // // apply on index 6.
        // self.data[i][6] = temp[6] + self.data[j][6];
        // self.data[j][6] = temp[6] - self.data[j][6];

        // // apply on index 7.
        // self.data[i][7] = temp[7] + self.data[j][7];
        // self.data[j][7] = temp[7] - self.data[j][7];
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

        // self.data[i] = data_at_i;
        // self.data[j] = data_at_j;

        // // apply on index 0.
        // self.data[j][0] = self.data[j][0].mul_base(twiddle);
        // self.data[i][0] = temp[0] + self.data[j][0];
        // self.data[j][0] = temp[0] - self.data[j][0];

        // // apply on index 1.
        // self.data[j][1] = self.data[j][1].mul_base(twiddle);
        // self.data[i][1] = temp[1] + self.data[j][1];
        // self.data[j][1] = temp[1] - self.data[j][1];

        // // apply on index 2.
        // self.data[j][2] = self.data[j][2].mul_base(twiddle);
        // self.data[i][2] = temp[2] + self.data[j][2];
        // self.data[j][2] = temp[2] - self.data[j][2];

        // // apply on index 3.
        // self.data[j][3] = self.data[j][3].mul_base(twiddle);
        // self.data[i][3] = temp[3] + self.data[j][3];
        // self.data[j][3] = temp[3] - self.data[j][3];

        // // apply on index 4.
        // self.data[j][4] = self.data[j][4].mul_base(twiddle);
        // self.data[i][4] = temp[4] + self.data[j][4];
        // self.data[j][4] = temp[4] - self.data[j][4];

        // // apply on index 5.
        // self.data[j][5] = self.data[j][5].mul_base(twiddle);
        // self.data[i][5] = temp[5] + self.data[j][5];
        // self.data[j][5] = temp[5] - self.data[j][5];

        // // apply on index 6.
        // self.data[j][6] = self.data[j][6].mul_base(twiddle);
        // self.data[i][6] = temp[6] + self.data[j][6];
        // self.data[j][6] = temp[6] - self.data[j][6];

        // // apply on index 7.
        // self.data[j][7] = self.data[j][7].mul_base(twiddle);
        // self.data[i][7] = temp[7] + self.data[j][7];
        // self.data[j][7] = temp[7] - self.data[j][7];
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
