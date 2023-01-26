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
    ///
    /// # Panics
    /// Panics if:
    /// * The provided rows of data is empty.
    /// * The remainder of the length of the data and the row width is not zero.
    /// * Number of rows is smaller than or equal to 1.
    /// * Number of rows is not a power of two.
    pub fn new(data: Vec<E>, row_width: usize) -> Self {
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

    /// Returns the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.row_width
    }

    /// Returns the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.data.len() / self.row_width
    }

    /// Returns the data in this matrix as a mutable slice.
    pub fn as_data_mut(&mut self) -> &mut [E] {
        &mut self.data
    }

    /// Returns the data in this matrix as a slice.
    pub fn as_data(&self) -> &[E] {
        &self.data
    }

    /// Returns the element located at the specified column and row indexes in this matrix.
    ///
    /// # Panics
    /// Panics if either `col_idx` or `row_idx` are out of bounds for this matrix.
    pub fn get(&self, col_idx: usize, row_idx: usize) -> E {
        self.data[row_idx * self.row_width + col_idx]
    }

    /// Set the cell in this matrix at the specified column and row indexes to the provided value.
    ///
    /// # Panics
    /// Panics if either `col_idx` or `row_idx` are out of bounds for this matrix.
    pub fn set(&mut self, col_idx: usize, row_idx: usize, value: E) {
        self.data[row_idx * self.row_width + col_idx] = value;
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
        row.copy_from_slice(&self.data[row_idx * self.row_width..(row_idx + 1) * self.row_width]);
    }

    /// Updates a row in this matrix at the specified index to the provided data.
    ///
    /// # Panics
    /// Panics if `row_idx` is out of bounds for this matrix.
    pub fn update_row(&mut self, row_idx: usize, row: &[E]) {
        self.data[row_idx * self.row_width..(row_idx + 1) * self.row_width].copy_from_slice(row);
    }

    /// Returns the columns of this matrix as a list of vectors.
    pub fn into_columns(&self) -> Vec<Vec<E>> {
        (0..self.num_cols()).map(|n| self.into_column(n)).collect()
    }

    /// Returns the column at the specified index.
    pub fn into_column(&self, index: usize) -> Vec<E> {
        self.data
            .iter()
            .copied()
            .skip(index)
            .step_by(self.num_cols())
            .collect()
    }

    // PUBLIC ACCESSORS
    // ================================================================================================

    /// Returns the number of elements in the underlying slice.
    pub fn len(&self) -> usize {
        self.num_rows()
    }

    /// Returns if the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // /// Safe mutable slice cast to avoid unnecessary lifetime complexity.
    // fn as_mut_slice(&mut self) -> &'a mut [E] {
    //     let ptr = self.data as *mut [E];
    //     // Safety: we still hold the mutable reference to the slice so no ownership rule is
    //     // violated.
    //     unsafe { ptr.as_mut().expect("the initial reference was not valid.") }
    // }

    // /// Splits the struct into two mutable struct at the given split point. Data of first
    // /// chunk will contain elements at indices [0, split_point), and the second chunk
    // /// will contain elements at indices [split_point, size).
    // fn split_at_mut(&mut self, split_point: usize) -> (Self, Self) {
    //     let at = split_point * self.row_width;
    //     let (left, right) = self.as_mut_slice().split_at_mut(at);
    //     let left = Self::new(left, self.row_width);
    //     let right = Self::new(right, self.row_width);
    //     (left, right)
    // }

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

        #[allow(unused_assignments)]
        let mut result = RowMatrix::new(Vec::new(), self.num_cols());
        // when `concurrent` feature is enabled, run the concurrent version of the function; unless
        // the polynomial is small, then don't bother with the concurrent version
        //
        // Cant return RowMatrix because result was created in this function. So we are returning a
        // Vec<E> instead. The caller of this function should convert it to a RowMatrix.
        if cfg!(feature = "concurrent") && self.len() >= MIN_CONCURRENT_SIZE {
            {
                // #[cfg(feature = "concurrent")]
                result = self.evaluate_poly_with_offset_concurrent(
                    twiddles,
                    domain_offset,
                    blowup_factor,
                );
            }
        } else {
            result = self.evaluate_poly_with_offset(twiddles, domain_offset, blowup_factor);
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
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                p.data.as_mut_slice(),
                p.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
            row_matrix_segment_i.fft_in_place(twiddles);
            row_matrix_segment_i.permute();
        }
    }

    /// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
    /// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
    pub fn evaluate_poly_with_offset(
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

        // for m in 0..self.num_cols() / ARR_SIZE {
        result
            .as_mut_slice()
            .chunks_mut(self.len() * self.row_width)
            .enumerate()
            .for_each(|(i, chunk)| {
                let idx = fft::permute_index(blowup_factor, i) as u64;
                let offset = g.exp(idx.into()) * domain_offset;

                let chunk_len = chunk.len() / self.row_width;

                for m in 0..self.num_cols() / ARR_SIZE {
                    let mut factor = E::BaseField::ONE;
                    for d in 0..chunk_len {
                        let row_idx = d * self.row_width + m * ARR_SIZE;
                        chunk[row_idx] = self.data[row_idx].mul_base(factor);
                        chunk[row_idx + 1] = self.data[row_idx + 1].mul_base(factor);
                        chunk[row_idx + 2] = self.data[row_idx + 2].mul_base(factor);
                        chunk[row_idx + 3] = self.data[row_idx + 3].mul_base(factor);
                        chunk[row_idx + 4] = self.data[row_idx + 4].mul_base(factor);
                        chunk[row_idx + 5] = self.data[row_idx + 5].mul_base(factor);
                        chunk[row_idx + 6] = self.data[row_idx + 6].mul_base(factor);
                        chunk[row_idx + 7] = self.data[row_idx + 7].mul_base(factor);

                        factor *= offset;
                    }
                    let mut row_matrix_segment_i = RowMatrixSegment::new(
                        chunk,
                        self.row_width,
                        m * ARR_SIZE,
                        (m + 1) * ARR_SIZE - 1,
                    );
                    FftInputs::fft_in_place(&mut row_matrix_segment_i, twiddles);
                }
            });

        let mut matrix_result = RowMatrix {
            data: result,
            row_width: self.row_width,
        };

        FftInputs::permute(&mut matrix_result);
        matrix_result
    }

    // POLYNOMIAL INTERPOLATION
    // ================================================================================================

    /// Interpolates `evaluations` over a domain of length `evaluations.len()` in the field specified
    /// `B` into a polynomial in coefficient form using the FFT algorithm.
    pub fn interpolate_poly(evaluations: &mut RowMatrix<E>, inv_twiddles: &[E::BaseField]) {
        let inv_length = E::BaseField::inv((evaluations.len() as u64).into());

        for i in 0..evaluations.row_width / ARR_SIZE {
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                evaluations.data.as_mut_slice(),
                evaluations.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
            row_matrix_segment_i.fft_in_place(inv_twiddles);
            row_matrix_segment_i.shift_by(inv_length);
            row_matrix_segment_i.permute();
        }
    }

    /// Interpolates `evaluations` over a domain of length `evaluations.len()` and shifted by
    /// `domain_offset` in the field specified by `B` into a polynomial in coefficient form using
    /// the FFT algorithm.
    pub fn interpolate_poly_with_offset(
        evaluations: &mut RowMatrix<E>,
        inv_twiddles: &[E::BaseField],
        domain_offset: E::BaseField,
    ) {
        let offset = E::BaseField::inv((evaluations.len() as u64).into());
        let domain_offset = E::BaseField::inv(domain_offset);

        for i in 0..evaluations.row_width / ARR_SIZE {
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                evaluations.data.as_mut_slice(),
                evaluations.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
            row_matrix_segment_i.fft_in_place(inv_twiddles);
            row_matrix_segment_i.permute();
            row_matrix_segment_i.shift_by_series(offset, domain_offset, 0)
        }
    }

    // CONCURRENT EVALUATION
    // ================================================================================================

    // #[cfg(feature = "concurrent")]
    /// Evaluates polynomial `p` over the domain of length `p.len()` in the field specified `B` using
    /// the FFT algorithm and returns the result.
    ///
    /// This function is only available when the `concurrent` feature is enabled.
    pub fn evaluate_poly_concurrent(p: &mut RowMatrix<E>, twiddles: &[E::BaseField]) {
        // TODO: implement concurrent evaluation using rayon across rows.
        for i in 0..p.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                p.data.as_mut_slice(),
                p.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
            row_matrix_segment_i.split_radix_fft(twiddles);
            row_matrix_segment_i.permute_concurrent();
        }
    }

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
                    .for_each(|(ix, (src, dest))| {
                        let src_len = src.len() / self.row_width;

                        for m in 0..self.row_width / ARR_SIZE {
                            let mut factor = offset.exp(((ix * batch_size) as u64).into());
                            for d in 0..src_len {
                                let row_idx = d * self.row_width + m * ARR_SIZE;
                                dest[row_idx] = src[row_idx].mul_base(factor);
                                dest[row_idx + 1] = src[row_idx + 1].mul_base(factor);
                                dest[row_idx + 2] = src[row_idx + 2].mul_base(factor);
                                dest[row_idx + 3] = src[row_idx + 3].mul_base(factor);
                                dest[row_idx + 4] = src[row_idx + 4].mul_base(factor);
                                dest[row_idx + 5] = src[row_idx + 5].mul_base(factor);
                                dest[row_idx + 6] = src[row_idx + 6].mul_base(factor);
                                dest[row_idx + 7] = src[row_idx + 7].mul_base(factor);

                                factor *= offset;
                            }
                        }
                    });
                for i in 0..self.row_width / ARR_SIZE {
                    let mut row_matrix_segment_i = RowMatrixSegment::new(
                        chunk,
                        self.row_width,
                        i * ARR_SIZE,
                        (i + 1) * ARR_SIZE - 1,
                    );
                    FftInputs::split_radix_fft(&mut row_matrix_segment_i, twiddles)
                }
            });

        let mut matrix_result = RowMatrix {
            data: result,
            row_width: self.row_width,
        };

        FftInputs::permute_concurrent(&mut matrix_result);
        matrix_result
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

        let batch_size = evaluations.len() / rayon::current_num_threads().next_power_of_two();
        for i in 0..evaluations.num_cols() / ARR_SIZE {
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                evaluations.data.as_mut_slice(),
                evaluations.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
            row_matrix_segment_i.split_radix_fft(inv_twiddles);
            rayon::iter::IndexedParallelIterator::enumerate(
                row_matrix_segment_i.par_mut_chunks(batch_size),
            )
            .for_each(|(_i, mut batch)| {
                // let n = batch.len();
                // let num_batches = rayon::current_num_threads().next_power_of_two();
                // let batch_size = n / num_batches;
                // rayon::scope(|s| {
                //     for batch_idx in 0..num_batches {
                //         // create another mutable reference to the slice of values to use in a new thread; this
                //         // is OK because we never write the same positions in the slice from different threads
                //         let values = unsafe { &mut *(&mut batch as *mut RowMatrix<E>) };
                //         s.spawn(move |_| {
                //             let batch_start = batch_idx * batch_size;
                //             let batch_end = batch_start + batch_size;
                //             for i in batch_start..batch_end {
                //                 values.shift_by(inv_length);
                //             }
                //         });
                //     }
                // });
                batch.shift_by(inv_length);
            });
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
            let mut row_matrix_segment_i = RowMatrixSegment::new(
                evaluations.data.as_mut_slice(),
                evaluations.row_width,
                i * ARR_SIZE,
                (i + 1) * ARR_SIZE - 1,
            );
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
}

/// Implementation of `FftInputs` for `RowMatrix`.
impl<E> FftInputs<E> for RowMatrix<E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixSegment<'b, E> where Self: 'b;
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
    fn par_mut_chunks(&mut self, _chunk_size: usize) -> MatrixChunksMut<'_, E> {
        unimplemented!("parallel chunks not implemented for RowMatrix")
    }
}

pub struct RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    data: &'a mut [E],
    row_width: usize,
    init_col: usize,
    end_col: usize,
}

impl<'a, E> RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    pub fn new(data: &'a mut [E], row_width: usize, init_col: usize, end_col: usize) -> Self {
        Self {
            data,
            row_width,
            init_col,
            end_col,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.row_width
    }

    pub fn get_init_col(&self) -> usize {
        self.init_col
    }

    pub fn get_end_col(&self) -> usize {
        self.end_col
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
        let left = Self::new(left, self.row_width, self.init_col, self.end_col);
        let right = Self::new(right, self.row_width, self.init_col, self.end_col);
        (left, right)
    }
}

/// Implementation of `FftInputs` for `RowMatrix`. This is used to perform FFT on the
/// rows of the matrix.
impl<'a, E> FftInputs<E> for RowMatrixSegment<'a, E>
where
    E: FieldElement,
{
    type ChunkItem<'b> = RowMatrixSegment<'b, E> where Self: 'b;
    type ParChunksMut<'c> = MatrixChunksMut<'c, E> where Self: 'c;

    fn len(&self) -> usize {
        self.data.len() / self.row_width
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = offset * self.row_width + self.init_col;
        let j = (offset + stride) * self.row_width + self.init_col;

        //  apply on 1st element of the array.
        let temp = self.data[i];
        self.data[i] = temp + self.data[j];
        self.data[j] = temp - self.data[j];

        //  apply on 2nd element of the array.
        let temp = self.data[i + 1];
        self.data[i + 1] = temp + self.data[j + 1];
        self.data[j + 1] = temp - self.data[j + 1];

        //  apply on 3rd element of the array.
        let temp = self.data[i + 2];
        self.data[i + 2] = temp + self.data[j + 2];
        self.data[j + 2] = temp - self.data[j + 2];

        //  apply on 4th element of the array.
        let temp = self.data[i + 3];
        self.data[i + 3] = temp + self.data[j + 3];
        self.data[j + 3] = temp - self.data[j + 3];

        //  apply on 5th element of the array.
        let temp = self.data[i + 4];
        self.data[i + 4] = temp + self.data[j + 4];
        self.data[j + 4] = temp - self.data[j + 4];

        //  apply on 6th element of the array.
        let temp = self.data[i + 5];
        self.data[i + 5] = temp + self.data[j + 5];
        self.data[j + 5] = temp - self.data[j + 5];

        //  apply on 7th element of the array.
        let temp = self.data[i + 6];
        self.data[i + 6] = temp + self.data[j + 6];
        self.data[j + 6] = temp - self.data[j + 6];

        //  apply on 8th element of the array.
        let temp = self.data[i + 7];
        self.data[i + 7] = temp + self.data[j + 7];
        self.data[j + 7] = temp - self.data[j + 7];
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: E::BaseField, offset: usize, stride: usize) {
        let i = offset * self.row_width + self.init_col;
        let j = (offset + stride) * self.row_width + self.init_col;

        //  apply on 1st element of the array.
        let temp = self.data[i];
        self.data[j] = self.data[j].mul_base(twiddle);
        self.data[i] = temp + self.data[j];
        self.data[j] = temp - self.data[j];

        //  apply on 2nd element of the array.
        let temp = self.data[i + 1];
        self.data[j + 1] = self.data[j + 1].mul_base(twiddle);
        self.data[i + 1] = temp + self.data[j + 1];
        self.data[j + 1] = temp - self.data[j + 1];

        //  apply on 3rd element of the array.
        let temp = self.data[i + 2];
        self.data[j + 2] = self.data[j + 2].mul_base(twiddle);
        self.data[i + 2] = temp + self.data[j + 2];
        self.data[j + 2] = temp - self.data[j + 2];

        //  apply on 4th element of the array.
        let temp = self.data[i + 3];
        self.data[j + 3] = self.data[j + 3].mul_base(twiddle);
        self.data[i + 3] = temp + self.data[j + 3];
        self.data[j + 3] = temp - self.data[j + 3];

        //  apply on 5th element of the array.
        let temp = self.data[i + 4];
        self.data[j + 4] = self.data[j + 4].mul_base(twiddle);
        self.data[i + 4] = temp + self.data[j + 4];
        self.data[j + 4] = temp - self.data[j + 4];

        //  apply on 6th element of the array.
        let temp = self.data[i + 5];
        self.data[j + 5] = self.data[j + 5].mul_base(twiddle);
        self.data[i + 5] = temp + self.data[j + 5];
        self.data[j + 5] = temp - self.data[j + 5];

        //  apply on 7th element of the array.
        let temp = self.data[i + 6];
        self.data[j + 6] = self.data[j + 6].mul_base(twiddle);
        self.data[i + 6] = temp + self.data[j + 6];
        self.data[j + 6] = temp - self.data[j + 6];

        //  apply on 8th element of the array.
        let temp = self.data[i + 7];
        self.data[j + 7] = self.data[j + 7].mul_base(twiddle);
        self.data[i + 7] = temp + self.data[j + 7];
        self.data[j + 7] = temp - self.data[j + 7];
    }

    fn swap(&mut self, i: usize, j: usize) {
        let i = i * self.row_width + self.init_col;
        let j = j * self.row_width + self.init_col;

        let (first_row, second_row) = self.data.split_at_mut(j);
        let (first_row, second_row) = (
            &mut first_row[i..i + ARR_SIZE],
            &mut second_row[0..ARR_SIZE],
        );

        // Swap the two rows.
        first_row.swap_with_slice(second_row);
    }

    fn shift_by_series(&mut self, offset: E::BaseField, increment: E::BaseField, num_skip: usize) {
        let increment = E::from(increment);
        let mut offset = E::from(offset);
        for idx in num_skip..self.len() {
            let row_idx = idx * self.row_width + self.init_col;
            // apply on index 0.
            self.data[row_idx] *= offset;

            // apply on index 1.
            self.data[row_idx + 1] *= offset;

            // apply on index 2.
            self.data[row_idx + 2] *= offset;

            // apply on index 3.
            self.data[row_idx + 3] *= offset;

            // apply on index 4.
            self.data[row_idx + 4] *= offset;

            // apply on index 5.
            self.data[row_idx + 5] *= offset;

            // apply on index 6.
            self.data[row_idx + 6] *= offset;

            // apply on index 7.
            self.data[row_idx + 7] *= offset;

            // increment the offset.
            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: E::BaseField) {
        let offset = E::from(offset);

        for idx in 0..self.len() {
            let row_idx = idx * self.row_width + self.init_col;
            // apply on index 0.
            self.data[row_idx] *= offset;

            // apply on index 1.
            self.data[row_idx + 1] *= offset;

            // apply on index 2.
            self.data[row_idx + 2] *= offset;

            // apply on index 3.
            self.data[row_idx + 3] *= offset;

            // apply on index 4.
            self.data[row_idx + 4] *= offset;

            // apply on index 5.shift_by
            self.data[row_idx + 5] *= offset;

            // apply on index 6.
            self.data[row_idx + 6] *= offset;

            // apply on index 7.
            self.data[row_idx + 7] *= offset;
        }
    }

    // #[cfg(feature = "concurrent")]
    fn par_mut_chunks(&mut self, chunk_size: usize) -> MatrixChunksMut<'_, E> {
        MatrixChunksMut {
            data: RowMatrixSegment {
                data: self.as_mut_slice(),
                row_width: self.row_width,
                init_col: self.init_col,
                end_col: self.end_col,
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
