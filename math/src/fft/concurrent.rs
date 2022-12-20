// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    field::{FieldElement, StarkField},
    utils::log2,
};
use rayon::prelude::*;

use super::fft_inputs::FftInputs;

// POLYNOMIAL EVALUATION
// ================================================================================================

/// Evaluates polynomial `p` using FFT algorithm; the evaluation is done in-place, meaning
/// `p` is updated with results of the evaluation.
pub fn evaluate_poly<B, E, I>(p: &mut I, twiddles: &[B])
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + Send,
{
    split_radix_fft(p, twiddles);
    permute(p);
}

/// Evaluates polynomial `p` using FFT algorithm and returns the result. The polynomial is
/// evaluated over domain specified by `twiddles`, expanded by the `blowup_factor`, and shifted
/// by the `domain_offset`.
pub fn evaluate_poly_with_offset<B, E, I>(
    p: &I,
    twiddles: &[B],
    domain_offset: B,
    blowup_factor: usize,
    result: &mut I,
) where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + Send + Sync,
{
    let g = B::get_root_of_unity(log2(p.size() * blowup_factor));

    result
        .par_mut_chunks(p.size())
        .enumerate()
        .for_each(|(i, mut chunk)| {
            let idx = super::permute_index(blowup_factor, i) as u64;
            let offset = g.exp(idx.into()) * domain_offset;
            // Note: We would need to implement a parallel verion of clone_and_shift_by in
            // FftInputs.
            chunk.par_clone_and_shift_by(p, offset);
            split_radix_fft(&mut chunk, twiddles);
        });

    permute(result);
}

// POLYNOMIAL INTERPOLATION
// ================================================================================================

/// Uses FFT algorithm to interpolate a polynomial from provided `values`; the interpolation
/// is done in-place, meaning `values` are updated with polynomial coefficients.
pub fn interpolate_poly<B, E, I>(v: &mut I, inv_twiddles: &[B])
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + Send,
{
    split_radix_fft(v, inv_twiddles);
    let inv_length = B::inv((v.size() as u64).into());

    let batch_size = v.size() / rayon::current_num_threads().next_power_of_two();

    // Note: One alternate solution would be to implement a parallel version of this loop in
    // FftInputs.
    v.par_mut_chunks(batch_size)
        .enumerate()
        .for_each(|(_i, mut batch)| {
            batch.shift_by(inv_length);
        });
    permute(v);
}

/// Uses FFT algorithm to interpolate a polynomial from provided `values` over the domain defined
/// by `inv_twiddles` and offset by `domain_offset` factor.
pub fn interpolate_poly_with_offset<B, E, I>(values: &mut I, inv_twiddles: &[B], domain_offset: B)
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + Send,
{
    split_radix_fft(values, inv_twiddles);
    permute(values);

    let domain_offset = B::inv(domain_offset);
    let inv_len = B::inv((values.size() as u64).into());
    let batch_size = values.size() / rayon::current_num_threads().next_power_of_two();

    values
        .par_mut_chunks(batch_size)
        .enumerate()
        .for_each(|(i, mut batch)| {
            let offset = domain_offset.exp(((i * batch_size) as u64).into()) * inv_len;
            batch.shift_by_series(offset, domain_offset, 0);
        });
}

// PERMUTATIONS
// ================================================================================================

pub fn permute<E, I>(v: &mut I)
where
    E: FieldElement,
    I: FftInputs<E> + Send,
{
    let n = v.size();
    let num_batches = rayon::current_num_threads().next_power_of_two();
    let batch_size = n / num_batches;
    rayon::scope(|s| {
        for batch_idx in 0..num_batches {
            // create another mutable reference to the slice of values to use in a new thread; this
            // is OK because we never write the same positions in the slice from different threads
            let values = unsafe { &mut *(&mut *v as *mut I) };
            s.spawn(move |_| {
                let batch_start = batch_idx * batch_size;
                let batch_end = batch_start + batch_size;
                for i in batch_start..batch_end {
                    let j = super::permute_index(n, i);
                    if j > i {
                        values.swap_elements(i, j);
                    }
                }
            });
        }
    });
}

// SPLIT-RADIX FFT
// ================================================================================================

/// In-place recursive FFT with permuted output.
/// Adapted from: https://github.com/0xProject/OpenZKP/tree/master/algebra/primefield/src/fft
pub(super) fn split_radix_fft<B, E, I>(values: &mut I, twiddles: &[B])
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E>,
{
    // generator of the domain should be in the middle of twiddles
    let n = values.size();
    let g = twiddles[twiddles.len() / 2];
    debug_assert_eq!(g.exp((n as u32).into()), E::BaseField::ONE);

    let inner_len = 1_usize << (log2(n) / 2);
    let outer_len = n / inner_len;
    let stretch = outer_len / inner_len;
    debug_assert!(outer_len == inner_len || outer_len == 2 * inner_len);
    debug_assert_eq!(outer_len * inner_len, n);

    // transpose inner x inner x stretch square matrix
    transpose_square_stretch(values, inner_len, stretch);

    // apply inner FFTs
    values
        .par_mut_chunks(outer_len)
        .for_each(|mut row| super::serial::fft_in_place(&mut row, twiddles, stretch, stretch, 0));

    // transpose inner x inner x stretch square matrix
    transpose_square_stretch(values, inner_len, stretch);

    // apply outer FFTs
    values
        .par_mut_chunks(outer_len)
        .enumerate()
        .for_each(|(i, mut row)| {
            if i > 0 {
                let i = super::permute_index(inner_len, i);
                let inner_twiddle = g.exp((i as u32).into());
                let outer_twiddle = inner_twiddle;
                row.shift_by_series(outer_twiddle, inner_twiddle, 1);
            }
            super::serial::fft_in_place(&mut row, twiddles, 1, 1, 0)
        });
}

// TRANSPOSING
// ================================================================================================

fn transpose_square_stretch<E, I>(matrix: &mut I, size: usize, stretch: usize)
where
    E: FieldElement,
    I: FftInputs<E>,
{
    assert_eq!(matrix.size(), size * size * stretch);
    match stretch {
        1 => transpose_square_1(matrix, size),
        2 => transpose_square_2(matrix, size),
        _ => unimplemented!("only stretch sizes 1 and 2 are supported"),
    }
}

fn transpose_square_1<E, I>(matrix: &mut I, size: usize)
where
    E: FieldElement,
    I: FftInputs<E>,
{
    debug_assert_eq!(matrix.size(), size * size);
    if size % 2 != 0 {
        unimplemented!("odd sizes are not supported");
    }

    // iterate over upper-left triangle, working in 2x2 blocks
    for row in (0..size).step_by(2) {
        let i = row * size + row;
        matrix.swap_elements(i + 1, i + size);
        for col in (row..size).step_by(2).skip(1) {
            let i = row * size + col;
            let j = col * size + row;
            matrix.swap_elements(i, j);
            matrix.swap_elements(i + 1, j + size);
            matrix.swap_elements(i + size, j + 1);
            matrix.swap_elements(i + size + 1, j + size + 1);
        }
    }
}

fn transpose_square_2<E, I>(matrix: &mut I, size: usize)
where
    E: FieldElement,
    I: FftInputs<E>,
{
    debug_assert_eq!(matrix.size(), 2 * size * size);

    // iterate over upper-left triangle, working in 1x2 blocks
    for row in 0..size {
        for col in (row..size).skip(1) {
            let i = (row * size + col) * 2;
            let j = (col * size + row) * 2;
            matrix.swap_elements(i, j);
            matrix.swap_elements(i + 1, j + 1);
        }
    }
}
