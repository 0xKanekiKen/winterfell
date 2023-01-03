// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::fft_inputs::FftInputs;
use crate::{field::StarkField, utils::log2, FieldElement};

// CONSTANTS
// ================================================================================================
const MAX_LOOP: usize = 256;

// POLYNOMIAL EVALUATION
// ================================================================================================

/// Evaluates polynomial `p` in-place over the domain of length `p.len()` in the field specified
/// by `B` using the FFT algorithm.
pub fn evaluate_poly<B, E, I>(p: &mut I, twiddles: &[B])
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    fft_in_place(p, twiddles, 1, 1, 0);
    permute(p);
}

/// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
/// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
pub fn evaluate_poly_with_offset<B, E, I>(
    p: &I,
    twiddles: &[B],
    domain_offset: B,
    blowup_factor: usize,
    result: &mut I,
) where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    let g = B::get_root_of_unity(log2(p.size() * blowup_factor));

    result
        .mut_chunks(p.size())
        .enumerate()
        .for_each(|(i, mut chunk)| {
            // convert chunk into Fftinputs. This is safe because we know that the chunk is
            // the same size as the input.
            let idx = super::permute_index(blowup_factor, i) as u64;
            let offset = g.exp(idx.into()) * domain_offset;
            let factor = B::ONE;
            chunk.clone_and_shift_by(p, factor, offset);
            fft_in_place(&mut chunk, twiddles, 1, 1, 0);
        });

    permute(result);
}

// POLYNOMIAL INTERPOLATION
// ================================================================================================

/// Interpolates `evaluations` over a domain of length `evaluations.len()` in the field specified
/// `B` into a polynomial in coefficient form using the FFT algorithm.
pub fn interpolate_poly<B, E, I>(evaluations: &mut I, inv_twiddles: &[B])
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    let offset = B::inv((evaluations.size() as u64).into());
    evaluations.shift_by(offset);
    permute(evaluations);
}

/// Interpolates `evaluations` over a domain of length `evaluations.len()` and shifted by
/// `domain_offset` in the field specified by `B` into a polynomial in coefficient form using
/// the FFT algorithm.
pub fn interpolate_poly_with_offset<B, E, I>(
    evaluations: &mut I,
    inv_twiddles: &[B],
    domain_offset: B,
) where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    permute(evaluations);
    let domain_offset = B::inv(domain_offset);
    let offset = B::inv((evaluations.size() as u64).into());
    evaluations.shift_by_series(offset, domain_offset, 0);
}

// PERMUTATIONS
// ================================================================================================

pub fn permute<B, E, I>(values: &mut I)
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    let n = values.size();
    for i in 0..n {
        let j = super::permute_index(n, i);
        if j > i {
            values.swap_elements(i, j);
        }
    }
}

// CORE FFT ALGORITHM
// ================================================================================================

/// In-place recursive FFT with permuted output.
///
/// Adapted from: https://github.com/0xProject/OpenZKP/tree/master/algebra/primefield/src/fft
pub(super) fn fft_in_place<B, E, I>(
    values: &mut I,
    twiddles: &[B],
    count: usize,
    stride: usize,
    offset: usize,
) where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    I: FftInputs<E> + ?Sized,
{
    let size = values.size() / stride;
    debug_assert!(size.is_power_of_two());
    debug_assert!(offset < stride);
    debug_assert_eq!(values.size() % size, 0);

    // Keep recursing until size is 2
    if size > 2 {
        if stride == count && count < MAX_LOOP {
            fft_in_place(values, twiddles, 2 * count, 2 * stride, offset);
        } else {
            fft_in_place(values, twiddles, count, 2 * stride, offset);
            fft_in_place(values, twiddles, count, 2 * stride, offset + stride);
        }
    }

    for offset in offset..(offset + count) {
        I::butterfly(values, offset, stride);
    }

    let last_offset = offset + size * stride;
    for (i, offset) in (offset..last_offset)
        .step_by(2 * stride)
        .enumerate()
        .skip(1)
    {
        for j in offset..(offset + count) {
            I::butterfly_twiddle(values, twiddles[i], j, stride);
        }
    }

    values.size();
}
