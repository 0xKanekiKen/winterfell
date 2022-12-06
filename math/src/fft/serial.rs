// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::fft_inputs::FftInputs;
use crate::{field::StarkField, utils::log2, FieldElement};
use utils::{collections::Vec, uninit_vector};

// CONSTANTS
// ================================================================================================
const MAX_LOOP: usize = 256;

// POLYNOMIAL EVALUATION
// ================================================================================================

/// Evaluates polynomial `p` in-place over the domain of length `p.len()` in the field specified
/// by `B` using the FFT algorithm.
pub fn evaluate_poly<B, I>(p: &mut I, twiddles: &[B])
where
    B: StarkField,
    I: FftInputs<B> + ?Sized,
{
    fft_in_place(p, twiddles, 1, 1, 0);
    permute(p);
}

/// Evaluates polynomial `p` over the domain of length `p.len()` * `blowup_factor` shifted by
/// `domain_offset` in the field specified `B` using the FFT algorithm and returns the result.
pub fn evaluate_poly_with_offset<B, E>(
    p: &[E],
    twiddles: &[B],
    domain_offset: B,
    blowup_factor: usize,
) -> Vec<E>
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
{
    let domain_size = p.len() * blowup_factor;
    let g = B::get_root_of_unity(log2(domain_size));
    let mut result = unsafe { uninit_vector(domain_size) };

    result
        .as_mut_slice()
        .chunks_mut(p.len())
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = super::permute_index(blowup_factor, i) as u64;
            let offset = g.exp(idx.into()) * domain_offset;
            let mut factor = E::BaseField::ONE;
            for (d, c) in chunk.iter_mut().zip(p.iter()) {
                *d = (*c).mul_base(factor);
                factor *= offset;
            }
            fft_in_place(chunk, twiddles, 1, 1, 0);
        });

    permute(result.as_mut_slice());
    result
}

// POLYNOMIAL INTERPOLATION
// ================================================================================================

/// Interpolates `evaluations` over a domain of length `evaluations.len()` in the field specified
/// `B` into a polynomial in coefficient form using the FFT algorithm.
pub fn interpolate_poly<B, I>(evaluations: &mut I, inv_twiddles: &[B])
where
    B: StarkField,
    I: FftInputs<B> + ?Sized,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    let mut offset = B::inv((evaluations.len() as u64).into());
    I::interpolate(evaluations, B::ONE, &mut offset);
    permute(evaluations);
}

/// Interpolates `evaluations` over a domain of length `evaluations.len()` and shifted by
/// `domain_offset` in the field specified by `B` into a polynomial in coefficient form using
/// the FFT algorithm.
pub fn interpolate_poly_with_offset<B, I>(evaluations: &mut I, inv_twiddles: &[B], domain_offset: B)
where
    B: StarkField,
    I: FftInputs<B> + ?Sized,
{
    fft_in_place(evaluations, inv_twiddles, 1, 1, 0);
    permute(evaluations);
    let mut offset = B::inv((evaluations.len() as u64).into());
    I::interpolate(evaluations, domain_offset, &mut offset)
}

// PERMUTATIONS
// ================================================================================================

pub fn permute<B, I>(values: &mut I)
where
    B: StarkField,
    I: FftInputs<B> + ?Sized,
{
    let n = values.len();
    for i in 0..n {
        let j = super::permute_index(n, i);
        if j > i {
            values.swap(i, j);
        }
    }
}

// CORE FFT ALGORITHM
// ================================================================================================

/// In-place recursive FFT with permuted output.
///
/// Adapted from: https://github.com/0xProject/OpenZKP/tree/master/algebra/primefield/src/fft
pub(super) fn fft_in_place<B, I>(
    values: &mut I,
    twiddles: &[B],
    count: usize,
    stride: usize,
    offset: usize,
) where
    B: StarkField,
    I: FftInputs<B> + ?Sized,
{
    let size = values.len() / stride;
    debug_assert!(size.is_power_of_two());
    debug_assert!(offset < stride);
    debug_assert_eq!(values.len() % size, 0);

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
}
