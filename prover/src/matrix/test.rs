// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    math::{
        fft::{fft_inputs::FftInputs, get_inv_twiddles, get_twiddles},
        fields::f64::BaseElement,
        get_power_series, log2, polynom, StarkField,
    },
    Matrix,
};

use super::RowMatrix;
use math::FieldElement;
use rand_utils::rand_vector;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use utils::collections::Vec;

#[test]
fn test_eval_poly_with_offset_matrix() {
    let n = 1024 * 512;
    let num_polys = 64;
    let blowup_factor = 8;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let matrix = RowMatrix::from_polys(&Matrix::new(columns.clone()), blowup_factor);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n * blowup_factor);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();

    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    assert_eq!(eval_cols_faltten, matrix.get_data());
}

// HELPER FUNCTIONS
// ================================================================================================

/// Builds a domain of size `size` using the primitive element of the field.
fn build_domain(size: usize) -> Vec<BaseElement> {
    let g = BaseElement::get_root_of_unity(log2(size));
    get_power_series(g, size)
}

/// Transposes a matrix stored in a column major format to a row major format.
fn transpose<E: FieldElement>(matrix: Vec<Vec<E>>) -> Vec<Vec<E>> {
    let num_rows = matrix.len();
    let num_cols = matrix[0].len();
    let mut result = vec![vec![E::ZERO; num_rows]; num_cols];
    result.iter_mut().enumerate().for_each(|(i, row)| {
        row.iter_mut().enumerate().for_each(|(j, col)| {
            *col = matrix[j][i];
        })
    });
    result
}

/// Transposes a matrix stored in a column major format to a row major format concurrently.
fn transpose_concurrent<E: FieldElement>(matrix: Vec<Vec<E>>) -> Vec<Vec<E>> {
    let num_rows = matrix.len();
    let num_cols = matrix[0].len();
    let mut result = vec![vec![E::ZERO; num_rows]; num_cols];
    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.par_iter_mut().enumerate().for_each(|(j, col)| {
            *col = matrix[j][i];
        })
    });
    result
}
