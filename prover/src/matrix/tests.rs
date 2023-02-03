// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    evaluate_poly_with_offset_concurrent,
    math::{
        fft::get_twiddles, fields::f64::BaseElement, get_power_series, log2, polynom, StarkField,
    },
    matrix::transpose,
    Matrix, RowMatrix,
};
use rand_utils::rand_vector;
use utils::collections::Vec;

#[test]
fn test_eval_poly_with_offset_matrix() {
    let n = 1024 * 512;
    let num_polys = 64;
    let blowup_factor = 8;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let row_matrix = RowMatrix::from_polys(&Matrix::new(columns.clone()), blowup_factor);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n * blowup_factor);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();

    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose(&Matrix::new(columns.clone()));

    assert_eq!(eval_col.as_data(), row_matrix.as_data());
}

// CONCURRENT TESTS
// ================================================================================================

#[test]
fn test_eval_poly_with_offset_matrix_concurrent() {
    let n = 128;
    let num_polys = 16;
    let blowup_factor = 2;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let row_matrix = transpose(&Matrix::new(columns.clone()));

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n * blowup_factor);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();

    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose(&Matrix::new(columns));
    let twiddles = get_twiddles::<BaseElement>(n);
    let eval_matrix =
        evaluate_poly_with_offset_concurrent(&row_matrix, &twiddles, offset, blowup_factor);

    assert_eq!(eval_col.as_data(), eval_matrix.as_data());
}

// HELPER FUNCTIONS
// ================================================================================================

/// Builds a domain of size `size` using the primitive element of the field.
fn build_domain(size: usize) -> Vec<BaseElement> {
    let g = BaseElement::get_root_of_unity(log2(size));
    get_power_series(g, size)
}
