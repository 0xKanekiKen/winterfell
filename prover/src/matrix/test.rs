// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::math::{
    fft::{fft_inputs::FftInputs, get_inv_twiddles, get_twiddles},
    fields::f64::BaseElement,
    get_power_series, log2, polynom, StarkField,
};

use super::RowMatrix;
use math::FieldElement;
use rand_utils::rand_vector;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use utils::collections::Vec;

#[test]
fn test_fft_in_place_matrix() {
    // degree 3
    let n = 4;
    let num_polys = 10;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    FftInputs::fft_in_place(&mut matrix, &twiddles);
    FftInputs::permute(&mut matrix);
    assert_eq!(eval_cols_faltten, matrix.get_data());

    // degree 7
    let n = 8;
    let num_polys = 10;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    FftInputs::fft_in_place(&mut matrix, &twiddles);
    FftInputs::permute(&mut matrix);
    assert_eq!(eval_cols_faltten, matrix.get_data());

    // degree 15
    let n = 16;
    let num_polys = 60;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    FftInputs::fft_in_place(&mut matrix, &twiddles);
    FftInputs::permute(&mut matrix);
    assert_eq!(eval_cols_faltten, matrix.get_data());

    // degree 1023
    let n = 1024;
    let num_polys = 120;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    FftInputs::fft_in_place(&mut matrix, &twiddles);
    FftInputs::permute(&mut matrix);
    assert_eq!(eval_cols_faltten, matrix.get_data());
}

#[test]
fn test_eval_poly_with_offset_matrix() {
    let n = 128;
    let num_polys = 64;
    let blowup_factor = 8;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n * blowup_factor);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();

    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(matrix.len());
    let eval_vector =
        RowMatrix::evaluate_poly_with_offset(&matrix, &twiddles, offset, blowup_factor);
    assert_eq!(eval_cols_faltten, eval_vector.get_data());
}

#[test]
fn test_interpolate_poly_with_offset_matrix() {
    // degree 127
    let n = 128;
    let num_polys = 64;

    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();
    let mut interpolate_matrix = RowMatrix::new(eval_cols_faltten, row_width);

    let inv_twiddles = get_inv_twiddles::<BaseElement>(matrix.len());
    RowMatrix::interpolate_poly_with_offset(&mut interpolate_matrix, &inv_twiddles, offset);
    assert_eq!(interpolate_matrix.get_data(), matrix.get_data());
}

#[test]
fn test_interpolate_poly_matrix() {
    let n = 128;
    let num_polys = 64;

    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();
    let mut interpolate_matrix = RowMatrix::new(eval_cols_faltten, row_width);

    let inv_twiddles = get_inv_twiddles::<BaseElement>(matrix.len());
    RowMatrix::interpolate_poly(&mut interpolate_matrix, &inv_twiddles);
    assert_eq!(interpolate_matrix.get_data(), matrix.get_data());
}

// CONCURRENT TESTS
// ================================================================================================

#[test]
fn test_eval_poly_matrix_concurrent() {
    let n = 1024;
    let num_polys = 64;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose_concurrent(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose_concurrent(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    RowMatrix::evaluate_poly(&mut matrix, &twiddles);
    // assert_eq!(eval_cols_faltten, matrix.get_data());
}

#[test]
fn test_eval_poly_with_offset_matrix_concurrent() {
    let n = 128;
    let num_polys = 64;
    let blowup_factor = 8;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose_concurrent(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n * blowup_factor);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();

    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose_concurrent(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = get_twiddles::<BaseElement>(n);
    let eval_vector = RowMatrix::evaluate_poly_with_offset_concurrent(
        &matrix,
        &twiddles,
        BaseElement::GENERATOR,
        blowup_factor,
    );
    assert_eq!(eval_cols_faltten, eval_vector.get_data());
}

#[test]
fn test_interpolate_poly_matrix_concurrent() {
    let n = 1024;
    let num_polys = 64;

    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose_concurrent(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose_concurrent(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();
    let mut interpolate_matrix = RowMatrix::new(eval_cols_faltten, row_width);

    let inv_twiddles = get_inv_twiddles::<BaseElement>(matrix.len());
    RowMatrix::interpolate_poly_concurrent(&mut interpolate_matrix, &inv_twiddles);
    assert_eq!(interpolate_matrix.get_data(), matrix.get_data());
}

#[test]
fn test_interpolate_poly_with_offset_matrix_concurrent() {
    // degree 15
    let n = 1024;
    let num_polys = 64;

    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose_concurrent(columns.clone());
    let row_width = rows[0].len();
    let flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let matrix = RowMatrix::new(flatten_rows, row_width);

    let offset = BaseElement::GENERATOR;
    let domain = build_domain(n);
    let shifted_domain = domain.iter().map(|&x| x * offset).collect::<Vec<_>>();
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &shifted_domain);
    }
    let eval_col = transpose_concurrent(columns);
    let eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();
    let mut interpolate_matrix = RowMatrix::new(eval_cols_faltten, row_width);

    let inv_twiddles = get_inv_twiddles::<BaseElement>(matrix.len());
    RowMatrix::interpolate_poly_with_offset_concurrent(
        &mut interpolate_matrix,
        &inv_twiddles,
        offset,
    );
    assert_eq!(interpolate_matrix.get_data(), matrix.get_data());
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
