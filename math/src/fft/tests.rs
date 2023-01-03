// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    fft::fft_inputs::RowMajor,
    field::{f64::BaseElement, StarkField},
    polynom,
    utils::{get_power_series, log2},
    FieldElement,
};
use rand_utils::rand_vector;
use utils::{collections::Vec, uninit_vector};

// CORE ALGORITHMS
// ================================================================================================

// SLICES
// --------------------------------------------------------------------------------------------

#[test]
fn fft_in_place() {
    // degree 3
    let n = 8;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let expected = polynom::eval_many(p.as_mut_slice(), &domain);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::concurrent::evaluate_poly(&mut p.as_mut_slice(), &twiddles);
    // super::permute(p.as_mut_slice());
    assert_eq!(expected, p);

    // degree 7
    let n = 8;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    let expected = polynom::eval_many(p.as_mut_slice(), &domain);
    super::serial::fft_in_place(&mut p.as_mut_slice(), &twiddles, 1, 1, 0);
    super::permute(p.as_mut_slice());
    assert_eq!(expected, p);

    // degree 15
    let n = 16;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let twiddles = super::get_twiddles::<BaseElement>(16);
    let expected = polynom::eval_many(p.as_mut_slice(), &domain);
    super::serial::fft_in_place(&mut p.as_mut_slice(), &twiddles, 1, 1, 0);
    super::permute(p.as_mut_slice());
    assert_eq!(expected, p);

    // degree 1023
    let n = 1024;
    let mut p = rand_vector(n);
    let domain = build_domain(n);
    let expected = polynom::eval_many(p.as_mut_slice(), &domain);
    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::serial::fft_in_place(&mut p.as_mut_slice(), &twiddles, 1, 1, 0);
    super::permute(p.as_mut_slice());
    assert_eq!(expected, p);
}

// MATRIX
// --------------------------------------------------------------------------------------------

#[test]
fn fft_in_place_matrix() {
    // degree 3
    let n = 4;
    let num_polys = 10;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let mut flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMajor::new(&mut flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let mut eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::serial::fft_in_place(&mut matrix, &twiddles, 1, 1, 0);
    super::serial::permute(&mut matrix);
    assert_eq!(
        RowMajor::new(&mut eval_cols_faltten, row_width).get_data(),
        matrix.get_data()
    );

    // degree 7
    let n = 8;
    let num_polys = 10;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let mut flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMajor::new(&mut flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let mut eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::serial::fft_in_place(&mut matrix, &twiddles, 1, 1, 0);
    super::serial::permute(&mut matrix);
    assert_eq!(
        RowMajor::new(&mut eval_cols_faltten, row_width).get_data(),
        matrix.get_data()
    );

    // degree 15
    let n = 16;
    let num_polys = 60;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let mut flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMajor::new(&mut flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let mut eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::serial::fft_in_place(&mut matrix, &twiddles, 1, 1, 0);
    super::serial::permute(&mut matrix);
    assert_eq!(
        RowMajor::new(&mut eval_cols_faltten, row_width).get_data(),
        matrix.get_data()
    );

    // degree 1023
    let n = 1024;
    let num_polys = 120;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let mut flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut matrix = RowMajor::new(&mut flatten_rows, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let mut eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    let twiddles = super::get_twiddles::<BaseElement>(n);
    super::serial::fft_in_place(&mut matrix, &twiddles, 1, 1, 0);
    super::serial::permute(&mut matrix);
    assert_eq!(
        RowMajor::new(&mut eval_cols_faltten, row_width).get_data(),
        matrix.get_data()
    );
}

#[test]
fn concurrent_test() {
    let n = 8;
    let num_polys = 1;
    let mut columns: Vec<Vec<BaseElement>> = (0..num_polys).map(|_| rand_vector(n)).collect();
    let rows = transpose(columns.clone());
    let row_width = rows[0].len();
    let mut flatten_rows = rows.into_iter().flatten().collect::<Vec<_>>();
    let mut flatten_copy = flatten_rows.clone();
    let mut matrix = RowMajor::new(&mut flatten_rows, row_width);
    // let mut matrix_concurrent = RowMajor::new(&mut flatten_copy, row_width);

    let domain = build_domain(n);
    for p in columns.iter_mut() {
        *p = polynom::eval_many(p, &domain);
    }
    let eval_col = transpose(columns);
    let mut eval_cols_faltten = eval_col.into_iter().flatten().collect::<Vec<_>>();

    // let mut result = unsafe { uninit_vector(n * num_polys * 8) };
    let mut result_matrix = unsafe { uninit_vector(n * num_polys * 8) };
    let mut result_table = RowMajor::new(&mut result_matrix, row_width);

    let twiddles = super::get_twiddles::<BaseElement>(n);
    let inv_twiddles = super::get_inv_twiddles::<BaseElement>(n);

    // println!("concurrent slice: ");
    // super::concurrent::evaluate_poly_with_offset(
    //     flatten_copy.as_mut_slice(),
    //     &twiddles,
    //     BaseElement::GENERATOR,
    //     8,
    //     &mut result,
    // );

    // println!("serial: ");
    // super::serial::fft_in_place(&mut matrix, &twiddles, 1, 1, 0);

    println!("concurrent matrix: ");
    super::concurrent::evaluate_poly_with_offset(
        &mut matrix,
        &twiddles,
        BaseElement::GENERATOR,
        8,
        &mut result_table,
    );

    // println!("Checking results...");
    // assert_eq!(flatten_copy, matrix.get_data());

    // println!("Permuting...");
    // super::serial::permute(&mut matrix);
    // super::concurrent::permute(&mut matrix_concurrent);

    // println!("Checking results...");
    // assert_eq!(matrix_concurrent.get_data(), matrix.get_data());

    // assert_eq!(
    //     RowMajor::new(&mut eval_cols_faltten, row_width).get_data(),
    //     matrix.get_data()
    // );
}

#[test]
fn fft_get_twiddles() {
    let n = super::MIN_CONCURRENT_SIZE * 2;
    let g = BaseElement::get_root_of_unity(log2(n));

    let mut expected = get_power_series(g, n / 2);
    super::permute(&mut expected);

    let twiddles = super::get_twiddles::<BaseElement>(n);
    assert_eq!(expected, twiddles);
}

// HELPER FUNCTIONS
// ================================================================================================

/// Builds a domain of size `size` using the primitive element of the field.
fn build_domain(size: usize) -> Vec<BaseElement> {
    let g = BaseElement::get_root_of_unity(log2(size));
    get_power_series(g, size)
}

/// Transposes a matrix stored in a column major format to a row major format.
/// fn transpose<E: FieldElement>(v: Vec<Vec<E>>) -> Vec<Vec<E>> {
fn transpose<E: FieldElement>(matrix: Vec<Vec<E>>) -> Vec<Vec<E>> {
    let num_rows = matrix.len();
    let num_cols = matrix[0].len();
    let mut result = vec![vec![E::ZERO; num_rows]; num_cols];
    for i in 0..num_rows {
        for j in 0..num_cols {
            result[j][i] = matrix[i][j];
        }
    }
    result
}
