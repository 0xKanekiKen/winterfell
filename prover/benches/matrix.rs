// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand_utils::rand_vector;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use std::time::Duration;

use math::{
    fft::{self},
    fields::f64::BaseElement,
    StarkField,
};

use winter_prover::{Matrix, RowMatrix};

const SIZE: usize = 524_288;
const NUM_POLYS: [usize; 1] = [64];

fn evaluate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_eval_poly_with_offset_column");
    let blowup_factor = 8;

    for &num_poly in NUM_POLYS.iter() {
        let columns: Vec<Vec<BaseElement>> = (0..num_poly).map(|_| rand_vector(SIZE)).collect();
        let mut column_matrix = Matrix::new(columns);
        let twiddles = fft::get_twiddles::<BaseElement>(SIZE);
        group.bench_function(BenchmarkId::new("with_offset", num_poly), |bench| {
            bench.iter_with_large_drop(|| {
                iter_mut!(column_matrix.columns).for_each(|column| {
                    fft::serial::evaluate_poly_with_offset(
                        column.as_mut_slice(),
                        &twiddles,
                        BaseElement::GENERATOR,
                        blowup_factor,
                    );
                });
            });
        });
    }
    group.finish();
}

fn evaluate_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_eval_poly_with_offset_matrix");
    let blowup_factor = 8;

    for &num_poly in NUM_POLYS.iter() {
        let columns: Vec<Vec<BaseElement>> = (0..num_poly).map(|_| rand_vector(SIZE)).collect();
        let mut column_matrix = Matrix::new(columns);

        group.bench_function(BenchmarkId::new("with_no_segments", num_poly), |bench| {
            bench.iter_with_large_drop(|| RowMatrix::from_polys(&column_matrix, blowup_factor));
        });
    }
    group.finish();
}

criterion_group!(matrix_group, evaluate_matrix, evaluate_columns,);
criterion_main!(matrix_group);

#[macro_export]
macro_rules! iter_mut {
    ($e: expr) => {{
        // #[cfg(feature = "concurrent")]
        // let result = $e.par_iter_mut();

        // #[cfg(not(feature = "concurrent"))]
        let result = $e.iter_mut();

        result
    }};
    ($e: expr, $min_len: expr) => {{
        // #[cfg(feature = "concurrent")]
        // let result = $e.par_iter().with_min_len($min_len);

        // #[cfg(not(feature = "concurrent"))]
        // let result = $e.iter();

        // result
    }};
}
