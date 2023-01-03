// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand_utils::rand_vector;
use std::time::Duration;
use utils::uninit_vector;

use winter_math::{
    fft::{self, fft_inputs::RowMajor},
    fields::f64::BaseElement,
    StarkField,
};

const SIZES: [usize; 3] = [262_144, 524_288, 1_048_576];

fn interpolate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_interpolate_columns");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES.iter() {
        let num_cols = 256;
        let mut columns: Vec<Vec<BaseElement>> = (0..num_cols).map(|_| rand_vector(size)).collect();
        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("simple", size), |bench| {
            bench.iter_with_large_drop(|| {
                for column in columns.iter_mut() {
                    fft::concurrent::interpolate_poly(column.as_mut_slice(), &inv_twiddles);
                }
            });
        });
    }

    for &size in SIZES.iter() {
        let num_cols = 256;
        let mut columns: Vec<Vec<BaseElement>> = (0..num_cols).map(|_| rand_vector(size)).collect();
        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("with_offset", size), |bench| {
            bench.iter_with_large_drop(|| {
                for column in columns.iter_mut() {
                    fft::concurrent::interpolate_poly_with_offset(
                        column.as_mut_slice(),
                        &inv_twiddles,
                        BaseElement::GENERATOR,
                    );
                }
            });
        });
    }
    group.finish();
}

fn evaluate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_evaluate_columns");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let blowup_factor = 8;

    for &size in SIZES.iter() {
        let num_cols = 256;
        let mut columns: Vec<Vec<BaseElement>> = (0..num_cols).map(|_| rand_vector(size)).collect();
        let twiddles = fft::get_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("simple", size), |bench| {
            bench.iter_with_large_drop(|| {
                for column in columns.iter_mut() {
                    fft::concurrent::evaluate_poly(column.as_mut_slice(), &twiddles);
                }
            });
        });
    }

    for &size in SIZES.iter() {
        let num_cols = 256;
        let mut columns: Vec<Vec<BaseElement>> = (0..num_cols).map(|_| rand_vector(size)).collect();
        let twiddles = fft::get_twiddles::<BaseElement>(size);
        let mut result = unsafe { uninit_vector(size * blowup_factor) };
        group.bench_function(BenchmarkId::new("with_offset", size), |bench| {
            bench.iter_with_large_drop(|| {
                for column in columns.iter_mut() {
                    fft::concurrent::evaluate_poly_with_offset(
                        column.as_mut_slice(),
                        &twiddles,
                        BaseElement::GENERATOR,
                        blowup_factor,
                        &mut result,
                    );
                }
            });
        });
    }
    group.finish();
}

fn interpolate_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_interpolate_matrix");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES.iter() {
        let num_cols = 256;
        let rows: Vec<Vec<BaseElement>> = (0..size).map(|_| rand_vector(num_cols)).collect();

        let row_width = rows[0].len();
        let mut flatten_table = rows.into_iter().flatten().collect::<Vec<_>>();
        let mut table = RowMajor::new(&mut flatten_table, row_width);

        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("simple", size), |bench| {
            bench.iter_with_large_drop(|| {
                fft::concurrent::interpolate_poly(&mut table, &inv_twiddles)
            });
        });
    }

    for &size in SIZES.iter() {
        let num_cols = 256;
        let rows: Vec<Vec<BaseElement>> = (0..size).map(|_| rand_vector(num_cols)).collect();

        let row_width = rows[0].len();
        let mut flatten_table = rows.into_iter().flatten().collect::<Vec<_>>();
        let mut table = RowMajor::new(&mut flatten_table, row_width);

        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("with_offset", size), |bench| {
            bench.iter_with_large_drop(|| {
                fft::concurrent::interpolate_poly_with_offset(
                    &mut table,
                    &inv_twiddles,
                    BaseElement::GENERATOR,
                )
            });
        });
    }
    group.finish();
}

fn evaluate_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_evaluate_matrix");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let blowup_factor = 8;

    for &size in SIZES.iter() {
        let num_cols = 256;
        let rows: Vec<Vec<BaseElement>> = (0..size).map(|_| rand_vector(num_cols)).collect();

        let row_width = rows[0].len();
        let mut flatten_table = rows.into_iter().flatten().collect::<Vec<_>>();
        let mut table = RowMajor::new(&mut flatten_table, row_width);

        let twiddles = fft::get_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("simple", size), |bench| {
            bench.iter_with_large_drop(|| fft::concurrent::evaluate_poly(&mut table, &twiddles));
        });
    }

    for &size in SIZES.iter() {
        let num_cols = 256;
        let rows: Vec<Vec<BaseElement>> = (0..size).map(|_| rand_vector(num_cols)).collect();

        let row_width = rows[0].len();
        let mut flatten_table = rows.into_iter().flatten().collect::<Vec<_>>();
        let table = RowMajor::new(&mut flatten_table, row_width);

        let twiddles = fft::get_twiddles::<BaseElement>(size);
        let mut result = unsafe { uninit_vector(size * num_cols * blowup_factor) };
        let mut result_table = RowMajor::new(&mut result, row_width);
        group.bench_function(BenchmarkId::new("with_offset", size), |bench| {
            bench.iter_with_large_drop(|| {
                fft::concurrent::evaluate_poly_with_offset(
                    &table,
                    &twiddles,
                    BaseElement::GENERATOR,
                    blowup_factor,
                    &mut result_table,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    matrix_group,
    interpolate_columns,
    interpolate_matrix,
    evaluate_columns,
    evaluate_matrix
);
criterion_main!(matrix_group);
