// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand_utils::rand_vector;
use std::time::Duration;

use winter_math::{
    fft::{self, fft_inputs::RowMajor},
    fields::f64::BaseElement,
};

const SIZES: [usize; 3] = [262_144, 524_288, 1_048_576];

fn interpolate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_interpolate_columns");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES.iter() {
        let num_cols = 128;
        let _stride = 8;
        let mut columns: Vec<Vec<BaseElement>> = (0..num_cols).map(|_| rand_vector(size)).collect();
        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("columns", size), |bench| {
            bench.iter_with_large_drop(|| {
                for column in columns.iter_mut() {
                    fft::serial::interpolate_poly(column.as_mut_slice(), &inv_twiddles);
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
        let num_cols = 128;
        let _stride = 8;
        let rows: Vec<Vec<BaseElement>> = (0..size).map(|_| rand_vector(num_cols)).collect();

        let row_width = rows[0].len();
        let mut flatten_table = rows.into_iter().flatten().collect::<Vec<_>>();
        let mut table = RowMajor::new(&mut flatten_table, row_width);

        let inv_twiddles = fft::get_inv_twiddles::<BaseElement>(size);
        group.bench_function(BenchmarkId::new("matrix", size), |bench| {
            bench.iter_with_large_drop(|| fft::serial::interpolate_poly(&mut table, &inv_twiddles));
        });
    }
    group.finish();
}

criterion_group!(matrix_group, interpolate_columns, interpolate_matrix);
criterion_main!(matrix_group);
