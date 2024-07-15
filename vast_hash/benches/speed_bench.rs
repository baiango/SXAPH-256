#![feature(portable_simd)]
use vast_hash::*;
use std::simd::u64x4;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ---------- 🥟 8-bit (small) hash ---------- //
fn bench_mul_hash_impl_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("mul_hash_impl_8", |b| b.iter(|| mul_hash_impl(black_box(input_data))));
}

fn bench_djb2_hash_8(c: &mut Criterion) {
	let input_data = &[123];
	c.bench_function("djb2_hash_8", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_vast_impl_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("vast_hash_impl_8", |b| b.iter(|| vast_hash_impl(black_box(input_data))));
}

fn bench_fnv_1a_hash_8(c: &mut Criterion) {
	let input_data = &[123];
	c.bench_function("fnv_1a_hash_8", |b| b.iter(|| fnv_1a_hash(black_box(input_data))));
}

// ---------- 🍔 256-bit (standard) hash ---------- //
fn bench_mul_hash_impl_256(c: &mut Criterion) {
	let input_data = u64x4::splat(0x123456789abcdef0);
	c.bench_function("mul_hash_impl_256", |b| b.iter(|| mul_hash_impl(black_box(input_data))));
}

fn bench_djb2_hash_256(c: &mut Criterion) {
	let input_data = &vec![140, 91, 171, 62].repeat(4);
	c.bench_function("djb2_hash_256", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_vast_hash_impl_256(c: &mut Criterion) {
	let input_data = u64x4::splat(0x123456789abcdef0);
	c.bench_function("vast_hash_impl_256", |b| b.iter(|| vast_hash_impl(black_box(input_data))));
}

fn bench_fnv_1a_hash_256(c: &mut Criterion) {
	let input_data = &vec![140, 91, 171, 62].repeat(4);
	c.bench_function("fnv_1a_hash_256", |b| b.iter(|| fnv_1a_hash(black_box(input_data))));
}

// ---------- 🍉 1 MiB (MP3) hash ---------- //
fn bench_mul_hash_1mib(c: &mut Criterion) {
	let input_data = &vec![u64x4::splat(0x123456789abcdef0); 32_768];
	c.bench_function("mul_hash_1mib", |b| b.iter(|| mul_hash(black_box(input_data))));
}

fn bench_djb2_hash_1mib(c: &mut Criterion) {
	let input_data = &vec![140, 91, 171, 62].repeat(1_048_576 / 4);
	c.bench_function("djb2_hash_1mib", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_vast_hash_1mib(c: &mut Criterion) {
	let input_data = &vec![u64x4::splat(0x123456789abcdef0); 32_768];
	c.bench_function("vast_hash_1mib", |b| b.iter(|| vast_hash(black_box(input_data))));
}

fn bench_fnv_1a_hash_1mib(c: &mut Criterion) {
	let input_data = &vec![140, 91, 171, 62].repeat(1_048_576 / 4);
	c.bench_function("fnv_1a_hash_1mib", |b| b.iter(|| fnv_1a_hash(black_box(input_data))));
}

// ---------- 📄 Miscellaneous ---------- //
fn bench_sum_u64x4_scalar(c: &mut Criterion) {
	let simd = u64x4::from_array([1, 2, 3, 4]);
	c.bench_function("sum_u64x4_scalar", |b| b.iter(|| sum_u64x4_scalar(simd)));
}

fn bench_sum_u64x4_icx(c: &mut Criterion) {
	let simd = u64x4::from_array([1, 2, 3, 4]);
	c.bench_function("sum_u64x4_icx", |b| b.iter(|| sum_u64x4_icx(simd)));
}


criterion_group!(
	benches,
	bench_mul_hash_impl_8, bench_djb2_hash_8, bench_vast_impl_8, bench_fnv_1a_hash_8, // 🥟 8-bit (small) hash
	bench_mul_hash_impl_256, bench_djb2_hash_256, bench_vast_hash_impl_256, bench_fnv_1a_hash_256, // 🍔 256-bit (standard) hash
	bench_mul_hash_1mib, bench_djb2_hash_1mib, bench_vast_hash_1mib, bench_fnv_1a_hash_1mib, // 🍉 1 MiB (MP3) hash
	bench_sum_u64x4_scalar, bench_sum_u64x4_icx // 📄 Miscellaneous
);
criterion_main!(benches);
