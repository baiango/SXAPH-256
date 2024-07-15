#![feature(portable_simd)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::simd::u64x4;


fn mul_hash(input_data: u64x4) -> u64x4 {
	input_data * u64x4::from_array([140, 91, 171, 62])
}

fn djb2_hash(input_bytes: &[u8]) -> u32 {
	let mut hash_value = 5381;
	for b in input_bytes {
		hash_value = hash_value * 33 + *b as u32;
	}
	hash_value
}

fn vast_hash_impl(input_data: u64x4) -> u64x4 {
	input_data ^ u64x4::from_array([6205865627071447409, 2067898264094941423, 1954899363002243873, 9928278621127670147])
}

fn fnv_1a_hash(data: &[u8]) -> u64 {
	let mut hash = 0xCBF29CE484222325;
	for byte in data.iter() {
		hash ^= *byte as u64;
		hash *= 0x00000100_000001B3;
	}
	hash
}

fn bench_mul_hash_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("mul_hash_8", |b| b.iter(|| mul_hash(black_box(input_data))));
}

fn bench_djb2_hash_8(c: &mut Criterion) {
	let input_data = &[123];
	c.bench_function("djb2_hash_8", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_vast_hash_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("vast_hash_8", |b| b.iter(|| vast_hash_impl(black_box(input_data))));
}

fn bench_fnv_1a_hash_8(c: &mut Criterion) {
	let input_data = &[123];
	c.bench_function("fnv_1a_hash_8", |b| b.iter(|| fnv_1a_hash(black_box(input_data))));
}

fn bench_mul_hash_256(c: &mut Criterion) {
	let input_data = u64x4::splat(0x123456789abcdef0);
	c.bench_function("mul_hash_256", |b| b.iter(|| mul_hash(black_box(input_data))));
}

fn bench_djb2_hash_256(c: &mut Criterion) {
	let input_data = &[
		140, 91, 171, 62,
		140, 91, 171, 62,
		140, 91, 171, 62,
		140, 91, 171, 62
	];
	c.bench_function("djb2_hash_256", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_vast_hash_256(c: &mut Criterion) {
	let input_data = u64x4::splat(0x123456789abcdef0);
	c.bench_function("vast_hash_256", |b| b.iter(|| vast_hash_impl(black_box(input_data))));
}

fn bench_fnv_1a_hash_256(c: &mut Criterion) {
	let input_data = &[
		140, 91, 171, 62,
		140, 91, 171, 62,
		140, 91, 171, 62,
		140, 91, 171, 62
	];
	c.bench_function("fnv_1a_hash_256", |b| b.iter(|| fnv_1a_hash(black_box(input_data))));
}


criterion_group!(
	benches,
	bench_mul_hash_8, bench_djb2_hash_8, bench_vast_hash_8, bench_fnv_1a_hash_8,
	bench_mul_hash_256, bench_djb2_hash_256, bench_vast_hash_256, bench_fnv_1a_hash_256
);
criterion_main!(benches);
