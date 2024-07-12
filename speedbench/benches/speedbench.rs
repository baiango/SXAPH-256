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

fn shift_xor_add_impl(input_data: u64x4) -> u64x4 {
	let ls_dat = input_data << u64x4::splat(31);
	let hash_256_a = ls_dat ^ u64x4::from_array([10449833687391262720, 12708638996869552128, 12083284059032971264, 5098133568216696832]);
	let rs_dat = input_data >> u64x4::splat(24);
	let hash_256_b = rs_dat ^ u64x4::from_array([9858113524293627904, 2849775663957600256, 12247827806936932352, 1651210329918801920]);
	hash_256_a + hash_256_b
}

fn bench_mul_hash_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("mul_hash_8", |b| b.iter(|| mul_hash(black_box(input_data))));
}

fn bench_djb2_hash_8(c: &mut Criterion) {
	let input_data = &[123];
	c.bench_function("djb2_hash_8", |b| b.iter(|| djb2_hash(black_box(input_data))));
}

fn bench_shift_xor_add_hash_8(c: &mut Criterion) {
	let input_data = u64x4::from_array([123, 0, 0, 0]);
	c.bench_function("shift_xor_add_hash_8", |b| b.iter(|| shift_xor_add_impl(black_box(input_data))));
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

fn bench_shift_xor_add_hash_256(c: &mut Criterion) {
	let input_data = u64x4::splat(0x123456789abcdef0);
	c.bench_function("shift_xor_add_hash_256", |b| b.iter(|| shift_xor_add_impl(black_box(input_data))));
}

criterion_group!(
	benches,
	bench_mul_hash_8, bench_djb2_hash_8, bench_shift_xor_add_hash_8,
	bench_mul_hash_256, bench_djb2_hash_256, bench_shift_xor_add_hash_256,
);
criterion_main!(benches);
