#![feature(portable_simd)]
use std::simd::u64x4;
use std::arch::x86_64::{_mm_loadu_si128, _mm_add_epi64, _mm_shuffle_epi32, _mm_cvtsi128_si64};
use std::mem;


#[inline] // Reduces latency
pub fn mul_hash_impl(input_data: u64x4) -> u64x4 {
	input_data * u64x4::from_array([140, 91, 171, 62])
}

pub fn mul_hash(input_data: &[u64x4]) -> u64 {
	let mut hash = u64x4::splat(0);
	for i in 0..input_data.len() {
		hash += mul_hash_impl(input_data[i]);
	}
	sum_u64x4_icx(hash)
}

pub fn djb2_hash(input_bytes: &[u8]) -> u32 {
	let mut hash_value = 5381;
	for b in input_bytes {
		hash_value = hash_value * 33 + *b as u32;
	}
	hash_value
}

#[inline] // Reduces latency
pub fn vast_hash_impl(input_data: u64x4) -> u64x4 {
	input_data ^ u64x4::from_array([6205865627071447409, 2067898264094941423, 1954899363002243873, 9928278621127670147])
}

pub fn vast_hash(input_data: &[u64x4]) -> u64 {
	let mut hash = u64x4::splat(0);
	for i in 0..input_data.len() {
		hash += vast_hash_impl(input_data[i]);
	}
	sum_u64x4_icx(hash)
}

pub fn fnv_1a_hash(data: &[u8]) -> u64 {
	let mut hash = 0xCBF29CE484222325;
	for byte in data.iter() {
		hash ^= *byte as u64;
		hash *= 0x00000100_000001B3;
	}
	hash
}

// ---------- 📄 Miscellaneous ---------- //
pub fn sum_u64x4_scalar(simd: u64x4) -> u64 {
	let arr: [u64; 4] = unsafe { std::mem::transmute(simd) };
	arr[0].wrapping_add(arr[1].wrapping_add(arr[2]).wrapping_add(arr[3]))
}

#[inline] // Same speed as the scalar version
pub fn sum_u64x4_icx(simd: u64x4) -> u64 {
	let arr: [u64; 4] = unsafe { mem::transmute(simd) };
	let v1 = unsafe { _mm_loadu_si128(arr.as_ptr() as *const _) };
	let v2 = unsafe { _mm_loadu_si128((arr.as_ptr().add(2)) as *const _) };
	let v3 = unsafe { _mm_add_epi64(v1, v2) };
	let v4 = unsafe { _mm_shuffle_epi32(v3, 0b11101110) }; // (v3, [2, 3, 2, 3])
	let v5 = unsafe { _mm_add_epi64(v3, v4) };
	unsafe { _mm_cvtsi128_si64(v5) as u64 }
}


#[cfg(test)]
mod tests {
	use super::*;
	use std::simd::u64x4;

	#[test]
	fn test_vast_hash_impl() {
		let input_data = u64x4::splat(123);
		let result = vast_hash_impl(input_data);
		assert_eq!(result, u64x4::from_array([6205865627071447306, 2067898264094941332, 1954899363002243930, 9928278621127670264]));
	}

	#[test]
	fn test_vast_hash() {
		let input_data = vec![u64x4::splat(123), u64x4::splat(123)];
		let result = vast_hash(&input_data);
		assert_eq!(result, 3420395603173502432);
	}

	#[test]
	fn test_sum_u64x4_icx() {
		let simd = u64x4::from_array([1, 2, 3, 4]);
		assert_eq!(sum_u64x4_icx(simd), sum_u64x4_scalar(simd));

		let simd = u64x4::from_array([5, 6, 7, 8]);
		assert_eq!(sum_u64x4_icx(simd), sum_u64x4_scalar(simd));

		let simd = u64x4::splat(0);
		assert_eq!(sum_u64x4_icx(simd), sum_u64x4_scalar(simd));

		let simd = u64x4::splat(u64::MAX);
		assert_eq!(sum_u64x4_icx(simd), sum_u64x4_scalar(simd));
	}
}
