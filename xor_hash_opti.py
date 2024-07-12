import zlib
import hashlib
import multiprocessing
import random
import functools
import json
import numpy as np
from fnvhash import fnv1a_32


def shift_xor_add_impl(input_data, l=31, r=24, hash_256_a=np.array([10449833687391262720, 12708638996869552128, 12083284059032971264, 5098133568216696832], dtype=np.uint64), hash_256_b=np.array([9858113524293627904, 2849775663957600256, 12247827806936932352, 1651210329918801920], dtype=np.uint64)): # 22327 BruteX_password.list
	ls_dat = np.left_shift(input_data, l)
	hash_256_a = np.bitwise_xor(ls_dat, hash_256_a)
	rs_dat = np.right_shift(input_data, r)
	hash_256_b = np.bitwise_xor(rs_dat, hash_256_b)
	return np.add(hash_256_a, hash_256_b)

def load_test(hash_file='conficker.txt'):
	chunks = []
	with open(hash_file, 'rb') as f:
		for i, line in enumerate(f.read().split(b'\n')):
			line = line.rstrip()
			padding_bytes = 8 - (len(line) % 8)
			padded_line = line + b'\x00' * padding_bytes
			chunk = np.frombuffer(padded_line, dtype=np.uint64)

			padded_length = ((len(chunk) - 1) // 4 + 1) * 4
			padded_data = np.pad(chunk, (0, padded_length - len(chunk)), 'constant')
			split_chunks = np.split(padded_data, padded_length // 4)
			chunks.append(split_chunks)
	return chunks

const_u64x4_chunks = load_test()

def test_hash_distribution(func, bucket_sizes=[11, 13, 17, 19], u64x4_chunks=const_u64x4_chunks):
	chi2 = 0
	for bucket_size in bucket_sizes:
		buckets = np.zeros(bucket_size, dtype=np.uint64)
		num_hashes = len(u64x4_chunks)

		for i, chk in enumerate(u64x4_chunks):
			hash_value = np.sum([func(c) for c in chk])
			bucket_indices = int(hash_value) % bucket_size
			buckets[bucket_indices] += 1

		expected_frequency = num_hashes // bucket_size
		chi2 += np.sum((buckets - expected_frequency) ** 2)
	return chi2

def is_prime(n):
	if n <= 1:
		return False
	if n <= 3:
		return True
	if n % 2 == 0 or n % 3 == 0:
		return False
	i = 5
	while i * i <= n:
		if n % i == 0 or n % (i + 2) == 0:
			return False
		i += 6
	return True

def objective(bucket_sizes=[i for i in range(10, 20) if is_prime(i)]):
	l = random.randrange(64)
	r = random.randrange(64)
	hash_256_a = np.random.randint(np.iinfo(np.uint64).max, size=4, dtype=np.uint64)
	hash_256_b = np.random.randint(np.iinfo(np.uint64).max, size=4, dtype=np.uint64)

	hash_func = functools.partial(shift_xor_add_impl, l=l, r=r, hash_256_a=hash_256_a, hash_256_b=hash_256_b)

	chi_squared = test_hash_distribution(hash_func, bucket_sizes=bucket_sizes)
	return {"best_params": {"l": l, "r": r, "hash_256_a": [int(x) for x in hash_256_a], "hash_256_b": [int(x) for x in hash_256_b]}, "best_value": chi_squared}

def brute_search_hash_func(jsonl_output_path='xor_hash_opti_2.jsonl'):
	best = {"best_params": {}, "best_value": np.iinfo(np.uint64).max}

	results = []
	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		for i in range(4000):
			results.append(pool.apply_async(objective))
		pool.close()
		pool.join()

	for result in results:
		result = result.get()
		if result["best_value"] < best["best_value"]:
			best = {"best_params": result["best_params"], "best_value": result["best_value"]}

	print("best", best)
	with open(jsonl_output_path, 'a') as f:
		json.dump(best, f)
		f.write('\n')

def sha256_hash(input_bytes):
	# Compute SHA-256 hash
	hash_object = hashlib.sha256(input_bytes)
	hash_hex = hash_object.hexdigest()

	# Convert the hash value to a 4-element array of 64-bit integers
	hash_int = int(hash_hex, 16)
	hash_array = np.array([(hash_int >> (64 * i)) & 0xffffffffffffffff for i in range(4)], dtype=np.uint64)

	# Return the hash value as a 4-element array of 64-bit integers
	return hash_array

def crc32_hash(input_bytes):
	# Compute CRC32 hash
	hash_object = zlib.crc32(input_bytes)

	# The CRC32 hash value is a single 32-bit integer, so we return it as a 1-element array
	hash_array = np.array([hash_object], dtype=np.uint32)

	# Return the hash value as a 1-element array of 32-bit integers
	return hash_array

def fnv1a_32_hash(input_bytes):
	hash_array = np.array([fnv1a_32(bytes(input_bytes))], dtype=np.uint32)
	return hash_array

def djb2_hash(input_bytes):
	hash_value = np.uint32(5381)
	for b in input_bytes:
		hash_value = hash_value * np.uint32(33) + b
	return hash_value

def mul_hash(input_data, hash_256=np.array([140, 91, 171, 62], dtype=np.uint64)):
	return hash_256 * input_data

def chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_list=load_test('BruteX_password.list')):
	test_distribution = functools.partial(test_hash_distribution, bucket_sizes=bucket_sizes, u64x4_chunks=hash_list)

	mul_hash_score = test_distribution(mul_hash)
	print("mul_hash chi2_benchmark score:", mul_hash_score)

	djb2_hash_score = test_distribution(djb2_hash)
	print("djb2_hash chi2_benchmark score:", djb2_hash_score)

	shift_xor_add_hash_score = test_distribution(shift_xor_add_impl)
	print("shift_xor_add_hash chi2_benchmark score:", shift_xor_add_hash_score)

	fnv1a_32_hash_hash_score = test_distribution(fnv1a_32_hash)
	print("fnv1a_32_hash_hash chi2_benchmark score:", fnv1a_32_hash_hash_score)

	crc32_hash_score = test_distribution(crc32_hash)
	print("crc32_hash chi2_benchmark score:", crc32_hash_score)

	sha256_hash_score = test_distribution(sha256_hash)
	print("sha256_hash chi2_benchmark score:", sha256_hash_score)

if __name__ == '__main__':
	assert shift_xor_add_impl(np.array([123, 123, 123, 123], dtype=np.uint64)).all() == np.array([1861203148712757248, 15558414740284047360, 5884367880307181568, 6749343681239650304], dtype=np.uint64).all()

	print("test_hash_distribution(mul_hash): ", test_hash_distribution(mul_hash)) # 1237
	print("test_hash_distribution(djb2_hash): ", test_hash_distribution(djb2_hash)) # 8207
	print("test_hash_distribution(shift_xor_add_impl): ", test_hash_distribution(shift_xor_add_impl)) # 223
	print("test_hash_distribution(fnv1a_32_hash): ", test_hash_distribution(fnv1a_32_hash)) # 683
	print("test_hash_distribution(crc32_hash):" , test_hash_distribution(crc32_hash)) # 695
	print("test_hash_distribution(sha256_hash):" , test_hash_distribution(sha256_hash)) # 793
	print()
	chi2_benchmark()
	print()
	chi2_benchmark(range(1, 101))

	# while True:
	# 	brute_search_hash_func()
