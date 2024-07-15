import zlib
import hashlib
import multiprocessing
import random
import functools
import json
import numpy as np
import sympy
from fnvhash import fnv1a_32


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

brute_operators = [
	{"f": np.bitwise_or, "max": np.iinfo(np.uint64).max, "size": 4},
	{"f": np.bitwise_and, "max": np.iinfo(np.uint64).max, "size": 4},
	{"f": np.left_shift, "max": 64, "size": 1},
	{"f": np.right_shift, "max": 64, "size": 1},
	{"f": np.bitwise_xor, "max": np.iinfo(np.uint64).max, "size": 4},
	{"f": np.add, "max": np.iinfo(np.uint64).max, "size": 4},
	{"f": np.subtract, "max": np.iinfo(np.uint64).max, "size": 4},
]

def generate_operator():
	operator = brute_operators[np.random.randint(len(brute_operators))]
	result = {"f": operator["f"]}
	if operator["size"] > 1:
		v = np.zeros(operator["size"], dtype=np.uint64)
		for i in range(operator["size"]):
			v[i] = sympy.randprime(1, operator["max"])
		result |= {"v": v}
	else:
		v = np.array(sympy.randprime(1, operator["max"]), dtype=np.uint64)
		result |= {"v": v}
	return result

def vast_hash_impl(input_data, hash_256=np.array([6205865627071447409, 2067898264094941423, 1954899363002243873, 9928278621127670147], dtype=np.uint64)):
	return np.bitwise_xor(input_data, hash_256)

def vast_hash(input_data):
	return np.sum([vast_hash_impl(dat) for dat in input_data])

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

const_u64x4_chunks = load_test('BruteX_password.list')

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
	return int(chi2)

def objective(bucket_sizes=[i for i in range(10, 20) if is_prime(i)]):
	hash_256 = np.array([sympy.randprime(1, np.iinfo(np.uint64).max) for _ in range(4)], dtype=np.uint64)

	hash_func = functools.partial(vast_hash_impl, hash_256=hash_256)
	chi_squared = test_hash_distribution(hash_func, bucket_sizes=bucket_sizes)

	return {"best_params": {"hash_256": hash_256.tolist()}, "best_value": chi_squared}

def brute_search_hash_func(jsonl_output_path='vast_hash_opti.jsonl'):
	best = {"best_params": {}, "best_value": np.iinfo(np.uint64).max}

	results = []
	with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
		for i in range(1000):
			results.append(pool.apply_async(objective))
		pool.close()
		pool.join()

	for result in results:
		result = result.get()
		if result["best_value"] < best["best_value"]:
			best = {"best_params": result["best_params"], "best_value": result["best_value"]}

	print("best:", best)
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

def chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_path='BruteX_password.list'):
	test_distribution = functools.partial(test_hash_distribution, bucket_sizes=bucket_sizes, u64x4_chunks=load_test(hash_path))

	mul_hash_score = test_distribution(mul_hash)
	print("mul_hash chi2_benchmark score:", hash_path, ":", mul_hash_score, ":", bucket_sizes)

	djb2_hash_score = test_distribution(djb2_hash)
	print("djb2_hash chi2_benchmark score:", hash_path, ":", djb2_hash_score, ":", bucket_sizes)

	vast_hash_score = test_distribution(vast_hash_impl)
	print("vast_hash chi2_benchmark score:", hash_path, ":", vast_hash_score, ":", bucket_sizes)

	fnv1a_32_hash_hash_score = test_distribution(fnv1a_32_hash)
	print("fnv1a_32_hash_hash chi2_benchmark score:", hash_path, ":", fnv1a_32_hash_hash_score, ":", bucket_sizes)

	crc32_hash_score = test_distribution(crc32_hash)
	print("crc32_hash chi2_benchmark score:", hash_path, ":", crc32_hash_score, ":", bucket_sizes)

	sha256_hash_score = test_distribution(sha256_hash)
	print("sha256_hash chi2_benchmark score:", hash_path, ":", sha256_hash_score, ":", bucket_sizes)

if __name__ == '__main__':
	assert vast_hash_impl(np.array([123, 123, 123, 123], dtype=np.uint64)).all() == np.array([6205865627071447306, 2067898264094941332, 1954899363002243930, 9928278621127670264], dtype=np.uint64).all()
	assert vast_hash(np.array([[123, 123, 123, 123], [123, 123, 123, 123]], dtype=np.uint64)) == 3420395603173502432

	chi2_benchmark(bucket_sizes=[11, 13, 17, 19], hash_path='conficker.txt')
	print()
	chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_path='BruteX_password.list')
	print()
	chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_path="rockyou-65.txt")
	print()
	chi2_benchmark(bucket_sizes=range(1, 101), hash_path='BruteX_password.list')

	# while True:
	# 	brute_search_hash_func()
