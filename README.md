# ✨ VastHash
Previously named `SXAPH-256 - Shift XOR Add Prime Hash 256-Bit`.

VastHash is a non-cryptographic experimental hash function designed for prime-sized HashMap to minimize memory accesses and maximize the performance on AVX2 systems while having distribution comparable to multiplication-based hash.

VastHash is trained on passwords for prime-sized HashMap. So, it'll perform much worse on non-prime-sized HashMap.

It's made from brute-forcing every combination of operators `|, &, <<, >>, ^, and +`. I found claims online, that `| and &` reduces hash distribution and `~` doesn't improve. But, I wasn't able to get consistently reduced hash distribution result in these claims. Still, they're not our best performers on the list. The Rust compiler will replace `-` (vpsubq) with `+` (vpaddq) so that's excluded from the trainer.

`hash_256_a`, and `hash_256_b` from VastHash are trained with SymPy PRNG (sympy.randprime), then run through the Chi² test with the password 500 times, and filter out the best result. Yet, constants were manually picked due to how the hash function wasn't able to generalize outside its test.

The Chi² test is vulnerable to guided optimization framework like Bayesian optimization from [Optuna](https://optuna.org/), but multiple bucket sizes are used to improve the hash's ability to generalize over different sized array; it's done on all system cores to ensure maximum core utilization.

Previously, VastHash was brute-force optimized with [Optuna](https://optuna.org/), but it was using only half the performance of each core, and, Optuna wasn't more effective than PRNGs on `<<, >>, ^, +` optimizations for its overhead. Now, its constants are brute-forced optimized on SymPy PRNG.

Current constants are found by SymPy, then converted to Rust code, and reverse engineered from Rust compiler's x86 ASM output.

# 🌟 Features
- Hash 8% faster than a SIMD multiply (`mul_hash`)
- Outperforms hash distribution of `djb2`; SHA-256 when on prime-sized array
- Reduced data race (Doesn't rely on the same previous state except when summing)

# ⛈️ Deal-breakers
- Only works with 256-bit SIMD (AVX2)
- Gray-box hash function (Developed through trial and error)
- Extreme bug-prone when implementing as it's sensitive to typos
- Ineffective with non-prime sized array (But more uniform than `mul_hash` and `djb2`)
- Weak diffusion
- Trivial to perform collision attacks
- Non-innovative design (Due to restrictive AVX2 instruction set and faster than 1 multiply requirement)

# 🐍 Original code (Python)
```py
import numpy as np

def vast_hash_impl(input_data, hash_256=np.array([6205865627071447409, 2067898264094941423, 1954899363002243873, 9928278621127670147], dtype=np.uint64)):
	return np.bitwise_xor(input_data, hash_256)

assert vast_hash_impl(np.array([123, 123, 123, 123], dtype=np.uint64)).all() == np.array([6205865627071447306, 2067898264094941332, 1954899363002243930, 9928278621127670264], dtype=np.uint64).all()
```

## 🔩 [Rust port](https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,selection:(endColumn:1,endLineNumber:14,positionColumn:1,positionLineNumber:14,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:'%23!!%5Bfeature(portable_simd)%5D%0Ause+std::simd::u64x4%3B%0A%0A%23%5Bno_mangle%5D%0Apub+fn+vast_hash_impl(input_data:+u64x4)+-%3E+u64x4+%7B%0A%09input_data+%5E+u64x4::from_array(%5B6205865627071447409,+2067898264094941423,+1954899363002243873,+9928278621127670147%5D)%0A%7D%0A%0Afn+main()+%7B%0A%09let+input_data+%3D+u64x4::splat(123)%3B%0A%09let+result+%3D+vast_hash_impl(input_data)%3B%0A%09assert_eq!!(result,+u64x4::from_array(%5B6205865627071447306,+2067898264094941332,+1954899363002243930,+9928278621127670264%5D))%3B%0A%7D%0A'),l:'5',n:'1',o:'Rust+source+%231',t:'0')),k:46.705949673462854,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:nightly,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'0',intel:'0',libraryCode:'1',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,libs:!(),options:'-C+opt-level%3D2+-C+target-feature%3D%2Bavx2',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+rustc+nightly+(Editor+%231)',t:'0')),k:28.298696015337942,l:'4',m:100,n:'0',o:'',s:0,t:'0'),(g:!((h:output,i:(compilerName:'x86-64+gcc+14.1',editorid:1,fontScale:14,fontUsePx:'0',j:1,wrap:'1'),l:'5',n:'0',o:'Output+of+rustc+nightly+(Compiler+%231)',t:'0')),header:(),k:24.995354311199208,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4)
```rs
#![feature(portable_simd)]
use std::simd::u64x4;

#[no_mangle]
pub fn vast_hash_impl(input_data: u64x4) -> u64x4 {
	input_data ^ u64x4::from_array([6205865627071447409, 2067898264094941423, 1954899363002243873, 9928278621127670147])
}

fn main() {
	let input_data = u64x4::splat(123);
	let result = vast_hash_impl(input_data);
	assert_eq!(result, u64x4::from_array([6205865627071447306, 2067898264094941332, 1954899363002243930, 9928278621127670264]));
}
```
## 🍪 ASM output from Rust port
```asm
.LCPI0_0:
		.quad   6205865627071447409
		.quad   2067898264094941423
		.quad   1954899363002243873
		.quad   -8518465452581881469
vast_hash_impl:
		mov     rax, rdi
		vmovaps ymm0, ymmword ptr [rsi]
		vxorps  ymm0, ymm0, ymmword ptr [rip + .LCPI0_0]
		vmovaps ymmword ptr [rdi], ymm0
		vzeroupper
		ret
```

# ⏩🚀 Hashing Performance
This is tested with i5-9300H CPU. The speed will be different from different CPUs.
## 🥟 8-bit (small) hash
| CPU      | Bench name    | Time      |
|----------|---------------|-----------|
| i5-9300H | mul_hash_8    | 1.3250 ns |
|          | djb2_hash_8   | 1.7282 ns |
|          | vast_hash_8   | 1.2172 ns |
|          | fnv_1a_hash_8 | 1.9730 ns |
## 🍔 256-bit (standard) hash
| CPU      | Bench name      | Time      |
|----------|-----------------|-----------|
| i5-9300H | mul_hash_256    | 1.3226 ns |
|          | djb2_hash_256   | 6.6162 ns |
|          | vast_hash_256   | 1.2161 ns |
|          | fnv_1a_hash_256 | 6.7366 ns |

Running command:
```py
cargo bench
```

# 🚄🔥 Chi² benchmark (Lower gives better distribution)
`test_hash_distribution()` tests the distribution of hash values generated by a given function for a set of input data. It calculates the chi-squared statistic to measure the deviation of the observed hash distribution from the expected uniform distribution.
```py
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
```
### Hash alignment implementations
```py
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
```

## 🧮 Prime benchmark
ℹ️ Disclosure: `vast_hash` were trained on the [BruteX_password.list](https://weakpass.com/wordlist/1902) and [11, 13, 17, 19] bucket sizes.  

### Bucket sizes = `[11, 13, 17, 19]`  
Hash file = [conficker.txt](https://weakpass.com/wordlist/60)  

| Hash name       | Non-uniform score |
|-----------------|-------------------|
| mul_hash        | 1,237             |
| djb2_hash       | 8,207             |
| vast_hash       | 779               |
| fnv1a_32_hash   | 683               |
| crc32_hash      | 695               |
| sha256_hash     | 793               |

Running command:
```py
chi2_benchmark(bucket_sizes=[11, 13, 17, 19], hash_path='conficker.txt')
```

### Bucket sizes = `[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]`  
Hash file = [BruteX_password.list](https://weakpass.com/wordlist/1902)  

| Hash name       | Non-uniform score |
|-----------------|-------------------|
| mul_hash        | 20,222,418        |
| djb2_hash       | 8,870,738         |
| vast_hash       | 70,326            |
| fnv1a_32_hash   | 86,620            |
| crc32_hash      | 93,790            |
| sha256_hash     | 79,962            |

```py
chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_path='BruteX_password.list')
```

### Bucket sizes = `[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]`  
Hash file = [rockyou-65.txt](https://weakpass.com/wordlist/87)  

| Hash name       | Non-uniform score |
|-----------------|-------------------|
| mul_hash        | 986,582,602       |
| djb2_hash       | 349,747,618       |
| vast_hash       | 732,138           |
| fnv1a_32_hash   | 923,010           |
| crc32_hash      | 739,366           |
| sha256_hash     | 771,564           |

Running command:
```py
chi2_benchmark(bucket_sizes=[i for i in range(1, 101) if is_prime(i)], hash_path="rockyou-65.txt")
```

## 1️⃣4️⃣6️⃣8️⃣9️⃣🔟 Non-prime benchmark
### Bucket sizes = `range(1, 101)`  
Hash file = [BruteX_password.list](https://weakpass.com/wordlist/1902)  
| Hash name       | Non-uniform score |
|-----------------|-------------------|
| mul_hash        | 144,201,418       |
| djb2_hash       | 57,182,530        |
| vast_hash       | 1,956,106         |
| fnv1a_32_hash   | 377,188           |
| crc32_hash      | 399,244           |
| sha256_hash     | 316,864           |

Running command:
```py
chi2_benchmark(bucket_sizes=range(1, 101), hash_path='BruteX_password.list')
```
