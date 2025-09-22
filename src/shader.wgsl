// Simplified Keccak-256 GPU compute shader for KALE mining
// Using 32-bit arithmetic due to WGSL limitations

const KECCAK_ROUNDS: u32 = 24u;

// Keccak round constants (split into high/low 32-bit parts)
const RC_LOW: array<u32, 24> = array<u32, 24>(
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
);

const RC_HIGH: array<u32, 24> = array<u32, 24>(
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
);

// Rotation offsets for rho step
const RHO: array<u32, 24> = array<u32, 24>(
    1u, 3u, 6u, 10u, 15u, 21u, 28u, 36u, 45u, 55u, 2u, 14u,
    27u, 41u, 56u, 8u, 25u, 43u, 62u, 18u, 39u, 61u, 20u, 44u
);

// Pi step permutation indices
const PI: array<u32, 24> = array<u32, 24>(
    10u, 7u, 11u, 17u, 18u, 3u, 5u, 16u, 8u, 21u, 24u, 4u,
    15u, 23u, 19u, 13u, 12u, 2u, 20u, 14u, 22u, 9u, 6u, 1u
);

struct MiningInput {
    block_low: u32,
    block_high: u32,
    hash: array<u32, 8>,     // 32 bytes as 8 u32s (low/high pairs)
    nonce_start_low: u32,
    nonce_start_high: u32,
    difficulty: u32,
    batch_size: u32,
    miner_len: u32,          // Length of miner address in bytes
    miner: array<u32, 16>,   // Up to 64 bytes for miner address
}

struct MiningResult {
    found: u32,
    nonce_low: u32,
    nonce_high: u32,
    result_hash: array<u32, 8>, // 32 bytes as 8 u32s
}

@group(0) @binding(0) var<storage, read> input: MiningInput;
@group(0) @binding(1) var<storage, read_write> results: array<MiningResult>;

// 32-bit pair structure for 64-bit emulation
struct u64_pair {
    low: u32,
    high: u32,
}

// 64-bit left rotation using 32-bit pairs
fn rotl64_32(x_low: u32, x_high: u32, n: u32) -> u64_pair {
    var result: u64_pair;
    if (n == 0u) {
        result.low = x_low;
        result.high = x_high;
    } else if (n < 32u) {
        result.low = (x_low << n) | (x_high >> (32u - n));
        result.high = (x_high << n) | (x_low >> (32u - n));
    } else {
        let n_mod = n - 32u;
        result.low = (x_high << n_mod) | (x_low >> (32u - n_mod));
        result.high = (x_low << n_mod) | (x_high >> (32u - n_mod));
    }
    return result;
}

// XOR two 64-bit pairs
fn xor64_32(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> u64_pair {
    var result: u64_pair;
    result.low = a_low ^ b_low;
    result.high = a_high ^ b_high;
    return result;
}

// Keccak-f[1600] permutation using 32-bit arithmetic
fn keccak_f_32(state: ptr<function, array<u32, 50>>) {
    for (var round: u32 = 0u; round < KECCAK_ROUNDS; round++) {
        // Theta step
        var C: array<u32, 10>; // 5 pairs of low/high
        for (var x: u32 = 0u; x < 5u; x++) {
            let idx = x * 2u;
            C[idx] = (*state)[idx] ^ (*state)[idx + 10u] ^ (*state)[idx + 20u] ^ (*state)[idx + 30u] ^ (*state)[idx + 40u];
            C[idx + 1u] = (*state)[idx + 1u] ^ (*state)[idx + 11u] ^ (*state)[idx + 21u] ^ (*state)[idx + 31u] ^ (*state)[idx + 41u];
        }

        var D: array<u32, 10>; // 5 pairs of low/high
        for (var x: u32 = 0u; x < 5u; x++) {
            let prev_idx = ((x + 4u) % 5u) * 2u;
            let next_idx = ((x + 1u) % 5u) * 2u;
            let rot = rotl64_32(C[next_idx], C[next_idx + 1u], 1u);
            D[x * 2u] = C[prev_idx] ^ rot.low;
            D[x * 2u + 1u] = C[prev_idx + 1u] ^ rot.high;
        }

        for (var i: u32 = 0u; i < 25u; i++) {
            let d_idx = (i % 5u) * 2u;
            (*state)[i * 2u] ^= D[d_idx];
            (*state)[i * 2u + 1u] ^= D[d_idx + 1u];
        }

        // Rho and Pi steps
        var current_low = (*state)[2]; // state[1] low
        var current_high = (*state)[3]; // state[1] high

        for (var t: u32 = 0u; t < 24u; t++) {
            let next_index = PI[t];
            let next_idx = next_index * 2u;
            let temp_low = (*state)[next_idx];
            let temp_high = (*state)[next_idx + 1u];

            let rot = rotl64_32(current_low, current_high, RHO[t]);
            (*state)[next_idx] = rot.low;
            (*state)[next_idx + 1u] = rot.high;

            current_low = temp_low;
            current_high = temp_high;
        }

        // Chi step
        for (var y: u32 = 0u; y < 5u; y++) {
            var temp: array<u32, 10>; // 5 pairs
            for (var x: u32 = 0u; x < 5u; x++) {
                let idx = (y * 5u + x) * 2u;
                temp[x * 2u] = (*state)[idx];
                temp[x * 2u + 1u] = (*state)[idx + 1u];
            }
            for (var x: u32 = 0u; x < 5u; x++) {
                let idx = (y * 5u + x) * 2u;
                let next_x = (x + 1u) % 5u;
                let next2_x = (x + 2u) % 5u;
                (*state)[idx] = temp[x * 2u] ^ ((~temp[next_x * 2u]) & temp[next2_x * 2u]);
                (*state)[idx + 1u] = temp[x * 2u + 1u] ^ ((~temp[next_x * 2u + 1u]) & temp[next2_x * 2u + 1u]);
            }
        }

        // Iota step
        (*state)[0] ^= RC_LOW[round];
        (*state)[1] ^= RC_HIGH[round];
    }
}

// Keccak-256 hash function using 32-bit arithmetic
fn keccak256_32(data: ptr<function, array<u32, 32>>, data_len: u32) -> array<u32, 8> {
    var state: array<u32, 50>; // 25 pairs of low/high

    // Initialize state to zero
    for (var i: u32 = 0u; i < 50u; i++) {
        state[i] = 0u;
    }

    // Absorb phase (rate = 136 bytes for Keccak-256)
    let rate_bytes = 136u;
    var offset = 0u;

    while (offset < data_len) {
        let block_size = min(rate_bytes, data_len - offset);

        // XOR data into state (process 8 bytes = 2 u32s at a time)
        for (var i: u32 = 0u; i < block_size; i += 8u) {
            let word_idx = (i / 8u) * 2u;
            if (i + 7u < block_size) {
                let data_idx = (offset + i) / 4u;
                state[word_idx] ^= (*data)[data_idx];
                state[word_idx + 1u] ^= (*data)[data_idx + 1u];
            } else {
                // Handle partial last block
                let data_idx = (offset + i) / 4u;
                if (data_idx < 32u) {
                    state[word_idx] ^= (*data)[data_idx];
                }
                if (data_idx + 1u < 32u) {
                    state[word_idx + 1u] ^= (*data)[data_idx + 1u];
                }
            }
        }

        // Apply padding if this is the last block
        if (offset + block_size >= data_len) {
            // Keccak padding: append 0x01, then 0x80 at the end of rate
            let pad_offset = block_size % 8u;
            let pad_word = (block_size / 8u) * 2u;

            if (pad_word < 34u) {
                if (pad_offset < 4u) {
                    state[pad_word] ^= 0x01u << (pad_offset * 8u);
                } else {
                    state[pad_word + 1u] ^= 0x01u << ((pad_offset - 4u) * 8u);
                }
                state[33] ^= 0x80000000u; // Set bit at rate boundary (high word of state[16])
            }
        }

        keccak_f_32(&state);
        offset += block_size;
    }

    // Extract first 256 bits (32 bytes = 8 u32s)
    var result: array<u32, 8>;
    for (var i: u32 = 0u; i < 8u; i++) {
        result[i] = state[i];
    }

    return result;
}

// Check if hash meets difficulty requirement using 32-bit pairs
fn check_difficulty_32(hash: array<u32, 8>, difficulty: u32) -> bool {
    let bytes_to_check = difficulty;

    for (var i: u32 = 0u; i < min(bytes_to_check, 32u); i++) {
        let word_idx = i / 4u;
        let byte_idx = i % 4u;
        let byte_val = (hash[word_idx] >> (byte_idx * 8u)) & 0xFFu;

        if (byte_val != 0u) {
            return false;
        }
    }

    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    if (thread_id >= input.batch_size) {
        return;
    }

    // Calculate 64-bit nonce using 32-bit arithmetic
    var nonce_low = input.nonce_start_low + thread_id;
    var nonce_high = input.nonce_start_high;
    if (nonce_low < input.nonce_start_low) { // Overflow check
        nonce_high += 1u;
    }

    // Build KALE mining data: block + hash + nonce + miner_address
    var data: array<u32, 32>; // 128 bytes = 32 u32s
    var offset = 0u;

    // Add block (8 bytes = 2 u32s, big-endian)
    data[0] = input.block_high; // Big-endian: high word first
    data[1] = input.block_low;
    offset += 8u;

    // Add previous hash (32 bytes = 8 u32s)
    for (var i: u32 = 0u; i < 8u; i++) {
        data[2u + i] = input.hash[i];
    }
    offset += 32u;

    // Add nonce (8 bytes = 2 u32s, big-endian)
    data[10] = nonce_high; // Big-endian: high word first
    data[11] = nonce_low;
    offset += 8u;

    // Add miner address (up to 64 bytes = 16 u32s)
    let miner_words_to_copy = min((input.miner_len + 3u) / 4u, 16u);
    for (var i: u32 = 0u; i < miner_words_to_copy; i++) {
        data[12u + i] = input.miner[i];
    }
    offset += min(input.miner_len, 64u);

    // Calculate Keccak-256 hash
    let hash_result = keccak256_32(&data, offset);

    // Store result
    results[thread_id].nonce_low = nonce_low;
    results[thread_id].nonce_high = nonce_high;
    results[thread_id].result_hash = hash_result;

    // Check difficulty
    if (check_difficulty_32(hash_result, input.difficulty)) {
        results[thread_id].found = 1u;
    } else {
        results[thread_id].found = 0u;
    }
}
