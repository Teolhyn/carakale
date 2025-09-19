use base64::{Engine as _, engine::general_purpose};
use rand::{Rng, rng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tiny_keccak::{Hasher, Keccak};

fn main() {
    println!("CaraKale - Rust KALE Miner");

    let test_data = b"hello world!";
    let hash = keccak256(test_data);
    println!("Hash of 'hello world': {}", hex::encode(hash));

    // Generate realistic randomized KALE parameters
    let (block, hash_b64, start_nonce, difficulty, miner_address) = generate_realistic_params();

    println!("\n--- KALE Mining Simulation ---");
    println!("Block: {}", block);
    println!("Previous Hash: {}", hash_b64);
    println!("Difficulty: {} leading zero bytes", difficulty);
    println!("Miner: {}", miner_address);

    if let Some((nonce, hash)) =
        mine_kale(block, &hash_b64, start_nonce, difficulty, &miner_address)
    {
        println!("\n🎉 Found solution!");
        println!("Nonce: {}", nonce);
        println!("Hash: {}", hex::encode(hash));
    } else {
        println!("No solution found in range");
    }
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    output
}

fn mine_kale(
    block: u64,
    hash_b64: &str,
    start_nonce: u64,
    difficulty: usize,
    miner_address: &str,
) -> Option<(u64, [u8; 32])> {
    let start = Instant::now();

    // Decode the base64 hash
    let hash_bytes = match general_purpose::STANDARD.decode(hash_b64) {
        Ok(bytes) => bytes,
        Err(_) => {
            println!("Failed to decode base64 hash");
            return None;
        }
    };

    println!("Starting parallel mining...");

    // Pre-calculate fixed parts to avoid repeated allocations
    let block_bytes = block.to_be_bytes();
    let miner_bytes = miner_address.as_bytes();
    let total_len = 8 + hash_bytes.len() + 8 + miner_bytes.len();

    let hash_counter = AtomicU64::new(0);

    let result = (start_nonce..start_nonce + 10_000_000_000)
        .into_par_iter()
        .find_any(|&nonce| {
            // Count every 1000th hash to reduce contention but keep accuracy
            if nonce % 1000 == 0 {
                hash_counter.fetch_add(1000, Ordering::Relaxed);
            }

            // Use stack-allocated array instead of heap Vec
            let mut data = [0u8; 120]; // Max size: 8 + 32 + 8 + 70 (max Stellar address)
            let mut offset = 0;

            // Build the KALE data format: block + hash + nonce + miner_address
            data[offset..offset + 8].copy_from_slice(&block_bytes);
            offset += 8;
            data[offset..offset + hash_bytes.len()].copy_from_slice(&hash_bytes);
            offset += hash_bytes.len();
            data[offset..offset + 8].copy_from_slice(&nonce.to_be_bytes());
            offset += 8;
            data[offset..offset + miner_bytes.len()].copy_from_slice(miner_bytes);
            offset += miner_bytes.len();

            let hash = keccak256(&data[..offset]);
            hash.iter().take(difficulty).all(|&b| b == 0)
        });

    if let Some(nonce) = result {
        // Calculate final hash for return (reuse pre-calculated parts)
        let mut data = Vec::with_capacity(total_len);
        data.extend_from_slice(&block_bytes);
        data.extend_from_slice(&hash_bytes);
        data.extend_from_slice(&nonce.to_be_bytes());
        data.extend_from_slice(miner_bytes);
        let hash = keccak256(&data);

        let elapsed = start.elapsed().as_secs_f64();
        let actual_hashes = hash_counter.load(Ordering::Relaxed) as f64;
        let hash_rate = actual_hashes / elapsed;
        println!(
            "Hash rate: {:.2} MH/s (KALE format) - ~{} hashes in {:.2}s",
            hash_rate / 1_000_000.0,
            hash_counter.load(Ordering::Relaxed),
            elapsed
        );
        Some((nonce, hash))
    } else {
        let elapsed = start.elapsed().as_secs_f64();
        println!("Gave up after 10B attempts in {:.2}s", elapsed);
        None
    }
}

fn generate_realistic_params() -> (u64, String, u64, usize, String) {
    let mut rng = rng();

    // Realistic block number (KALE has been running for a while)
    let block = rng.random_range(50_000..200_000);

    // Generate realistic previous hash (32 random bytes, base64 encoded)
    let mut hash_bytes = [0u8; 32];
    rng.fill(&mut hash_bytes);
    let hash_b64 = general_purpose::STANDARD.encode(hash_bytes);

    // Random starting nonce
    let start_nonce = rng.random_range(0..1_000_000);

    // Realistic difficulty (KALE typically uses 2-4 leading zero bytes)
    let difficulty = rng.random_range(3..=4);

    // Generate realistic Stellar address
    let miner_address = generate_stellar_address();

    (block, hash_b64, start_nonce, difficulty, miner_address)
}

fn generate_stellar_address() -> String {
    let mut rng = rng();
    let mut address = String::from("G");

    // Stellar addresses are base32 encoded, so use A-Z and 2-7
    let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    for _ in 0..55 {
        let idx = rng.random_range(0..chars.len());
        address.push(chars.chars().nth(idx).unwrap());
    }

    address
}
