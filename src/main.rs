use base64::{Engine as _, engine::general_purpose};
use crossterm::event::{self, Event, KeyCode};
use rand::{Rng, rng};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc,
};
use std::time::{Duration, Instant};
use tiny_keccak::{Hasher, Keccak};

#[derive(Clone)]
struct MiningState {
    hash_rate: f64,
    block: u64,
    difficulty: usize,
    status: String,
    solution: Option<(u64, String)>,
    uptime: Duration,
}

enum MiningUpdate {
    HashRate(f64),
    Solution(u64, String),
    Status(String),
}

fn draw_ui(frame: &mut Frame, state: &MiningState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Length(5), // Mining info
            Constraint::Min(0),    // Results
        ])
        .split(frame.area());

    // Title
    let title = Paragraph::new("Carakale - Rust KALE Miner")
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL));
    frame.render_widget(title, chunks[0]);

    // Mining info
    let info_text = vec![
        Line::from(vec![
            Span::raw("Block: "),
            Span::styled(
                format!("{}", state.block),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Difficulty: "),
            Span::styled(
                format!("{} leading zeros", state.difficulty),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Hash Rate: "),
            Span::styled(
                format!("{:.2} MH/s", state.hash_rate / 1_000_000.0),
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ];
    let info = Paragraph::new(info_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Mining Status"),
    );
    frame.render_widget(info, chunks[1]);

    // Results
    let result_text = if let Some((nonce, hash)) = &state.solution {
        vec![
            Line::from(Span::styled(
                "🎉 SOLUTION FOUND!",
                Style::default().fg(Color::Green),
            )),
            Line::from(vec![
                Span::raw("Nonce: "),
                Span::styled(nonce.to_string(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("Hash: "),
                Span::styled(&hash[..32], Style::default().fg(Color::Blue)),
            ]),
        ]
    } else {
        vec![
            Line::from(vec![
                Span::raw("Status: "),
                Span::styled(&state.status, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::raw("Uptime: "),
                Span::styled(
                    format!("{}s", state.uptime.as_secs()),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            Line::from(Span::styled(
                "Press 'q' to quit",
                Style::default().fg(Color::Gray),
            )),
        ]
    };
    let results =
        Paragraph::new(result_text).block(Block::default().borders(Borders::ALL).title("Results"));
    frame.render_widget(results, chunks[2]);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize TUI
    let mut terminal = ratatui::init();

    // Create communication channel
    let (tx, rx) = mpsc::channel();

    // Initialize mining state
    let mut mining_state = MiningState {
        hash_rate: 0.0,
        block: 0,
        difficulty: 0,
        status: "Initializing...".to_string(),
        solution: None,
        uptime: Duration::from_secs(0),
    };

    // Generate parameters and start mining in background
    let (block, hash_b64, start_nonce, difficulty, miner_address) = generate_realistic_params();
    mining_state.block = block;
    mining_state.difficulty = difficulty;

    let tx_clone = tx.clone();
    std::thread::spawn(move || {
        tx_clone
            .send(MiningUpdate::Status("Mining started...".to_string()))
            .unwrap();

        if let Some((nonce, hash)) = mine_kale(
            block,
            &hash_b64,
            start_nonce,
            difficulty,
            &miner_address,
            Some(tx_clone.clone()),
        ) {
            tx_clone
                .send(MiningUpdate::Solution(nonce, hex::encode(hash)))
                .unwrap();
        }
    });

    let start_time = Instant::now();

    // Main UI loop
    loop {
        // Update uptime
        mining_state.uptime = start_time.elapsed();

        // Check for mining updates
        while let Ok(update) = rx.try_recv() {
            match update {
                MiningUpdate::HashRate(rate) => mining_state.hash_rate = rate,
                MiningUpdate::Solution(nonce, hash) => {
                    mining_state.solution = Some((nonce, hash));
                    mining_state.status = "Solution found!".to_string();
                }
                MiningUpdate::Status(status) => mining_state.status = status,
            }
        }

        // Draw UI
        terminal.draw(|frame| draw_ui(frame, &mining_state))?;

        // Handle input (non-blocking)
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    break;
                }
            }
        }
    }

    // Cleanup
    ratatui::restore();
    Ok(())
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
    tx: Option<mpsc::Sender<MiningUpdate>>,
) -> Option<(u64, [u8; 32])> {
    let start = Instant::now();

    // Decode the base64 hash
    let hash_bytes = match general_purpose::STANDARD.decode(hash_b64) {
        Ok(bytes) => bytes,
        Err(_) => {
            // println!("Failed to decode base64 hash");
            return None;
        }
    };

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

                // Send hash rate update every 100k hashes
                if nonce % 100_000 == 0 && tx.is_some() {
                    let elapsed = start.elapsed().as_secs_f64();
                    if elapsed > 0.1 {
                        // Avoid division by zero
                        let current_hashes = hash_counter.load(Ordering::Relaxed) as f64;
                        let hash_rate = current_hashes / elapsed;
                        let _ = tx.as_ref().unwrap().send(MiningUpdate::HashRate(hash_rate));
                    }
                }
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
    let difficulty = 4;

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
