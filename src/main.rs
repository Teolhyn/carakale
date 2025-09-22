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
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[cfg(not(feature = "gpu"))]
use {
    rayon::prelude::*,
    std::sync::atomic::{AtomicU64, Ordering},
    tiny_keccak::{Hasher, Keccak},
};

#[cfg(feature = "gpu")]
use {bytemuck, pollster, wgpu::util::DeviceExt};

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

        #[cfg(feature = "gpu")]
        let result = mine_kale_gpu(
            block,
            &hash_b64,
            start_nonce,
            difficulty,
            &miner_address,
            Some(tx_clone.clone()),
        );

        #[cfg(not(feature = "gpu"))]
        let result = mine_kale_cpu(
            block,
            &hash_b64,
            start_nonce,
            difficulty,
            &miner_address,
            Some(tx_clone.clone()),
        );

        if let Some((nonce, hash)) = result {
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

#[cfg(not(feature = "gpu"))]
fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    hasher.update(data);
    let mut output = [0u8; 32];
    hasher.finalize(&mut output);
    output
}

// CPU Mining Implementation
#[cfg(not(feature = "gpu"))]
fn mine_kale_cpu(
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

// GPU Mining Implementation
#[cfg(feature = "gpu")]
fn mine_kale_gpu(
    block: u64,
    hash_b64: &str,
    start_nonce: u64,
    difficulty: usize,
    miner_address: &str,
    tx: Option<mpsc::Sender<MiningUpdate>>,
) -> Option<(u64, [u8; 32])> {
    use bytemuck;

    let start = Instant::now();

    // Set up GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");

    if let Some(ref sender) = tx {
        let _ = sender.send(MiningUpdate::Status(format!(
            "GPU: {}",
            adapter.get_info().name
        )));
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    // Decode hash
    let hash_bytes = match general_purpose::STANDARD.decode(hash_b64) {
        Ok(bytes) => bytes,
        Err(_) => return None,
    };

    // Prepare input data to match original shader
    // The shader expects:
    // - binding 0: inputData: array<u32> (the KALE data)
    // - binding 1: params: vec3<u32> (len, batch, diff)
    // - binding 2: nonce: vec3<u32> (nonceLow, nonceHigh, nonceOffset)
    // - binding 3: outputHash: array<u32> (32 u32s for hash result)
    // - binding 4: outputNonce: array<u32> (2 u32s for nonce result)
    // - binding 5: found: atomic<u32>

    // Build KALE data format: block + hash + nonce_placeholder + miner_address
    let miner_bytes = miner_address.as_bytes();
    let block_bytes = block.to_be_bytes();

    // Create input data array (convert to u32 array for shader)
    let mut kale_data_bytes = Vec::new();
    kale_data_bytes.extend_from_slice(&block_bytes); // 8 bytes
    kale_data_bytes.extend_from_slice(&hash_bytes); // 32 bytes
    kale_data_bytes.extend_from_slice(&[0u8; 8]); // 8 bytes nonce placeholder
    kale_data_bytes.extend_from_slice(miner_bytes); // variable length

    // Convert to u32 array (the shader expects array<u32>)
    let mut input_data = Vec::new();
    for chunk in kale_data_bytes.chunks(4) {
        let mut bytes = [0u8; 4];
        for (i, &byte) in chunk.iter().enumerate() {
            bytes[i] = byte;
        }
        input_data.push(u32::from_le_bytes(bytes));
    }

    let data_len = kale_data_bytes.len() as u32;
    let nonce_offset = (8 + hash_bytes.len()) as u32; // Position where nonce goes in the data
    let batch_size = 256u32; // workgroup_size from shader

    // Create compute pipeline
    let shader_module = device.create_shader_module(wgpu::include_wgsl!("keccak.wgsl"));
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mining Pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create buffers to match shader bindings
    // Binding 0: inputData buffer
    let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Data"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Binding 3: outputHash buffer (32 u32s = 128 bytes)
    let output_hash_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Hash"),
        size: 32 * 4, // 32 u32s
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Binding 4: outputNonce buffer (2 u32s = 8 bytes)
    let output_nonce_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Nonce"),
        size: 2 * 4, // 2 u32s
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Binding 5: found buffer (1 atomic u32 = 4 bytes)
    let found_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Found"),
        size: 4, // 1 u32
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Download buffers for reading results
    let hash_download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hash Download"),
        size: output_hash_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let nonce_download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Nonce Download"),
        size: output_nonce_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let found_download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Found Download"),
        size: found_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut current_nonce = start_nonce;
    let mut total_hashes = 0u64;

    // Mining loop
    for iteration in 0..10_000 {
        // Clear found flag
        queue.write_buffer(&found_buffer, 0, bytemuck::cast_slice(&[0u32]));

        // Create uniform buffers for this iteration
        // Binding 1: params (len, batch, diff)
        let params_data = [data_len, batch_size, difficulty as u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Binding 2: nonce (nonceLow, nonceHigh, nonceOffset)
        let nonce_data = [
            current_nonce as u32,
            (current_nonce >> 32) as u32,
            nonce_offset,
        ];
        let nonce_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nonce"),
            contents: bytemuck::cast_slice(&nonce_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mining Bind Group"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nonce_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_hash_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_nonce_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: found_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mining Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mining Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1); // Single workgroup with 256 threads
        }

        // Copy results to download buffers
        encoder.copy_buffer_to_buffer(
            &found_buffer,
            0,
            &found_download_buffer,
            0,
            found_buffer.size(),
        );
        encoder.copy_buffer_to_buffer(
            &output_hash_buffer,
            0,
            &hash_download_buffer,
            0,
            output_hash_buffer.size(),
        );
        encoder.copy_buffer_to_buffer(
            &output_nonce_buffer,
            0,
            &nonce_download_buffer,
            0,
            output_nonce_buffer.size(),
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Check if solution was found
        let found_slice = found_download_buffer.slice(..);
        let (found_sender, found_receiver) = std::sync::mpsc::channel();
        found_slice.map_async(wgpu::MapMode::Read, move |result| {
            found_sender.send(result).unwrap();
        });

        let _ = device.poll(wgpu::PollType::Wait);

        if found_receiver.recv().unwrap().is_ok() {
            let found_data = found_slice.get_mapped_range();
            let found_value =
                u32::from_le_bytes([found_data[0], found_data[1], found_data[2], found_data[3]]);
            drop(found_data);
            found_download_buffer.unmap();

            if found_value != 0 {
                // Solution found! Read the hash and nonce
                let hash_slice = hash_download_buffer.slice(..);
                let (hash_sender, hash_receiver) = std::sync::mpsc::channel();
                hash_slice.map_async(wgpu::MapMode::Read, move |result| {
                    hash_sender.send(result).unwrap();
                });

                let nonce_slice = nonce_download_buffer.slice(..);
                let (nonce_sender, nonce_receiver) = std::sync::mpsc::channel();
                nonce_slice.map_async(wgpu::MapMode::Read, move |result| {
                    nonce_sender.send(result).unwrap();
                });

                let _ = device.poll(wgpu::PollType::Wait);

                if hash_receiver.recv().unwrap().is_ok() && nonce_receiver.recv().unwrap().is_ok() {
                    let hash_data = hash_slice.get_mapped_range();
                    let nonce_data = nonce_slice.get_mapped_range();

                    // Convert hash from u32 array to [u8; 32]
                    let mut hash_bytes = [0u8; 32];
                    for i in 0..32 {
                        hash_bytes[i] = hash_data[i];
                    }

                    // Read nonce (2 u32s)
                    let nonce_low = u32::from_le_bytes([
                        nonce_data[0],
                        nonce_data[1],
                        nonce_data[2],
                        nonce_data[3],
                    ]);
                    let nonce_high = u32::from_le_bytes([
                        nonce_data[4],
                        nonce_data[5],
                        nonce_data[6],
                        nonce_data[7],
                    ]);
                    let nonce = ((nonce_high as u64) << 32) | (nonce_low as u64);

                    drop(hash_data);
                    drop(nonce_data);
                    hash_download_buffer.unmap();
                    nonce_download_buffer.unmap();

                    return Some((nonce, hash_bytes));
                }
            }
        } else {
            found_download_buffer.unmap();
        }

        current_nonce += batch_size as u64;
        total_hashes += batch_size as u64;

        // Send hash rate update every 10 iterations
        if iteration % 10 == 0 && tx.is_some() {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.1 {
                let hash_rate = total_hashes as f64 / elapsed;
                let _ = tx.as_ref().unwrap().send(MiningUpdate::HashRate(hash_rate));
            }
        }
    }

    None
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
    let difficulty = 5;

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
