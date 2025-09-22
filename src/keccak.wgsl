/*
    MIT License
    Author: Fred Kyung-jin Rezeau <fred@litemint.com>, 2024
    Permission is granted to use, copy, modify, and distribute this software for any purpose
    with or without fee.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

    Description:
    WGSL WebGPU compute shader implementing int32-based Keccak-256 hashing for KALE mining.
    
    Performance Notes:
    - Averages 1.1 GH/s on Chrome using an NVIDIA GeForce RTX 4080 with the Direct3D backend on Windows 11.
    - Metal backend currently underperforms and requires further optimization and debugging.
*/

@group(0) @binding(0) var<storage, read> inputData: array<u32>;
@group(0) @binding(1) var<uniform> params: vec3<u32>;
@group(0) @binding(2) var<uniform> nonce: vec3<u32>;
@group(0) @binding(3) var<storage, read_write> outputHash: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputNonce: array<u32>;
@group(0) @binding(5) var<storage, read_write> found: atomic<u32>;

struct uint64 {
    low: u32,
    high: u32
};

struct slice {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
    s8: u32,
    s9: u32,
    s10: u32,
    s11: u32,
    s12: u32,
    s13: u32,
    s14: u32,
    s15: u32,
    s16: u32,
    s17: u32,
    s18: u32,
    s19: u32,
    s20: u32,
    s21: u32,
    s22: u32,
    s23: u32,
    s24: u32,
    s25: u32,
    s26: u32,
    s27: u32,
    s28: u32,
    s29: u32,
    s30: u32,
    s31: u32
};

fn uint64FromBytes(bytes: ptr<function, array<u32, 200>>, offset: u32) -> uint64 {
    return uint64(
        (*bytes)[offset] | ((*bytes)[offset + 1u] << 8u) | ((*bytes)[offset + 2u] << 16u) | ((*bytes)[offset + 3u] << 24u),
        (*bytes)[offset + 4u] | ((*bytes)[offset + 5u] << 8u) | ((*bytes)[offset + 6u] << 16u) | ((*bytes)[offset + 7u] << 24u)
    );
}

fn uint64ToBytes(val: uint64, bytes: ptr<function, array<u32, 200>>, offset: u32) {
    (*bytes)[offset] = val.low & 0xFFu;
    (*bytes)[offset + 1u] = (val.low >> 8u) & 0xFFu;
    (*bytes)[offset + 2u] = (val.low >> 16u) & 0xFFu;
    (*bytes)[offset + 3u] = (val.low >> 24u) & 0xFFu;
    (*bytes)[offset + 4u] = val.high & 0xFFu;
    (*bytes)[offset + 5u] = (val.high >> 8u) & 0xFFu;
    (*bytes)[offset + 6u] = (val.high >> 16u) & 0xFFu;
    (*bytes)[offset + 7u] = (val.high >> 24u) & 0xFFu;
}

fn uint64Xor(a: uint64, b: uint64) -> uint64 {
    return uint64(a.low ^ b.low, a.high ^ b.high);
}

fn uint64And(a: uint64, b: uint64) -> uint64 {
    return uint64(a.low & b.low, a.high & b.high);
}

fn uint64Not(a: uint64) -> uint64 {
    return uint64(~a.low, ~a.high);
}

fn uint64Rotl(a: uint64, n: u32) -> uint64 {
    let nmod: u32 = n % 64u;
    if nmod == 0u {
        return a;
    } else if nmod < 32u {
        return uint64((a.low << nmod) | (a.high >> (32u - nmod)),
            (a.high << nmod) | (a.low >> (32u - nmod)));
    } else if nmod == 32u {
        return uint64(a.high, a.low);
    } else {
        let shift: u32 = nmod - 32u;
        return uint64((a.high << shift) | (a.low >> (32u - shift)),
            (a.low << shift) | (a.high >> (32u - shift)));
    }
}

fn xorState(state: ptr<function, array<u32, 200>>, index: u32, x: u32) {
    (*state)[index] ^= x;
}

fn keccakF1600(state: ptr<function, array<u32, 200>>) {
    var s0: uint64 = uint64FromBytes(state, 0u * 8u);
    var s1: uint64 = uint64FromBytes(state, 1u * 8u);
    var s2: uint64 = uint64FromBytes(state, 2u * 8u);
    var s3: uint64 = uint64FromBytes(state, 3u * 8u);
    var s4: uint64 = uint64FromBytes(state, 4u * 8u);
    var s5: uint64 = uint64FromBytes(state, 5u * 8u);
    var s6: uint64 = uint64FromBytes(state, 6u * 8u);
    var s7: uint64 = uint64FromBytes(state, 7u * 8u);
    var s8: uint64 = uint64FromBytes(state, 8u * 8u);
    var s9: uint64 = uint64FromBytes(state, 9u * 8u);
    var s10: uint64 = uint64FromBytes(state, 10u * 8u);
    var s11: uint64 = uint64FromBytes(state, 11u * 8u);
    var s12: uint64 = uint64FromBytes(state, 12u * 8u);
    var s13: uint64 = uint64FromBytes(state, 13u * 8u);
    var s14: uint64 = uint64FromBytes(state, 14u * 8u);
    var s15: uint64 = uint64FromBytes(state, 15u * 8u);
    var s16: uint64 = uint64FromBytes(state, 16u * 8u);
    var s17: uint64 = uint64FromBytes(state, 17u * 8u);
    var s18: uint64 = uint64FromBytes(state, 18u * 8u);
    var s19: uint64 = uint64FromBytes(state, 19u * 8u);
    var s20: uint64 = uint64FromBytes(state, 20u * 8u);
    var s21: uint64 = uint64FromBytes(state, 21u * 8u);
    var s22: uint64 = uint64FromBytes(state, 22u * 8u);
    var s23: uint64 = uint64FromBytes(state, 23u * 8u);
    var s24: uint64 = uint64FromBytes(state, 24u * 8u);

    var C0: uint64; var C1: uint64; var C2: uint64; var C3: uint64; var C4: uint64;
    var D0: uint64; var D1: uint64; var D2: uint64; var D3: uint64; var D4: uint64;
    var B0: uint64; var B1: uint64; var B2: uint64; var B3: uint64; var B4: uint64;
    var B5: uint64; var B6: uint64; var B7: uint64; var B8: uint64; var B9: uint64;
    var B10: uint64; var B11: uint64; var B12: uint64; var B13: uint64; var B14: uint64;
    var B15: uint64; var B16: uint64; var B17: uint64; var B18: uint64; var B19: uint64;
    var B20: uint64; var B21: uint64; var B22: uint64; var B23: uint64; var B24: uint64;

    for (var round: u32 = 0u; round < 24u; round = round + 1u) {
        // θ step
        C0 = uint64Xor(uint64Xor(uint64Xor(uint64Xor(s0, s5), s10), s15), s20);
        C1 = uint64Xor(uint64Xor(uint64Xor(uint64Xor(s1, s6), s11), s16), s21);
        C2 = uint64Xor(uint64Xor(uint64Xor(uint64Xor(s2, s7), s12), s17), s22);
        C3 = uint64Xor(uint64Xor(uint64Xor(uint64Xor(s3, s8), s13), s18), s23);
        C4 = uint64Xor(uint64Xor(uint64Xor(uint64Xor(s4, s9), s14), s19), s24);

        D0 = uint64Xor(C4, uint64Rotl(C1, 1u));
        D1 = uint64Xor(C0, uint64Rotl(C2, 1u));
        D2 = uint64Xor(C1, uint64Rotl(C3, 1u));
        D3 = uint64Xor(C2, uint64Rotl(C4, 1u));
        D4 = uint64Xor(C3, uint64Rotl(C0, 1u));

        s0 = uint64Xor(s0, D0);
        s5 = uint64Xor(s5, D0);
        s10 = uint64Xor(s10, D0);
        s15 = uint64Xor(s15, D0);
        s20 = uint64Xor(s20, D0);

        s1 = uint64Xor(s1, D1);
        s6 = uint64Xor(s6, D1);
        s11 = uint64Xor(s11, D1);
        s16 = uint64Xor(s16, D1);
        s21 = uint64Xor(s21, D1);

        s2 = uint64Xor(s2, D2);
        s7 = uint64Xor(s7, D2);
        s12 = uint64Xor(s12, D2);
        s17 = uint64Xor(s17, D2);
        s22 = uint64Xor(s22, D2);

        s3 = uint64Xor(s3, D3);
        s8 = uint64Xor(s8, D3);
        s13 = uint64Xor(s13, D3);
        s18 = uint64Xor(s18, D3);
        s23 = uint64Xor(s23, D3);

        s4 = uint64Xor(s4, D4);
        s9 = uint64Xor(s9, D4);
        s14 = uint64Xor(s14, D4);
        s19 = uint64Xor(s19, D4);
        s24 = uint64Xor(s24, D4);

        // ρ and π steps
        B0 = s0;
        B1 = uint64Rotl(s6, 44u);
        B2 = uint64Rotl(s12, 43u);
        B3 = uint64Rotl(s18, 21u);
        B4 = uint64Rotl(s24, 14u);
        B5 = uint64Rotl(s3, 28u);
        B6 = uint64Rotl(s9, 20u);
        B7 = uint64Rotl(s10, 3u);
        B8 = uint64Rotl(s16, 45u);
        B9 = uint64Rotl(s22, 61u);
        B10 = uint64Rotl(s1, 1u);
        B11 = uint64Rotl(s7, 6u);
        B12 = uint64Rotl(s13, 25u);
        B13 = uint64Rotl(s19, 8u);
        B14 = uint64Rotl(s20, 18u);
        B15 = uint64Rotl(s4, 27u);
        B16 = uint64Rotl(s5, 36u);
        B17 = uint64Rotl(s11, 10u);
        B18 = uint64Rotl(s17, 15u);
        B19 = uint64Rotl(s23, 56u);
        B20 = uint64Rotl(s2, 62u);
        B21 = uint64Rotl(s8, 55u);
        B22 = uint64Rotl(s14, 39u);
        B23 = uint64Rotl(s15, 41u);
        B24 = uint64Rotl(s21, 2u);

        // χ step
        var t0: uint64;
        var t1: uint64;
        var t2: uint64;
        var t3: uint64;
        var t4: uint64;
        t0 = B0; t1 = B1; t2 = B2; t3 = B3; t4 = B4;
        s0 = uint64Xor(t0, uint64And(uint64Not(t1), t2));
        s1 = uint64Xor(t1, uint64And(uint64Not(t2), t3));
        s2 = uint64Xor(t2, uint64And(uint64Not(t3), t4));
        s3 = uint64Xor(t3, uint64And(uint64Not(t4), t0));
        s4 = uint64Xor(t4, uint64And(uint64Not(t0), t1));
        t0 = B5; t1 = B6; t2 = B7; t3 = B8; t4 = B9;
        s5 = uint64Xor(t0, uint64And(uint64Not(t1), t2));
        s6 = uint64Xor(t1, uint64And(uint64Not(t2), t3));
        s7 = uint64Xor(t2, uint64And(uint64Not(t3), t4));
        s8 = uint64Xor(t3, uint64And(uint64Not(t4), t0));
        s9 = uint64Xor(t4, uint64And(uint64Not(t0), t1));
        t0 = B10; t1 = B11; t2 = B12; t3 = B13; t4 = B14;
        s10 = uint64Xor(t0, uint64And(uint64Not(t1), t2));
        s11 = uint64Xor(t1, uint64And(uint64Not(t2), t3));
        s12 = uint64Xor(t2, uint64And(uint64Not(t3), t4));
        s13 = uint64Xor(t3, uint64And(uint64Not(t4), t0));
        s14 = uint64Xor(t4, uint64And(uint64Not(t0), t1));
        t0 = B15; t1 = B16; t2 = B17; t3 = B18; t4 = B19;
        s15 = uint64Xor(t0, uint64And(uint64Not(t1), t2));
        s16 = uint64Xor(t1, uint64And(uint64Not(t2), t3));
        s17 = uint64Xor(t2, uint64And(uint64Not(t3), t4));
        s18 = uint64Xor(t3, uint64And(uint64Not(t4), t0));
        s19 = uint64Xor(t4, uint64And(uint64Not(t0), t1));
        t0 = B20; t1 = B21; t2 = B22; t3 = B23; t4 = B24;
        s20 = uint64Xor(t0, uint64And(uint64Not(t1), t2));
        s21 = uint64Xor(t1, uint64And(uint64Not(t2), t3));
        s22 = uint64Xor(t2, uint64And(uint64Not(t3), t4));
        s23 = uint64Xor(t3, uint64And(uint64Not(t4), t0));
        s24 = uint64Xor(t4, uint64And(uint64Not(t0), t1));

        // ι step
        var roundConstant: uint64;
        switch (round) {
            case 0u: { roundConstant = uint64(0x00000001u, 0x00000000u); }
            case 1u: { roundConstant = uint64(0x00008082u, 0x00000000u); }
            case 2u: { roundConstant = uint64(0x0000808au, 0x80000000u); }
            case 3u: { roundConstant = uint64(0x80008000u, 0x80000000u); }
            case 4u: { roundConstant = uint64(0x0000808bu, 0x00000000u); }
            case 5u: { roundConstant = uint64(0x80000001u, 0x00000000u); }
            case 6u: { roundConstant = uint64(0x80008081u, 0x80000000u); }
            case 7u: { roundConstant = uint64(0x00008009u, 0x80000000u); }
            case 8u: { roundConstant = uint64(0x0000008au, 0x00000000u); }
            case 9u: { roundConstant = uint64(0x00000088u, 0x00000000u); }
            case 10u: { roundConstant = uint64(0x80008009u, 0x00000000u); }
            case 11u: { roundConstant = uint64(0x8000000au, 0x00000000u); }
            case 12u: { roundConstant = uint64(0x8000808bu, 0x00000000u); }
            case 13u: { roundConstant = uint64(0x0000008bu, 0x80000000u); }
            case 14u: { roundConstant = uint64(0x00008089u, 0x80000000u); }
            case 15u: { roundConstant = uint64(0x00008003u, 0x80000000u); }
            case 16u: { roundConstant = uint64(0x00008002u, 0x80000000u); }
            case 17u: { roundConstant = uint64(0x00000080u, 0x80000000u); }
            case 18u: { roundConstant = uint64(0x0000800au, 0x00000000u); }
            case 19u: { roundConstant = uint64(0x8000000au, 0x80000000u); }
            case 20u: { roundConstant = uint64(0x80008081u, 0x80000000u); }
            case 21u: { roundConstant = uint64(0x00008080u, 0x80000000u); }
            case 22u: { roundConstant = uint64(0x80000001u, 0x00000000u); }
            case 23u: { roundConstant = uint64(0x80008008u, 0x80000000u); }
            default: { roundConstant = uint64(0u, 0u); }
        }
        s0 = uint64Xor(s0, roundConstant);
    }

    uint64ToBytes(s0, state, 0u * 8u);
    uint64ToBytes(s1, state, 1u * 8u);
    uint64ToBytes(s2, state, 2u * 8u);
    uint64ToBytes(s3, state, 3u * 8u);
    uint64ToBytes(s4, state, 4u * 8u);
    uint64ToBytes(s5, state, 5u * 8u);
    uint64ToBytes(s6, state, 6u * 8u);
    uint64ToBytes(s7, state, 7u * 8u);
    uint64ToBytes(s8, state, 8u * 8u);
    uint64ToBytes(s9, state, 9u * 8u);
    uint64ToBytes(s10, state, 10u * 8u);
    uint64ToBytes(s11, state, 11u * 8u);
    uint64ToBytes(s12, state, 12u * 8u);
    uint64ToBytes(s13, state, 13u * 8u);
    uint64ToBytes(s14, state, 14u * 8u);
    uint64ToBytes(s15, state, 15u * 8u);
    uint64ToBytes(s16, state, 16u * 8u);
    uint64ToBytes(s17, state, 17u * 8u);
    uint64ToBytes(s18, state, 18u * 8u);
    uint64ToBytes(s19, state, 19u * 8u);
    uint64ToBytes(s20, state, 20u * 8u);
    uint64ToBytes(s21, state, 21u * 8u);
    uint64ToBytes(s22, state, 22u * 8u);
    uint64ToBytes(s23, state, 23u * 8u);
    uint64ToBytes(s24, state, 24u * 8u);
}

fn keccak256Update(state: ptr<function, array<u32, 200>>, data: ptr<function, array<u32, 256>>, length: u32) -> array<u32, 32u> {
    let rate: u32 = 136u;
    var len: u32 = length;
    var doff: u32 = 0u;
    var offset: u32 = 0u;
    while len > 0u {
        var chunk: u32 = len;
        if len > (rate - offset) {
            chunk = rate - offset;
        }
        let t: u32 = offset;
        for (var i: u32 = 0u; i < chunk; i = i + 1u) {
            xorState(state, t + i, (*data)[doff + i]);
        }
        offset += chunk;
        doff += chunk;
        len -= chunk;
        if offset == rate {
            keccakF1600(state);
            offset = 0u;
        }
    }

    xorState(state, offset, 0x01u);
    xorState(state, 135u, 0x80u);
    keccakF1600(state);

    var s: slice = slice(
        (*state)[0], (*state)[1], (*state)[2], (*state)[3],
        (*state)[4], (*state)[5], (*state)[6], (*state)[7],
        (*state)[8], (*state)[9], (*state)[10], (*state)[11],
        (*state)[12], (*state)[13], (*state)[14], (*state)[15],
        (*state)[16], (*state)[17], (*state)[18], (*state)[19],
        (*state)[20], (*state)[21], (*state)[22], (*state)[23],
        (*state)[24], (*state)[25], (*state)[26], (*state)[27],
        (*state)[28], (*state)[29], (*state)[30], (*state)[31]
    );
    var output: array<u32, 32u>;
    output[0] = s.s0; output[1] = s.s1; output[2] = s.s2;
    output[3] = s.s3; output[4] = s.s4; output[5] = s.s5;
    output[6] = s.s6; output[7] = s.s7; output[8] = s.s8;
    output[9] = s.s9; output[10] = s.s10; output[11] = s.s11;
    output[12] = s.s12; output[13] = s.s13; output[14] = s.s14;
    output[15] = s.s15; output[16] = s.s16; output[17] = s.s17;
    output[18] = s.s18; output[19] = s.s19; output[20] = s.s20;
    output[21] = s.s21; output[22] = s.s22; output[23] = s.s23;
    output[24] = s.s24; output[25] = s.s25; output[26] = s.s26;
    output[27] = s.s27; output[28] = s.s28; output[29] = s.s29;
    output[30] = s.s30; output[31] = s.s31;
    return output;
}

fn keccak256(data: ptr<function, array<u32, 256>>, length: u32) -> array<u32, 32u> {
    var state: array<u32, 200>;
    return keccak256Update(&state, data, length);
}

fn check(out: array<u32, 32>, diff: u32) -> bool {
    var zeros: u32 = 0u;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let byte: u32 = out[i] & 0xFFu;
        if byte == 0u {
            zeros = zeros + 2u;
        } else if (byte >> 4u) == 0u {
            zeros = zeros + 1u;
            break;
        } else {
            break;
        }
        if zeros >= diff {
            break;
        }
    }
    return zeros >= diff;
}

fn updateNonce(nonce: uint64, data: ptr<function, array<u32, 256>>, offset: u32) {
    (*data)[offset] = (nonce.high >> 24u) & 0xFFu;
    (*data)[offset + 1u] = (nonce.high >> 16u) & 0xFFu;
    (*data)[offset + 2u] = (nonce.high >> 8u) & 0xFFu;
    (*data)[offset + 3u] = nonce.high & 0xFFu;
    (*data)[offset + 4u] = (nonce.low >> 24u) & 0xFFu;
    (*data)[offset + 5u] = (nonce.low >> 16u) & 0xFFu;
    (*data)[offset + 6u] = (nonce.low >> 8u) & 0xFFu;
    (*data)[offset + 7u] = nonce.low & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let len = params[0];
    let batch = params[1];
    let diff = params[2];
    let nonceLow = nonce[0] + (gid.x * batch);
    let nonceHigh = nonce[1];
    let nonceOffset = nonce[2];
    var nonce = uint64(nonceLow, nonceHigh);
    var data: array<u32, 256u>;
    for (var i: u32 = 0u; i < len; i = i + 1u) {
        data[i] = inputData[i];
    }
    for (var i: u32 = 0u; i < batch; i = i + 1u) {
        updateNonce(nonce, &data, nonceOffset);
        let hash = keccak256(&data, len);
        if check(hash, diff) {
            if atomicLoad(&found) == 1u {
                return;
            }
            atomicStore(&found, 1u);
            for (var j: u32 = 0u; j < 32u; j = j + 1u) {
                outputHash[j] = hash[j];
            }
            outputNonce[0] = nonce.low;
            outputNonce[1] = nonce.high;
            return;
        }
        nonce = uint64(nonce.low + 1u, nonce.high);
    }
}
