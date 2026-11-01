"""저장된 두 모델의 전체 가중치와 재개 상태를 CPU에서 대조한다."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def tensors(path):
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
    data = np.memmap(path, dtype='u1', mode='r')
    return header, data, 8 + length


def compare_weights(a_path, b_path):
    ah, ad, ao = tensors(a_path)
    bh, bd, bo = tensors(b_path)
    names = sorted(set(ah) - {'__metadata__'})
    if set(names) != set(bh) - {'__metadata__'}:
        raise ValueError('parameter names differ')
    rows = []
    total = changed = nonfinite = 0
    delta_square = reference_square = maximum = 0.0
    for name in names:
        a, b = ah[name], bh[name]
        if (a['dtype'], a['shape']) != (b['dtype'], b['shape']):
            raise ValueError(f'parameter layout differs: {name}')
        dtype = {'F32': '<f4', 'F16': '<f2', 'F64': '<f8'}[a['dtype']]
        av = np.frombuffer(ad, dtype=dtype, offset=ao+a['data_offsets'][0],
                           count=(a['data_offsets'][1]-a['data_offsets'][0]) // np.dtype(dtype).itemsize)
        bv = np.frombuffer(bd, dtype=dtype, offset=bo+b['data_offsets'][0], count=av.size)
        diff = bv.astype(np.float64) - av.astype(np.float64)
        count = int(np.count_nonzero(av != bv))
        max_abs = float(np.abs(diff).max(initial=0))
        d2 = float(np.dot(diff, diff))
        a64 = av.astype(np.float64)
        a2 = float(np.dot(a64, a64))
        nf = int(np.count_nonzero(~np.isfinite(bv)))
        rows.append({'parameter': name, 'elements': int(av.size), 'different': count,
                     'max_abs': max_abs, 'relative_l2': (d2/a2)**0.5 if a2 else None,
                     'nonfinite': nf})
        total += av.size
        changed += count
        nonfinite += nf
        maximum = max(maximum, max_abs)
        delta_square += d2
        reference_square += a2
    return {'parameters': int(total), 'different_elements': changed,
            'max_abs_weight_difference': maximum,
            'relative_l2': (delta_square/reference_square)**0.5,
            'nonfinite': nonfinite, 'per_parameter': rows}


def optimizer_state(a_path, b_path):
    ah, ad, ao = tensors(a_path)
    bh, bd, bo = tensors(b_path)
    result = {}
    for name in sorted((set(ah) | set(bh)) - {'__metadata__'}):
        if name not in ah or name not in bh:
            result[name] = False
            continue
        a, b = ah[name], bh[name]
        if (a['dtype'], a['shape']) != (b['dtype'], b['shape']):
            result[name] = False
            continue
        av = ad[ao+a['data_offsets'][0]:ao+a['data_offsets'][1]]
        bv = bd[bo+b['data_offsets'][0]:bo+b['data_offsets'][1]]
        result[name] = bool(np.array_equal(av, bv))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = compare_weights(a.reference/'weights.safetensors', a.candidate/'weights.safetensors')
    ma = json.loads((a.reference/'manifest.json').read_text())
    mb = json.loads((a.candidate/'manifest.json').read_text())
    result['reference'] = str(a.reference)
    result['candidate'] = str(a.candidate)
    result['step_equal'] = ma['step'] == mb['step']
    result['data_rng_equal'] = ma.get('rng_state') == mb.get('rng_state')
    if ma.get('optimizer') and mb.get('optimizer'):
        result['optimizer_tensors_equal'] = optimizer_state(a.reference/ma['optimizer'], a.candidate/mb['optimizer'])
    output = json.dumps(result, indent=2)
    if a.out:
        a.out.write_text(output+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'per_parameter'}, indent=2))
    else:
        print(output)


if __name__ == '__main__':
    main()
