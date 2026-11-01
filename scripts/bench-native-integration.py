"""원본 학습 경로의 재현 가능한 개발용 비교 입력과 측정 결과를 관리한다."""
import argparse
import array
import copy
import copy
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(root):
    rng = random.Random(99271)
    d, tokens, n = 64, 32, 65
    contexts, queries = [], []
    for i in range(n):
        request = [float(i == j) for j in range(d)]
        queries.append(request)
        contexts.append(request + [rng.gauss(0, 1/math.sqrt(d)) for _ in range((tokens-1)*d)])
    targets = []
    for _ in range(8):
        v = [rng.gauss(0, 1) for _ in range(d)]
        norm = math.sqrt(sum(x*x for x in v))
        targets.append([x/norm for x in v])
    memory = dict(config=dict(id='sejong-memory-integration-benchmark', dimension=d,
        tokens=tokens, layers=2, heads=4, hidden=256, ffn_width=256,
        norm_epsilon=2**-23, readout_temperature=1., residual_scale=.5, seed=99277),
        requests=[f'request-{i}' for i in range(n)], values=[f'value-{i}' for i in range(8)],
        contexts=contexts, target_queries=queries, value_targets=targets,
        labels=[i % 8 for i in range(n)], query_steps=4, fact_steps=16,
        query_learning_rate=.002, min_query_learning_rate=.00002, learning_rate=.001,
        checkpoint_every=8, query_loss={}, background={'kind':'empirical'},
        optimizer={}, batch={'max_examples':16})
    save(root/'specs/memory.json', memory)
    save(root/'specs/revision.json', dict(labels=[(i+1)%8 for i in range(n)],
        steps=16, learning_rate=.001, checkpoint_every=8, optimizer={}, batch={'max_examples':16}))
    source = ROOT/'docs/measurements/2026-11-01-parameter-vjp/native-b.json'
    value = json.loads(source.read_text())
    text = (ROOT/'docs/measurements/2026-11-01/train.txt').read_text()
    # 기존 실험 텍스트와 요청으로 길이 510바이트의 입력 workload를 구성한다.
    encoded = (text*8).encode()
    prompts = []
    for i, prompt in enumerate(value['prompts']):
        prefix = (prompt+' ').encode()
        tail = encoded[i:i+510-len(prefix)]
        prompts.append({'hex':(prefix+tail).hex()})
    value.update(prompts=prompts, steps=16, checkpoint_every=8,
                 batch={'max_examples':8}, resume=None)
    save(root/'specs/value.json', value)
    scale = copy.deepcopy(value)
    scale['model'].update(id='sejong-native-scale-benchmark', width=1024,
                          layers=13, heads=16, kv_heads=8, ffn_width=3072)
    scale['model']['experts'].update(count=8, top_k=2, ffn_width=768)
    for prefix in ('encoder', 'decoder'):
        scale['model']['hierarchy'].update({prefix+'_width':256, prefix+'_layers':2,
                                           prefix+'_heads':4, prefix+'_ffn_width':768})
    for prompt in scale['prompts']:
        prompt['hex'] = (bytes.fromhex(prompt['hex'])*3)[:1022].hex()
    scale.update(optimizer_kind='sgd', steps=3, checkpoint_every=0)
    save(root/'specs/value-scale.json', scale)
    save(root/'specs/provenance.json', dict(purpose='development regression and timing workload',
        memory_source='existing create-native-data.py construction, explicit benchmark geometry',
        native_source=str(source.relative_to(ROOT)),
        input_text='docs/measurements/2026-11-01/train.txt',
        native_sequence_bytes=512, memory_requests=n))
    # Query 3-view 한 예제와 fact 한 예제의 workspace에서 파일 캐시 경계를 정한다.
    query_work = 33*(d*4) + 6*(256*4) + 2*(4*4) + 2*(d*4) + 4
    fact_work = 4*(256*4) + 3*(d*4)
    budget = 6*query_work + fact_work
    file_rows = (1 << (budget.bit_length()-1))//(d*4) + 1
    file_memory = copy.deepcopy(memory)
    file_memory['config'].update(id='sejong-memory-file-cache-benchmark', tokens=1)
    file_memory.update(requests=[f'request-{i}' for i in range(file_rows)],
        contexts=[[float(i%d==j) if i!=d else 0. for j in range(d)] for i in range(file_rows)],
        labels=[i%8 for i in range(file_rows)], query_steps=2, fact_steps=4,
        checkpoint_every=2, batch={'max_examples':1,'memory_bytes':budget})
    file_memory['target_queries'] = copy.deepcopy(file_memory['contexts'])
    save(root/'specs/memory-file.json', file_memory)
    for source, field in [('memory','fact_steps'),('revision','steps'),('value','steps')]:
        long = json.loads((root/'specs'/f'{source}.json').read_text())
        long[field] = 64 if source == 'value' else 128
        long['checkpoint_every'] = 0
        save(root/'specs'/f'{source}-long.json', long)
    checkpoint = copy.deepcopy(value)
    checkpoint['activation_checkpointing'] = True
    save(root/'specs/value-checkpoint.json', checkpoint)
    sgd = copy.deepcopy(value)
    sgd.update(optimizer_kind='sgd', activation_checkpointing=False)
    sgd['optimizer'] = {k:v for k,v in sgd['optimizer'].items() if k in ('weight_decay','clip_norm')}
    save(root/'specs/value-sgd.json', sgd)


def tensors(path):
    data = path.read_bytes()
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8+size])
    for name, item in header.items():
        if name == '__metadata__':
            continue
        low, high = item['data_offsets']
        dtype = {'F32':'f', 'F64':'d'}[item['dtype']]
        values = array.array(dtype)
        values.frombytes(data[8+size+low:8+size+high])
        if sys.byteorder != 'little':
            values.byteswap()
        yield name, item['shape'], values


def compare(left, right):
    result = {}
    for source in sorted((left/'model').rglob('*.safetensors')):
        relative = source.relative_to(left/'model')
        other = right/'model'/relative
        a, b = dict((n,(s,v)) for n,s,v in tensors(source)), dict((n,(s,v)) for n,s,v in tensors(other))
        squared, baseline, maximum, count = 0., 0., 0., 0
        for name, (shape, av) in a.items():
            bs, bv = b[name]
            if shape != bs:
                raise ValueError((name, shape, bs))
            for x,y in zip(av,bv):
                delta = x-y
                squared += delta*delta
                baseline += x*x
                maximum = max(maximum, abs(delta))
                count += 1
        result[str(relative)] = dict(elements=count, max_abs=maximum,
            relative_l2=math.sqrt(squared/baseline) if baseline else math.sqrt(squared))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','compare'])
    p.add_argument('--root', type=Path, default=ROOT/'.cache/native-integration')
    p.add_argument('--executable', type=Path)
    p.add_argument('--tag')
    p.add_argument('--mode', choices=['memory','value','revision','query'])
    p.add_argument('--spec', type=Path)
    p.add_argument('--original', type=Path)
    p.add_argument('--left', type=Path)
    p.add_argument('--right', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'compare':
        print(json.dumps(compare(args.left,args.right), indent=2))
    else:
        out = args.root/args.tag/args.mode
        command = [str(args.executable.resolve()), args.mode,
                   str((args.spec or args.root/'specs'/f'{args.mode}.json').resolve()), str(out.resolve())]
        if args.original:
            command.append(str(args.original.resolve()))
        environment = dict(os.environ, SEJONG_ROOT=str(ROOT), PYTHONPATH=str(ROOT/'python'),
                           SEJONG_PYTHON=str(ROOT/'.venv/bin/python'))
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent/f'{args.mode}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        print((out/'benchmark.json').read_text())


if __name__ == '__main__':
    main()
