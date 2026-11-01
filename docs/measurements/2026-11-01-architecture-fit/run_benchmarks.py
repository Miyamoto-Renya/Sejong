"""준비한 실제 Sejong 실행파일을 같은 학습 예산으로 순차 측정한다."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/renya/Sejong')
work = root / '.cache/fit-research'
plan = json.loads(Path(sys.argv[1]).read_text())
for case in plan:
    name = case['name']
    destination = root / 'runs' / ('fit-' + name)
    command = [case['executable'], 'benchmark', '--config', case['config'],
               '--data', 'docs/measurements/2026-11-01/train.txt',
               '--validation', 'docs/measurements/2026-11-01/validation.txt',
               '--out', str(destination), '--warmup-steps', str(case.get('warmup_steps', 2)),
               '--measure-steps', str(case.get('measure_steps', 64)), '--evaluation-batches', str(8 // case['batch']),
               '--decode-bytes', '32']
    env = os.environ.copy()
    env.update(case.get('env', {}))
    started = time.monotonic()
    print(json.dumps({'started': name, 'command': command}), flush=True)
    with (work / (name + '.log')).open('w') as log:
        run = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if run.returncode:
        print((work / (name + '.log')).read_text()[-6000:], flush=True)
        sys.exit(run.returncode)
    report = json.loads((destination / 'benchmark.json').read_text())['runs'][0]
    rows = report['measured_step_records']
    counters = rows[0]['counters']
    compact = {'name': name, 'wall_seconds': time.monotonic() - started,
               'step': report['training_step_seconds'],
               'compile_free': report['steps_without_compilation_seconds'],
               'bpb': report['final_validation_bits_per_byte_token'],
               'initial_bpb': report['initial_validation_bits_per_byte_token'],
               'counters_example': counters, 'execution': report['complete_execution'],
               'benchmark': str(destination / 'benchmark.json')}
    (work / (name + '-result.json')).write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps({k: v for k, v in compact.items() if k not in ['execution', 'counters_example']}), flush=True)
