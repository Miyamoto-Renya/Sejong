"""Persistent JSON-lines compiler service; Rust loads the generated HIP modules."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from sejong_kernels.ops import kernels, common
import importlib


def compile_request(request):
    operation=request['operation']
    if operation.startswith('parser_'):
        module=importlib.import_module('sejong_kernels.ops.parser')
    elif hasattr(common,operation): module=common
    else: module=kernels
    source_digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    identity = {**request, 'source': source_digest, 'bridge': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'compiler': importlib.metadata.version('triton')}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(os.environ.get('SEJONG_KERNEL_CACHE', '.cache/kernels')).resolve() / digest
    manifest = directory / 'kernel.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if Path(result['binary']).exists():
            return result
    fn = getattr(module, request['operation'])
    compiled = triton.compile(
        ASTSource(fn, signature=request['signature'], constexprs=request.get('constants', {})),
        target=GPUTarget('hip', request['architecture'], request['wavefront']),
        options={'num_warps': request.get('warps', 4), 'enable_fp_fusion': True},
    )
    metadata = compiled.metadata
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / 'kernel.hsaco'
    binary_temp = directory / f'kernel.{os.getpid()}.hsaco.tmp'
    binary_temp.write_bytes(compiled.asm['hsaco'])
    binary_temp.replace(binary)
    result = {'binary': str(binary), 'symbol': metadata.name, 'shared': metadata.shared,
              'threads': metadata.num_warps * metadata.warp_size,
              'global_scratch': metadata.global_scratch_size,
              'profile_scratch': metadata.profile_scratch_size}
    temp = directory / f'kernel.{os.getpid()}.tmp'
    temp.write_text(json.dumps(result, sort_keys=True))
    temp.replace(manifest)
    return result


def main():
    for line in sys.stdin:
        try:
            result = {'ok': compile_request(json.loads(line))}
        except Exception:
            result = {'error': traceback.format_exc()}
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
