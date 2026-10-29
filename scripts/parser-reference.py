"""Independent official ONNX output for the Rust/Triton encoder comparison."""
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import runpy
REVISION=runpy.run_path(str(Path(__file__).with_name('fetch-minilm.py')))['REVISION']

root=Path('.cache/minilm')
path=root/'reference.onnx'
if not path.exists():
    url=f'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/onnx/model.onnx'
    with urllib.request.urlopen(url) as response,path.open('wb') as output:
        while chunk:=response.read(1024*1024):output.write(chunk)
encoder=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
records=[]
for text in ['Alice called Bob.','Do not use Alice.','Where does the person who called Bob work?']:
    encoded=tokenizer.encode(text);ids=np.array([encoded.ids],dtype=np.int64)
    hidden=encoder.run(None,{'input_ids':ids,'attention_mask':np.ones_like(ids),'token_type_ids':np.zeros_like(ids)})[0]
    records.append({'text':text,'ids':encoded.ids,'offsets':encoded.offsets,'hidden':hidden.reshape(-1).tolist()})
(root/'reference.json').write_text(json.dumps(records)+'\n')
print('reference SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
