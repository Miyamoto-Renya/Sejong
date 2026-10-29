"""Fetch the declared frozen encoder from its official, pinned model revision."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
EXPECTED={'config.json': '953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41', 'model.safetensors': '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db', 'tokenizer.json': 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037', 'tokenizer_config.json': 'acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b', 'special_tokens_map.json': '303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3', 'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'}
FILES=('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('.cache/minilm'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    records={}
    for name in FILES:
        destination=args.out/name
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()!=EXPECTED[name]:
            temporary=args.out/(name+'.download')
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
            with urllib.request.urlopen(url) as response,temporary.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
            temporary.replace(destination)
        records[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
        if records[name]!=EXPECTED[name]:raise ValueError(f'Asset digest differs: {name}')
    (args.out/'source.json').write_text(json.dumps({'repository':REPO,'revision':REVISION,'sha256':records},indent=2)+'\n')
    print(args.out)

if __name__=='__main__':main()
