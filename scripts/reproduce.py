"""Byte-for-byte regeneration using a streaming digest, avoiding a second dataset."""
import hashlib
from itertools import chain
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exact_sequential_inference.common import dumps, load_json
from exact_sequential_inference.generate import active_records, temporal_records, config_check

root=Path(sys.argv[1])
manifest=load_json(root/'manifest.json')
config=config_check(manifest['config'])
sha=hashlib.sha256();count=0;size=0
for r in chain(active_records(config),temporal_records(config)):
    data=(dumps(r)+'\n').encode()
    sha.update(data);count+=1;size+=len(data)
result={'records':count,'bytes':size,'sha256':sha.hexdigest(),'matches_release':count==manifest['records'] and size==manifest['bytes'] and sha.hexdigest()==manifest['records_sha256']}
print(dumps(result))
assert result['matches_release']
