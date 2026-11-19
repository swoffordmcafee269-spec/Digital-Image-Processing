import json
from pathlib import Path
from .common import load_config


if __name__=='__main__':
    cfg=load_config();root=Path(cfg['artifact_root'])
    for folder in ['logs','results']:
        for p in sorted((root/folder).glob('*/status.json')):
            print(str(p.relative_to(root)))
            print(json.dumps(json.loads(p.read_text(encoding='utf-8')),ensure_ascii=False,indent=2))
