"""Validate the official Hotpot fullwiki archive and extract its compressed shards."""
from concurrent.futures import ThreadPoolExecutor
import bz2
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT/'data/hotpotqa/raw/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'
output = ROOT/'data/hotpotqa/corpus'
output.mkdir(exist_ok=True)
sha, md5 = hashlib.sha256(), hashlib.md5()
with archive.open('rb') as f:
    for block in iter(lambda: f.read(8*1024*1024), b''):
        sha.update(block)
        md5.update(block)
assert archive.stat().st_size == 1553565403
assert md5.hexdigest() == '01edf64cd120ecc03a2745352779514c'
shards = []
with tarfile.open(archive, 'r|bz2') as tar:
    for member in tar:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        dest = output.joinpath(*path.parts)
        if member.isdir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            assert member.isfile(), 'Unexpected non-file archive entry'
            assert dest.resolve().is_relative_to(output.resolve()) and not dest.is_symlink()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name+'.extracting')
            with tar.extractfile(member) as source, tmp.open('wb') as target:
                shutil.copyfileobj(source, target, 1024*1024)
            assert tmp.stat().st_size == member.size
            tmp.replace(dest)
            if dest.suffix == '.bz2':
                shards.append(dest)
print(f'Archive extracted: {len(shards)} compressed shards', flush=True)

def inspect(path):
    count, keys = 0, None
    with bz2.open(path, 'rt') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            assert {'id', 'title', 'text'} <= set(row)
            assert isinstance(row['text'], list)
            keys = sorted(row)
            count += 1
    return count, keys

counts = []
with ThreadPoolExecutor(max_workers=4) as ex:
    for count, fields in ex.map(inspect, shards):
        counts.append(count)
        if len(counts)%100 == 0:
            print(f'Validated {len(counts)}/{len(shards)} shards', flush=True)
result = {'source': 'https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2', 'snapshot': '2017-10-01', 'scope': 'official introductory paragraphs used in HotpotQA fullwiki', 'archive': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'compressed_shards': len(shards), 'documents': sum(counts), 'schema': fields, 'extracted_directory': str(output.relative_to(ROOT)), 'archive_and_shard_integrity': 'passed', 'retrieval_index_built': False}
(ROOT/'evidence/preparation/hotpot_corpus_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
print('HOTPOT_CORPUS_VERIFIED',sum(counts),'documents',flush=True)
