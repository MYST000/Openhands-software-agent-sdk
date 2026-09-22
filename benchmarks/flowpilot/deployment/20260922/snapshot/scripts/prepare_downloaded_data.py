"""Verify pinned downloads and prepare frozen, controller-private task manifests.

No model, agent, benchmark evaluator, or collection run is started.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'handoff_reference_20260921'
EVIDENCE = ROOT / 'evidence/preparation'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify_hf():
    sources = json.loads((EVIDENCE / 'hf_sources.lock.json').read_text())
    work = [(source, f) for source in sources for f in source['files']]
    def verify(item):
        source, f = item
        path = ROOT / source['directory'] / f['path']
        assert path.stat().st_size == f['size'], str(path)
        sha = digest(path)
        if f['sha256']:
            assert sha == f['sha256'], str(path)
        else:
            h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode() + path.read_bytes())
            assert h.hexdigest() == f['git_blob'], str(path)
        result = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha}
        return result
    # Keep the hash check streaming; benchmark rows are never logged.
    with ThreadPoolExecutor(max_workers=4) as ex:
        result = list(ex.map(verify, work))
    save(EVIDENCE / 'hf_verified_files.json', result)
    print(f'Verified {len(result)} Hugging Face files against pinned upstream hashes', flush=True)


def prepare_lcb():
    lock = json.loads((REFERENCE / 'protocol/lcb_release_v6.lock.json').read_text())
    hashes = {Path(r['path']).name: r['sha256'] for r in json.loads((EVIDENCE / 'hf_verified_files.json').read_text()) if 'livecodebench/' in r['path']}
    for name, expected in lock['files'].items():
        assert hashes[name] == expected, name
    checker = ROOT / 'data/livecodebench/testing_util.reference.py'
    assert digest(checker) == lock['checker_sha256']
    protocol = REFERENCE / 'protocol/code_splits'
    frozen = [json.loads(line) for line in (protocol / 'livecodebench_v1.jsonl').read_text().splitlines()]
    by_id = {row['question_id']: row for row in frozen}
    assert len(by_id) == len(frozen) == 362
    selected = ROOT / 'data/livecodebench/lcb362.jsonl'
    tmp = selected.with_suffix('.jsonl.tmp')
    seen, all_ids = set(), set()
    raw_count, split_counts = 0, Counter()
    with tmp.open('wb') as out:
        for name in lock['files']:
            with (ROOT / 'data/livecodebench/release_v6' / name).open('rb') as source:
                for line in source:
                    row = json.loads(line)
                    task_id = str(row['question_id'])
                    assert task_id not in all_ids
                    all_ids.add(task_id)
                    raw_count += 1
                    if task_id not in by_id:
                        continue
                    expected = by_id[task_id]
                    assert task_id not in seen
                    assert expected['source_file'] == name
                    for key in ['platform', 'difficulty', 'contest_date']:
                        assert expected[key] == row[key]
                    assert hashlib.sha256(row['question_content'].encode()).hexdigest() == expected['statement_sha256']
                    public = json.loads(row['public_test_cases'])
                    meta = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    assert public and all(c['testtype'] == 'stdin' for c in public)
                    assert not meta.get('func_name') and row['private_test_cases']
                    assert row['difficulty'] in ['easy', 'medium']
                    out.write(line if line.endswith(b'\n') else line + b'\n')
                    seen.add(task_id)
                    split_counts[expected['research_split']] += 1
    assert raw_count == 1055 and seen == set(by_id)
    assert dict(split_counts) == json.loads((protocol / 'manifest.json').read_text())['counts']
    tmp.replace(selected)
    dest = ROOT / 'data/livecodebench/protocol'
    shutil.copytree(protocol, dest, dirs_exist_ok=True)
    shutil.copy2(REFERENCE / 'protocol/lcb_release_v6.lock.json', dest)
    result = {'raw_rows': raw_count, 'selected_rows': len(seen), 'splits': dict(split_counts), 'path': str(selected.relative_to(ROOT)), 'sha256': digest(selected), 'protocol_sha256': digest(dest / 'livecodebench_v1.jsonl'), 'private_tests': 'retained in upstream representation; no evaluator or pickle executed'}
    save(EVIDENCE / 'lcb_preparation.json', result)
    print('LCB verified: 1055 raw rows, 362 frozen tasks, original splits preserved', flush=True)
    return result


def prepare_browsecomp():
    corpus_root = ROOT / 'data/browsecomp_plus/corpus/data'
    docids = set()
    corpus_count = 0
    for file in sorted(corpus_root.glob('*.parquet')):
        table = pq.read_table(file, columns=['docid'])
        for docid in table['docid'].to_pylist():
            assert docid not in docids
            docids.add(docid)
            corpus_count += 1
    assert corpus_count == 100195
    source = REFERENCE / 'upstream/browsecomp_decrypt_dataset.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    # The official downloader import is unnecessary for already-downloaded Parquet.
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == 'datasets')]
    decoder = {'__name__': 'official_browsecomp_decoder'}
    exec(compile(tree, str(source), 'exec'), decoder)
    destination = ROOT / 'data/browsecomp_plus/queries_private.jsonl'
    tmp = destination.with_suffix('.jsonl.tmp')
    ids = set()
    relevance = Counter()
    with tmp.open('w') as out:
        for file in sorted((ROOT / 'data/browsecomp_plus/questions/data').glob('*.parquet')):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    slim = {k: row[k] for k in ['query_id', 'query', 'answer']}
                    for kind in ['gold', 'negative', 'evidence']:
                        slim[kind + '_doc_ids'] = [doc['docid'] for doc in row[kind + '_docs']]
                    decoded = decoder['transform_decrypt'](slim, decoder['DEFAULT_CANARY'], {'query_id'})
                    assert decoded['query_id'] not in ids
                    assert decoded['query'] and decoded['answer']
                    ids.add(decoded['query_id'])
                    for kind in ['gold', 'negative', 'evidence']:
                        refs = decoded[kind + '_doc_ids']
                        assert set(refs) <= docids, 'Question references missing corpus document'
                        relevance[kind] += len(refs)
                    out.write(json.dumps(decoded, ensure_ascii=False) + '\n')
    assert len(ids) == 830
    tmp.replace(destination)
    destination.chmod(0o600)
    shutil.copy2(REFERENCE / 'protocol/browsecomp_known_history.ids', destination.parent)
    result = {'queries': len(ids), 'documents': corpus_count, 'all_relevance_docids_in_corpus': True, 'relevance_references': dict(relevance), 'prepared_queries': str(destination.relative_to(ROOT)), 'prepared_queries_sha256': digest(destination), 'decoder_sha256': digest(source), 'prepared_schema': 'query_id, query, answer, gold_doc_ids, evidence_doc_ids, negative_doc_ids; document text retained in fixed corpus and raw Parquet', 'known_history': 'preserved browsecomp_known_history.ids; no new split generated'}
    save(EVIDENCE / 'browsecomp_preparation.json', result)
    print('BrowseComp-Plus verified: 830 queries, 100195 unique documents; relevance IDs all resolve', flush=True)
    return result


def verify_hotpot_questions():
    counts = Counter()
    schema = None
    ids = set()
    for file in sorted((ROOT / 'data/hotpotqa/questions/fullwiki').glob('*.parquet')):
        split = file.name.split('-')[0]
        pf = pq.ParquetFile(file)
        counts[split] += pf.metadata.num_rows
        schema = pf.schema_arrow.names
        for batch in pf.iter_batches(columns=['id'], batch_size=4096):
            for task_id in batch.column(0).to_pylist():
                assert task_id not in ids
                ids.add(task_id)
    assert dict(counts) == {'train': 90447, 'validation': 7405, 'test': 7405}
    result = {'rows': dict(counts), 'unique_ids': len(ids), 'schema': schema, 'source': 'pinned hotpotqa/hotpot_qa fullwiki Parquet mirror; corpus is a separate official Stanford archive'}
    save(EVIDENCE / 'hotpot_questions_preparation.json', result)
    print('Hotpot fullwiki questions verified: 90447 train / 7405 validation / 7405 test', flush=True)
    return result


def verify_repositories():
    result = {}
    pairs = [('openhands', 'repos/Openhands-software-agent-sdk', '8e8ed520c8abb09c4c4d3369492d3b02988e2c18'), ('quixbugs', 'data/QuixBugs', '4257f44b0ff1181dedaedee6a447e133219fcebf')]
    for key, path, expected in pairs:
        def git(*args):
            return subprocess.check_output(['git', '-C', str(ROOT/path), *args], text=True).strip()
        commit = git('rev-parse', 'HEAD')
        assert commit == expected and not git('status', '--porcelain')
        subprocess.run(['git', '-C', str(ROOT/path), 'fsck', '--no-progress'], check=True, capture_output=True)
        result[key] = {'path': path, 'commit': commit, 'clean': True, 'git_fsck': 'passed'}
    result['openhands']['branch'] = subprocess.check_output(['git', '-C', str(ROOT/pairs[0][1]), 'branch', '--show-current'], text=True).strip()
    assert result['openhands']['branch'] == 'ql'
    quix_ids = (REFERENCE / 'protocol/code_splits/quixbugs_dev40.ids').read_text().splitlines()
    assert len(set(quix_ids)) == 40
    quix_root = ROOT / 'data/QuixBugs'
    for task in quix_ids:
        for folder, name in [('python_programs', task+'.py'), ('correct_python_programs', task+'.py'), ('python_testcases', 'test_'+task+'.py')]:
            assert (quix_root/folder/name).is_file(), task
    result['quixbugs']['tasks'] = len(quix_ids)
    result['quixbugs']['selection'] = 'data/livecodebench/protocol/quixbugs_dev40.ids'
    save(EVIDENCE / 'repository_preparation.json', result)
    print('Repositories verified: OpenHands ql, QuixBugs pinned commit and all 40 tasks', flush=True)
    return result


def main():
    os.umask(0o077)
    actions = {'hashes': verify_hf, 'lcb': prepare_lcb, 'browsecomp': prepare_browsecomp, 'hotpot_questions': verify_hotpot_questions, 'repositories': verify_repositories}
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', choices=list(actions), default=list(actions))
    args = ap.parse_args()
    for stage in args.stages:
        actions[stage]()
    files = {'lcb': 'lcb_preparation.json', 'browsecomp': 'browsecomp_preparation.json', 'hotpot_questions': 'hotpot_questions_preparation.json', 'repositories': 'repository_preparation.json'}
    if all((EVIDENCE/name).is_file() for name in files.values()):
        save(EVIDENCE / 'dataset_preparation.json', {key: json.loads((EVIDENCE/name).read_text()) for key, name in files.items()})
        print('DATASET_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
