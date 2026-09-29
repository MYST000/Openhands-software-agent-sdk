"""Local-only launcher for the unchanged official BrowseComp-Plus BM25 tools."""

import argparse
import json
import importlib.util
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "repos/BrowseComp-Plus/searcher"
# Pyserini's Lucene package also imports unused neural searchers eagerly.
# Execute its pinned initializer with only those two imports omitted; retain the
# unchanged Java bindings and official _searcher.py implementation.
importlib.import_module("pyserini.search")
lucene_spec = importlib.util.find_spec("pyserini.search.lucene")
assert lucene_spec is not None and lucene_spec.origin is not None
lucene_source = Path(lucene_spec.origin).read_text()
for unused_import in (
    "from ._impact_searcher import LuceneImpactSearcher, SlimSearcher",
    "from ._hnsw_searcher import LuceneHnswDenseSearcher, LuceneFlatDenseSearcher",
):
    if lucene_source.count(unused_import) != 1:
        raise RuntimeError(
            "Unexpected Pyserini initializer; revalidate sparse bootstrap"
        )
    lucene_source = lucene_source.replace(unused_import, "")
lucene_module = importlib.util.module_from_spec(lucene_spec)
sys.modules["pyserini.search.lucene"] = lucene_module
exec(compile(lucene_source, lucene_spec.origin, "exec"), lucene_module.__dict__)
# Load only the official sparse modules, avoiding the upstream eager dense registry.
package = types.ModuleType("searchers")
package.__path__ = [str(SOURCE / "searchers")]
sys.modules["searchers"] = package
sys.path.insert(0, str(SOURCE))
from searchers.bm25_searcher import BM25Searcher  # noqa: E402
from tools import register_tools  # noqa: E402
from fastmcp import FastMCP  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    searcher = BM25Searcher(
        argparse.Namespace(
            index_path=str(ROOT / "data/browsecomp_plus/native_indexes/bm25")
        )
    )
    server = FastMCP(name="search-server")
    register_tools(
        server, searcher, snippet_max_tokens=512, k=5, include_get_document=True
    )
    print(
        json.dumps(
            {
                "event": "native_bm25_ready",
                "pid": os.getpid(),
                "documents": searcher.searcher.num_docs,
                "url": f"http://127.0.0.1:{args.port}/mcp",
                "k": 5,
                "snippet_max_tokens": 512,
            }
        ),
        flush=True,
    )
    server.run(
        transport="streamable-http", host="127.0.0.1", port=args.port, path="/mcp"
    )


if __name__ == "__main__":
    main()
