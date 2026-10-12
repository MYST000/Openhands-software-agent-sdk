from dataclasses import replace

import pytest

from benchmark_adapters.adapter_registry import resolve_adapter
from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.reuse_profile import export_registry, retrieval_scope
from benchmark_adapters.wikipedia_live import WikipediaLiveEnvironment


@pytest.mark.parametrize(
    "kind,dataset_id",
    [
        ("hotpot", "hotpotqa"),
        ("twowiki", "Alab-NII/2wikimultihop"),
    ],
)
def test_live_wikipedia_tools_and_session_isolation(kind, dataset_id, monkeypatch):
    config = Config(
        dataset=DatasetConfig(
            kind=kind, id=dataset_id, revision="fixture", setting="fullwiki-live-wikipedia-v1"
        ),
        retrieval=RetrievalConfig(
            backend="wikipedia_live",
            corpus_revision="session-1",
            server_policy_revision="live-wiki-v1",
        ),
    )
    assert resolve_adapter(config).tool_profile == "live-wikipedia-sentences-v1"
    environment = WikipediaLiveEnvironment(config)

    def api(parameters):
        if parameters["action"] == "query":
            assert parameters["srsearch"] == "Tea"
            return {"query": {"search": [{"title": "Tea", "snippet": "<b>Tea</b> drink"}]}}
        return {
            "parse": {
                "title": "Tea",
                "revid": 123,
                "text": {
                    "*": "<nav>Ignore me.</nav><p>Tea is a drink. It is popular.<sup class='reference'>[1]</sup></p>"
                },
            }
        }

    monkeypatch.setattr(environment, "wiki_api", api)
    try:
        assert environment.prepare()["snapshot"] is False
        hit = environment.search("Tea", 1)[0]
        assert hit["docid"] == "Tea" and hit["snippet"] == "Tea drink"
        page = environment.read("Tea", max_sentences=1)
        assert page["sentences"] == [[0, "Tea is a drink."]]
        assert page["page_revision_id"] == 123 and page["next_sentence"] == 1
        assert environment.read("Tea", start_sentence=1)["sentences"] == [[1, "It is popular."]]
        with pytest.raises(ValueError, match="Invalid sentence range"):
            environment.read("Tea", start_sentence=-1)
    finally:
        environment.close()
    other = replace(config, retrieval=replace(config.retrieval, corpus_revision="session-2"))
    assert retrieval_scope(config) != retrieval_scope(other)
    assert {entry["tool_name"] for entry in export_registry(config)} == {"search", "read_document"}
