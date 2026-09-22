"""Route declared task identities to benchmark protocols and environment capabilities."""

from dataclasses import dataclass, replace

from .config import Config


@dataclass(frozen=True)
class AdapterRegistration:
    kind: str
    dataset_id: str
    setting: str
    tool_profile: str
    environment_tools: tuple[str, ...]
    code_workspace: bool = False


ADAPTERS = {
    "quixbugs": AdapterRegistration(
        "quixbugs",
        "jkoppel/QuixBugs",
        "openhands-python-code-v1",
        "python-workspace-v1",
        ("code_terminal", "code_file_editor"),
        True,
    ),
    "livecodebench": AdapterRegistration(
        "livecodebench",
        "livecodebench/code_generation_lite",
        "openhands-python-code-v1",
        "python-workspace-v1",
        ("code_terminal", "code_file_editor"),
        True,
    ),
    "hotpot": AdapterRegistration(
        "hotpot",
        "hotpotqa",
        "fullwiki-fixed-corpus-v1",
        "fixed-corpus-sentences-v1",
        ("search", "read_document"),
    ),
    "browsecomp": AdapterRegistration(
        "browsecomp",
        "Tevatron/browsecomp-plus",
        "openhands-search-read-v1",
        "fixed-corpus-pages-v1",
        ("search", "get_document"),
    ),
}


def resolve_adapter(config: Config) -> AdapterRegistration:
    registration = ADAPTERS.get(config.dataset.kind)
    if registration is None:
        raise ValueError("No registered mixed-workload adapter for dataset.kind")
    if (config.dataset.id, config.dataset.setting) != (
        registration.dataset_id,
        registration.setting,
    ):
        raise ValueError("Declared benchmark identity/protocol does not match adapter")
    if not isinstance(config.dataset.revision, str) or not config.dataset.revision.strip():
        raise ValueError("Routing requires a pinned dataset revision")
    if registration.kind == "browsecomp" and config.retrieval.backend == "browsecomp_mcp":
        return replace(registration, tool_profile="browsecomp-native-bm25-mcp-v1")
    return registration


def load_adapter_tasks(config, ids, *, checker_path=None):
    """Only the controller reads labels or private tests; routing never uses them."""
    registration = resolve_adapter(config)
    if (
        not ids
        or any(not isinstance(task_id, str) or not task_id.strip() for task_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Explicit nonempty unique string task IDs are required")
    if registration.code_workspace:
        from .code_collection import load_selected

        bundles, provenance = load_selected(config, ids, checker_path, check_isolation=False)
        return [(bundle.task, bundle) for bundle in bundles], provenance
    from .retrieval import BrowseCompAdapter, HotpotAdapter
    from .swe import checked_dataset_digest

    dataset_digest = checked_dataset_digest(config)

    adapter = HotpotAdapter if registration.kind == "hotpot" else BrowseCompAdapter
    tasks = adapter.load(
        config.dataset.path,
        dataset_id=config.dataset.id,
        revision=config.dataset.revision,
        split=config.dataset.split,
        ids=ids,
        limit=0,
    )
    return [(task, None) for task in tasks], {
        "revision": config.dataset.revision,
        "dataset_sha256": dataset_digest,
        "corpus_revision": config.retrieval.corpus_revision,
        "selection_rule": "explicit task IDs; no model-based routing or outcome filtering",
    }
