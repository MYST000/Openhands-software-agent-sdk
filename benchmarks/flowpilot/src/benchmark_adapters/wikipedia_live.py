"""Live English Wikipedia, with page-local sentence IDs and no hidden result cache."""

import re

from bs4 import BeautifulSoup

from .research_environment import ResearchEnvironment


class WikipediaLiveEnvironment(ResearchEnvironment):
    def prepare(self):
        return {
            "backend": "wikipedia_live",
            "corpus_revision": self.config.retrieval.corpus_revision,
            "snapshot": False,
            "supporting_fact_sentence_ids": "current page-local; not historical Hotpot sentence IDs",
            "network_proxy": self.config.retrieval.web_proxy,
        }

    def search(self, query, top_k):
        if not 1 <= top_k <= self.config.retrieval.top_k:
            raise ValueError("top_k exceeds the profile limit")
        self.last_http_attempts = []
        return [
            {"docid": row["title"], **row}
            for row in self.search_engine_query(query, max_results=top_k)
        ]

    def read(self, docid, *, start_sentence=0, max_sentences=20):
        if start_sentence < 0 or not 1 <= max_sentences <= 100:
            raise ValueError("Invalid sentence range")
        self.last_http_attempts = []
        data = self.wiki_api(
            {"action": "parse", "page": docid, "prop": "text|revid", "format": "json"}
        )
        if "error" in data:
            raise ValueError(data["error"])
        page = data["parse"]
        soup = BeautifulSoup(page["text"]["*"], "html.parser")
        for tag in soup.select("script,style,nav,footer,sup.reference,.mw-editsection"):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        sentences = [
            sentence
            for paragraph in paragraphs
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
            if sentence.strip()
        ]
        end = min(len(sentences), start_sentence + max_sentences)
        return {
            "docid": docid,
            "title": page["title"],
            "page_revision_id": page.get("revid"),
            "sentences": [[i, sentences[i]] for i in range(start_sentence, end)],
            "next_sentence": end if end < len(sentences) else None,
        }
