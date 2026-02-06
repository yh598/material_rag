from __future__ import annotations

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TEXT_FIELDS = [
    # High-signal "material identity" fields
    "PCX_MATL_NBR",
    "MATERIAL_FAMILY_NM",
    "MATERIAL_INTENT_DESCRIPTION",
    "MATL_ITM_DESC",
    "MATERIAL_CONTENT",
    "MATERIAL_BENEFITS_NM",
    "SUPPLIED_MATERIAL_STATE_NM",
    "MATERIAL_STATUS",
    "SUPPLIED_MATERIAL_ID",
    "SUPPLEMENTAL_MATERIAL_NM",
    "MATL_COMMENT",
    "SUPLD_MATL_COMMENT",
    # Optional yarn composition fields
    "YARN_1_MATERIAL_CONTENT",
    "YARN_2_MATERIAL_CONTENT",
    "YARN_3_MATERIAL_CONTENT",
    "YARN_4_MATERIAL_CONTENT",
]

DEFAULT_MATERIAL_LABEL_FIELDS = [
    "MATL_ITM_DESC",
    "MATERIAL_FAMILY_NM",
    "PCX_MATL_NBR",
    "SUPPLIED_MATERIAL_ID",
]


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


class PaletteIndex:
    def __init__(self, df: pd.DataFrame, vectorizer: TfidfVectorizer, X):
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.X = X

    @classmethod
    def from_csv(cls, csv_path: str) -> "PaletteIndex":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Build doc text per row
        docs = []
        for _, row in df.iterrows():
            parts = []
            for f in DEFAULT_TEXT_FIELDS:
                if f in df.columns:
                    v = _safe_str(row.get(f))
                    if v:
                        parts.append(f"{f}: {v}")
            docs.append("\n".join(parts))

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=80_000,
            stop_words="english",
        )
        X = vectorizer.fit_transform(docs)
        df["_doc_text"] = docs

        return cls(df=df, vectorizer=vectorizer, X=X)

    def _material_label(self, row: pd.Series) -> str:
        pieces = []
        for f in DEFAULT_MATERIAL_LABEL_FIELDS:
            if f in self.df.columns:
                v = _safe_str(row.get(f))
                if v:
                    pieces.append(v)
        return " | ".join(pieces)[:160]

    def search(self, query: str, top_k: int = 6):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).ravel()

        if top_k <= 0:
            top_k = 6
        k = min(top_k, len(sims))
        idxs = np.argpartition(-sims, k-1)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

        results = []
        for i in idxs:
            row = self.df.iloc[int(i)]
            full = row["_doc_text"]
            snippet = full[:280].replace("\n", " ")
            results.append({
                "row_id": int(i),
                "score": float(sims[int(i)]),
                "material": self._material_label(row),
                "snippet": snippet,
                "fulltext": full,
            })
        return results
