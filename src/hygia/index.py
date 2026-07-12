"""PASSO 4 — Indexação vetorial e léxica dos chunks.

Duas representações complementares, ambas autocontidas (sem chamada de
rede, o que mantém o CI reprodutível e a comparação justa):

  vetorial : TF-IDF + LSA (SVD truncada). Captura similaridade semântica
             por sobreposição distribucional de termos. É o "embedding"
             do baseline denso.
  léxica   : BM25 clássico. Captura correspondência exata de termos, onde
             embeddings costumam patinar (nomes próprios, siglas).

O ganho do Graph RAG NÃO virá de um embedding melhor — os baselines usam
exatamente este índice. Virá de o grafo alcançar chunks que nenhuma das
duas representações aproxima da consulta.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import (
    DEFAULT_RETRIEVAL,
    INDEX_META,
    INDEX_NPZ,
    RetrievalConfig,
    garantir_diretorios,
    semear_tudo,
)
from .corpus import Chunk, carregar_chunks

_STOP = {
    "de", "da", "do", "das", "dos", "e", "a", "o", "as", "os", "um", "uma",
    "para", "por", "com", "sem", "em", "no", "na", "nos", "nas", "que", "ao",
    "aos", "se", "sua", "seu", "suas", "seus", "ou", "ser", "este", "esta",
    "deve", "pode", "quando", "sobre", "the", "of",
}


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return t


def tokenizar(texto: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if t not in _STOP and len(t) > 1]


def _identidade(x: str) -> str:
    """Preprocessador nomeado (lambdas não são serializáveis com pickle)."""
    return x


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


@dataclass
class BM25:
    ids: list[str]
    df: dict[str, int]
    tf: list[dict[str, int]]
    dl: list[int]
    avgdl: float
    n: int
    k1: float
    b: float

    @classmethod
    def treinar(cls, chunks: list[Chunk], cfg: RetrievalConfig) -> "BM25":
        ids, tf, dl = [], [], []
        df: dict[str, int] = {}
        for c in chunks:
            toks = tokenizar(f"{c.titulo} {c.secao} {c.texto}")
            cont: dict[str, int] = {}
            for t in toks:
                cont[t] = cont.get(t, 0) + 1
            for t in cont:
                df[t] = df.get(t, 0) + 1
            ids.append(c.id)
            tf.append(cont)
            dl.append(len(toks))
        n = len(chunks)
        avgdl = sum(dl) / n if n else 0.0
        return cls(ids, df, tf, dl, avgdl, n, cfg.bm25_k1, cfg.bm25_b)

    def buscar(self, consulta: str, top_k: int) -> list[tuple[str, float]]:
        q = tokenizar(consulta)
        escores = np.zeros(self.n)
        for termo in q:
            if termo not in self.df:
                continue
            idf = log((self.n - self.df[termo] + 0.5) / (self.df[termo] + 0.5) + 1.0)
            for i in range(self.n):
                f = self.tf[i].get(termo, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                escores[i] += idf * (f * (self.k1 + 1)) / denom
        ordem = np.argsort(-escores)[:top_k]
        return [(self.ids[i], float(escores[i])) for i in ordem if escores[i] > 0]

    def to_dict(self) -> dict:
        return {
            "ids": self.ids, "df": self.df, "tf": self.tf, "dl": self.dl,
            "avgdl": self.avgdl, "n": self.n, "k1": self.k1, "b": self.b,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25":
        return cls(d["ids"], d["df"], d["tf"], d["dl"], d["avgdl"], d["n"], d["k1"], d["b"])


# --------------------------------------------------------------------------
# Índice vetorial (TF-IDF + LSA)
# --------------------------------------------------------------------------


@dataclass
class IndiceVetorial:
    ids: list[str]
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    matriz: np.ndarray  # (n, d) já L2-normalizada

    @classmethod
    def treinar(cls, chunks: list[Chunk], cfg: RetrievalConfig) -> "IndiceVetorial":
        semear_tudo()
        textos = [f"{c.titulo}. {c.secao}. {c.texto}" for c in chunks]
        vec = TfidfVectorizer(
            tokenizer=tokenizar, preprocessor=_identidade,
            token_pattern=None, min_df=1, ngram_range=(1, 2), sublinear_tf=True,
        )
        X = vec.fit_transform(textos)
        n_comp = min(cfg.n_componentes, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        Z = svd.fit_transform(X)
        Z = normalize(Z)
        return cls([c.id for c in chunks], vec, svd, Z)

    def buscar(self, consulta: str, top_k: int) -> list[tuple[str, float]]:
        q = self.vectorizer.transform([f"{consulta}"])
        z = normalize(self.svd.transform(q))
        sims = (self.matriz @ z.T).ravel()
        ordem = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in ordem]


# --------------------------------------------------------------------------
# Empacotamento
# --------------------------------------------------------------------------


@dataclass
class Indice:
    vetorial: IndiceVetorial
    bm25: BM25

    @classmethod
    def treinar(cls, chunks: list[Chunk] | None = None, cfg: RetrievalConfig = DEFAULT_RETRIEVAL) -> "Indice":
        chunks = chunks or carregar_chunks()
        return cls(IndiceVetorial.treinar(chunks, cfg), BM25.treinar(chunks, cfg))

    def salvar(self) -> None:
        """Persiste a matriz LSA como artefato versionado (inspeção/auditoria).

        Os transformadores (TfidfVectorizer/TruncatedSVD) NÃO são serializados
        por pickle — eles referenciam funções de módulo e a desserialização
        seria frágil ao contexto de importação. Como todo o pipeline é
        determinístico (semente fixa em config.SEED), `carregar()` retreina o
        índice de forma idêntica bit a bit. O artefato serve para versionar o
        embedding, não para acelerar o carregamento.
        """
        garantir_diretorios()
        np.savez_compressed(INDEX_NPZ, matriz=self.vetorial.matriz)
        INDEX_META.write_text(
            json.dumps(
                {
                    "ids": self.vetorial.ids,
                    "dim_lsa": int(self.vetorial.matriz.shape[1]),
                    "vocabulario_bm25": len(self.bm25.df),
                    "nota": "matriz é artefato de inspeção; carregar() retreina deterministicamente",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def carregar(cls) -> "Indice":
        """Reconstrói o índice de forma determinística (ver salvar())."""
        return cls.treinar()


def executar(salvar: bool = True) -> Indice:
    idx = Indice.treinar()
    if salvar:
        idx.salvar()
    return idx


if __name__ == "__main__":
    idx = executar()
    print(f"índice: {len(idx.vetorial.ids)} chunks | LSA d={idx.vetorial.matriz.shape[1]} "
          f"| vocabulário BM25={len(idx.bm25.df)}")
    for q in ["risco de sangramento com varfarina", "miopatia por estatina"]:
        print(f"\nconsulta: {q}")
        for cid, sc in idx.vetorial.buscar(q, 3):
            print(f"  vec  {sc:.3f}  {cid}")
        for cid, sc in idx.bm25.buscar(q, 3):
            print(f"  bm25 {sc:.3f}  {cid}")
