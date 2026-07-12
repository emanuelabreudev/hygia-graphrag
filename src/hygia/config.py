"""Configuração central: caminhos, sementes e hiperparâmetros.

Toda aleatoriedade do projeto passa por SEED. Nenhum módulo deve chamar
random/numpy diretamente sem derivar de `rng()`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED_DIR = DATA / "seed"
CORPUS_DIR = DATA / "corpus"
PROCESSED_DIR = DATA / "processed"
RESULTS_DIR = ROOT / "resultados"
FIG_DIR = ROOT / "docs" / "img"

MEDICAMENTOS_JSON = SEED_DIR / "medicamentos.json"
INTERACOES_JSON = SEED_DIR / "interacoes.json"

CHUNKS_JSON = CORPUS_DIR / "chunks.json"
GRAPH_JSON = PROCESSED_DIR / "grafo.json"
TRIPLAS_JSON = PROCESSED_DIR / "triplas.json"
INDEX_NPZ = PROCESSED_DIR / "indice.npz"
INDEX_META = PROCESSED_DIR / "indice_meta.json"
GOLD_JSON = DATA / "gold" / "perguntas.json"

MODELO_LLM = "claude-opus-4-8"


@dataclass(frozen=True)
class RetrievalConfig:
    """Hiperparâmetros de recuperação.

    Os valores default foram fixados por varredura em conjunto de
    desenvolvimento (ver docs/METODOLOGIA.md, seção 5).
    """

    top_k: int = 8
    # nº de dimensões da decomposição LSA usada como espaço denso
    n_componentes: int = 128
    # profundidade máxima da caminhada no grafo a partir das âncoras
    max_hops: int = 3
    # constante do Reciprocal Rank Fusion
    rrf_k: int = 60
    # peso relativo dos candidatos vindos do grafo na fusão
    peso_grafo: float = 1.0
    peso_vetor: float = 1.0
    peso_bm25: float = 1.0
    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # nº máximo de fatos do subgrafo injetados no prompt
    max_fatos_grafo: int = 12


DEFAULT_RETRIEVAL = RetrievalConfig()


def rng(offset: int = 0) -> np.random.Generator:
    """Gerador determinístico. `offset` permite streams independentes."""
    return np.random.default_rng(SEED + offset)


def semear_tudo(offset: int = 0) -> None:
    """Fixa as sementes globais (random, numpy, PYTHONHASHSEED)."""
    random.seed(SEED + offset)
    np.random.seed(SEED + offset)
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))


def garantir_diretorios() -> None:
    for d in (CORPUS_DIR, PROCESSED_DIR, RESULTS_DIR, FIG_DIR, GOLD_JSON.parent):
        d.mkdir(parents=True, exist_ok=True)
