"""PASSO 5 — Recuperação. Baselines e Graph RAG sob a mesma interface.

Todos os recuperadores implementam `recuperar(consulta, entidades) -> Resultado`.
`entidades` são os nós âncora já resolvidos da pergunta (medicamentos,
classes) — a ligação linguagem→grafo. Isolar isso mantém a comparação
honesta: o baseline recebe a MESMA consulta textual; o que ele não faz é
usar o grafo.

Recuperadores:
  VetorialRetriever   — apenas LSA (baseline denso)
  BM25Retriever       — apenas BM25 (baseline léxico)
  HibridoRetriever    — LSA + BM25 fundidos por RRF (baseline forte, "plain RAG")
  GraphRAGRetriever   — LSA + BM25 + caminhada no grafo, RRF (proposta)

Fusão: Reciprocal Rank Fusion. Robusta a escalas incompatíveis entre
similaridade de cosseno, escore BM25 e prioridade de grafo — só a ordem
importa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import networkx as nx

from .config import DEFAULT_RETRIEVAL, RetrievalConfig
from .graph import Achado, analisar_par, chunks_de_entidades, vizinhanca
from .index import Indice
from .schema import TipoEntidade, no_id


@dataclass
class Resultado:
    chunks: list[str]                    # ids de chunk ordenados
    escores: dict[str, float]
    achados_grafo: list[Achado] = field(default_factory=list)
    subgrafo_nos: list[str] = field(default_factory=list)
    proveniencia: dict[str, str] = field(default_factory=dict)  # chunk -> fonte


def rrf(rankings: list[list[str]], pesos: list[float], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion ponderada."""
    escore: dict[str, float] = {}
    for ranking, peso in zip(rankings, pesos):
        for pos, item in enumerate(ranking):
            escore[item] = escore.get(item, 0.0) + peso / (k + pos + 1)
    return escore


class Retriever(Protocol):
    nome: str

    def recuperar(self, consulta: str, entidades: list[tuple[TipoEntidade, str]]) -> Resultado: ...


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


@dataclass
class VetorialRetriever:
    indice: Indice
    cfg: RetrievalConfig = DEFAULT_RETRIEVAL
    nome: str = "vetorial"

    def recuperar(self, consulta, entidades):
        pares = self.indice.vetorial.buscar(consulta, self.cfg.top_k)
        chunks = [c for c, _ in pares]
        return Resultado(chunks, dict(pares), proveniencia={c: "vetor" for c in chunks})


@dataclass
class BM25Retriever:
    indice: Indice
    cfg: RetrievalConfig = DEFAULT_RETRIEVAL
    nome: str = "bm25"

    def recuperar(self, consulta, entidades):
        pares = self.indice.bm25.buscar(consulta, self.cfg.top_k)
        chunks = [c for c, _ in pares]
        return Resultado(chunks, dict(pares), proveniencia={c: "bm25" for c in chunks})


@dataclass
class HibridoRetriever:
    """Plain RAG forte: recuperação densa + léxica fundidas por RRF.

    É o baseline que interessa vencer. Não usa o grafo.
    """

    indice: Indice
    cfg: RetrievalConfig = DEFAULT_RETRIEVAL
    nome: str = "hibrido"

    def recuperar(self, consulta, entidades):
        n = self.cfg.top_k * 3
        r_vec = [c for c, _ in self.indice.vetorial.buscar(consulta, n)]
        r_bm = [c for c, _ in self.indice.bm25.buscar(consulta, n)]
        fund = rrf([r_vec, r_bm], [self.cfg.peso_vetor, self.cfg.peso_bm25], self.cfg.rrf_k)
        ordenados = sorted(fund, key=lambda c: -fund[c])[: self.cfg.top_k]
        prov = {c: ("vetor+bm25" if c in r_vec and c in r_bm else "vetor" if c in r_vec else "bm25")
                for c in ordenados}
        return Resultado(ordenados, {c: fund[c] for c in ordenados}, proveniencia=prov)


# --------------------------------------------------------------------------
# Graph RAG
# --------------------------------------------------------------------------


@dataclass
class GraphRAGRetriever:
    """Recuperação híbrida guiada por grafo (a proposta).

    Duas contribuições sobre o baseline híbrido:

      1. Caminhada no grafo a partir das entidades âncora recupera chunks
         que evidenciam as arestas do subgrafo — inclusive chunks que
         nenhuma das duas representações textuais aproximaria da consulta
         (ex.: a monografia da classe do OUTRO fármaco).

      2. Análise de par produz `achados` estruturados (interações diretas,
         por classe e por mecanismo) com o caminho de raciocínio e a
         severidade. Isso vai para o prompt como contexto estruturado —
         é o "graph context" do Passo 6.
    """

    indice: Indice
    grafo: nx.MultiDiGraph
    cfg: RetrievalConfig = DEFAULT_RETRIEVAL
    nome: str = "graphrag"

    def recuperar(self, consulta, entidades):
        n = self.cfg.top_k * 3
        r_vec = [c for c, _ in self.indice.vetorial.buscar(consulta, n)]
        r_bm = [c for c, _ in self.indice.bm25.buscar(consulta, n)]

        # --- caminhada no grafo ---
        sementes = [no_id(t, nome) for t, nome in entidades if no_id(t, nome) in self.grafo]
        subgrafo = vizinhanca(self.grafo, sementes, hops=self.cfg.max_hops)
        chunk_para_nos = chunks_de_entidades(self.grafo, subgrafo)
        # prioriza chunks que tocam MAIS entidades do subgrafo (mais informativos)
        r_grafo = sorted(chunk_para_nos, key=lambda c: -len(chunk_para_nos[c]))[:n]

        fund = rrf(
            [r_vec, r_bm, r_grafo],
            [self.cfg.peso_vetor, self.cfg.peso_bm25, self.cfg.peso_grafo],
            self.cfg.rrf_k,
        )
        ordenados = sorted(fund, key=lambda c: -fund[c])[: self.cfg.top_k]

        # --- achados estruturados (todos os pares de medicamentos âncora) ---
        meds = [nome for t, nome in entidades if t is TipoEntidade.MEDICAMENTO]
        achados: list[Achado] = []
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                achados.extend(analisar_par(self.grafo, meds[i], meds[j]))

        # garante que as evidências dos achados estejam no contexto textual
        for ach in achados:
            for cid in ach.chunks:
                if cid and cid not in ordenados:
                    ordenados.append(cid)

        prov: dict[str, str] = {}
        for c in ordenados:
            fontes = []
            if c in r_vec:
                fontes.append("vetor")
            if c in r_bm:
                fontes.append("bm25")
            if c in r_grafo or c in chunk_para_nos:
                fontes.append("grafo")
            prov[c] = "+".join(fontes) or "grafo"

        return Resultado(
            chunks=ordenados,
            escores={c: fund.get(c, 0.0) for c in ordenados},
            achados_grafo=achados,
            subgrafo_nos=sorted(subgrafo),
            proveniencia=prov,
        )


def construir_recuperadores(
    indice: Indice, grafo: nx.MultiDiGraph, cfg: RetrievalConfig = DEFAULT_RETRIEVAL
) -> dict[str, Retriever]:
    return {
        "vetorial": VetorialRetriever(indice, cfg),
        "bm25": BM25Retriever(indice, cfg),
        "hibrido": HibridoRetriever(indice, cfg),
        "graphrag": GraphRAGRetriever(indice, grafo, cfg),
    }
