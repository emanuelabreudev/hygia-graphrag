"""PASSO 3 — Grafo de conhecimento + motor de inferência mecanística.

O grafo é um `networkx.MultiDiGraph`. Cada nó é `Tipo::nome` e cada aresta
carrega a relação, os atributos clínicos e o `chunk_id` que a evidencia.

A parte que interessa é o motor de inferência. Uma interação entre dois
fármacos pode ser alcançada por quatro caminhos de comprimento crescente:

  [1] direta      A --INTERAGE_COM--> B
                  (um documento afirma isso literalmente)

  [2] classe      A --INTERAGE_COM_CLASSE--> C <--E_DA_CLASSE-- B
                  (a bula de A cita a classe; a monografia diz que B é dela)

  [3] classe×classe
                  A --E_DA_CLASSE--> Ca --INTERAGE_COM_CLASSE--> Cb <--E_DA_CLASSE-- B

  [4] mecanismo   A --INIBE/INDUZ--> Enzima <--SUBSTRATO_DE/ATIVADO_POR-- B
                  (nenhum documento cita A e B juntos; a interação é DERIVADA
                   do cruzamento de duas bulas que só falam de enzimas)

Os caminhos [2], [3] e [4] são exatamente onde o RAG por similaridade
falha: nenhum trecho de texto contém os dois fármacos, então nenhum
vetor fica próximo da consulta. O grafo, ao contrário, chega lá por
construção — e devolve o caminho percorrido, o que torna a resposta
auditável.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx

from .config import GRAPH_JSON, INTERACOES_JSON, garantir_diretorios
from .schema import (
    OrigemAresta,
    Severidade,
    TipoEntidade,
    TipoRelacao,
    Tripla,
    no_id,
)

_INTENSIDADE_RELEVANTE = {"forte", "moderada"}
_SUBSTRATO_SENSIVEL = {"sensivel", "sensível"}


@dataclass
class Achado:
    """Uma interação encontrada entre dois fármacos, com sua proveniência."""

    tipo: str  # 'direta' | 'classe' | 'classe_classe' | 'mecanismo'
    a: str
    b: str
    severidade: Severidade
    mecanismo: str
    efeito: str
    manejo: str
    evidencia: str
    caminho: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    regra: str | None = None

    @property
    def n_saltos(self) -> int:
        return max(len(self.caminho) - 1, 1)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["severidade"] = self.severidade.value
        d["n_saltos"] = self.n_saltos
        return d


# --------------------------------------------------------------------------
# Construção
# --------------------------------------------------------------------------


def construir_grafo(triplas: Iterable[Tripla]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for t in triplas:
        s = no_id(t.tipo_sujeito, t.sujeito)
        o = no_id(t.tipo_objeto, t.objeto)
        if s not in G:
            G.add_node(s, tipo=t.tipo_sujeito.value, nome=t.sujeito)
        if o not in G:
            G.add_node(o, tipo=t.tipo_objeto.value, nome=t.objeto)
        G.add_edge(
            s, o,
            key=f"{t.relacao.value}|{t.chunk_id or t.origem.value}",
            relacao=t.relacao.value,
            chunk_id=t.chunk_id,
            origem=t.origem.value,
            **t.atributos,
        )
    return G


# --------------------------------------------------------------------------
# Motor de inferência (regras R1–R4 de data/seed/interacoes.json)
# --------------------------------------------------------------------------


def _severidade_mecanistica(papel_a: str, intensidade_a: str, papel_b: str, intensidade_b: str) -> Severidade:
    """Gradua a severidade da interação derivada.

    Inibição forte de uma via da qual o outro fármaco é substrato sensível
    (ou, pior, pró-fármaco dependente) é o cenário de maior risco.
    """
    if papel_b == TipoRelacao.ATIVADO_POR.value:
        # bloquear a bioativação de um pró-fármaco = falha terapêutica silenciosa
        return Severidade.CONTRAINDICADO if intensidade_a == "forte" else Severidade.GRAVE
    if intensidade_a == "forte" and intensidade_b in _SUBSTRATO_SENSIVEL:
        return Severidade.CONTRAINDICADO
    if intensidade_a == "forte" or intensidade_b in _SUBSTRATO_SENSIVEL:
        return Severidade.GRAVE
    return Severidade.MODERADA


def inferir_mecanismos(G: nx.MultiDiGraph) -> list[Tripla]:
    """Deriva interações fármaco-fármaco a partir das vias enzimáticas.

    Percorre todo par (modulador, alvo) que compartilha uma enzima e aplica
    as regras R1–R4. As arestas resultantes são marcadas `origem=INFERIDA`
    e carregam o caminho que as justifica — sem isso, a resposta final não
    seria auditável e o sistema viraria uma caixa-preta clínica.
    """
    regras = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))["regras_mecanismo"]
    txt_regra = {r["id"]: r for r in regras}

    moduladores: dict[str, list[tuple[str, str, str]]] = {}  # enzima -> [(med, papel, intens)]
    alvos: dict[str, list[tuple[str, str, str]]] = {}

    for s, o, d in G.edges(data=True):
        rel = d["relacao"]
        if G.nodes[s].get("tipo") != TipoEntidade.MEDICAMENTO.value:
            continue
        if G.nodes[o].get("tipo") != TipoEntidade.ENZIMA.value:
            continue
        med, enz = G.nodes[s]["nome"], G.nodes[o]["nome"]
        inten = d.get("intensidade", "moderada")
        if rel in (TipoRelacao.INIBE.value, TipoRelacao.INDUZ.value):
            moduladores.setdefault(enz, []).append((med, rel, inten))
        elif rel in (TipoRelacao.SUBSTRATO_DE.value, TipoRelacao.ATIVADO_POR.value):
            alvos.setdefault(enz, []).append((med, rel, inten))

    inferidas: list[Tripla] = []
    for enz, mods in moduladores.items():
        for med_a, rel_a, int_a in mods:
            if int_a not in _INTENSIDADE_RELEVANTE:
                continue
            for med_b, rel_b, int_b in alvos.get(enz, []):
                if med_a == med_b:
                    continue

                if rel_b == TipoRelacao.ATIVADO_POR.value and rel_a == TipoRelacao.INIBE.value:
                    regra = "R4_inibicao_xantina_oxidase" if enz == "XO" else "R3_inibicao_pro_farmaco"
                elif rel_a == TipoRelacao.INIBE.value:
                    regra = "R4_inibicao_xantina_oxidase" if enz == "XO" else "R1_inibicao_substrato"
                elif rel_a == TipoRelacao.INDUZ.value:
                    if int_a != "forte":
                        continue  # R2 exige indução forte
                    regra = "R2_inducao_substrato"
                else:
                    continue

                sev = _severidade_mecanistica(rel_a, int_a, rel_b, int_b)
                if regra == "R4_inibicao_xantina_oxidase":
                    sev = Severidade.CONTRAINDICADO

                verbo = "inibe" if rel_a == TipoRelacao.INIBE.value else "induz"
                papel_b = (
                    "é bioativado por"
                    if rel_b == TipoRelacao.ATIVADO_POR.value
                    else "é metabolizado por"
                )
                mecanismo = (
                    f"{med_a} {verbo} a {enz} com intensidade {int_a}; "
                    f"{med_b} {papel_b} essa mesma via ({int_b}). "
                    f"{txt_regra[regra]['efeito']}"
                )
                if rel_b == TipoRelacao.ATIVADO_POR.value and rel_a == TipoRelacao.INIBE.value:
                    efeito = (
                        f"Perda de eficácia de {med_b}: sem a {enz} ativa, o fármaco não é "
                        f"convertido em seu metabólito ativo."
                    )
                    manejo = (
                        f"Substituir {med_a} por alternativa que não iniba a {enz}, ou trocar "
                        f"{med_b} por fármaco que não dependa dessa bioativação."
                    )
                elif rel_a == TipoRelacao.INIBE.value:
                    efeito = (
                        f"Acúmulo de {med_b} por redução da depuração, com risco de toxicidade "
                        f"dose-dependente."
                    )
                    manejo = (
                        f"Evitar a associação, reduzir a dose de {med_b} ou monitorar "
                        f"concentração/efeito enquanto durar o uso de {med_a}."
                    )
                else:
                    efeito = f"Queda da concentração plasmática de {med_b} e risco de falha terapêutica."
                    manejo = (
                        f"Monitorar a resposta a {med_b} e considerar ajuste de dose ou "
                        f"substituição de {med_a}."
                    )

                inferidas.append(
                    Tripla(
                        sujeito=med_a,
                        tipo_sujeito=TipoEntidade.MEDICAMENTO,
                        relacao=TipoRelacao.INTERAGE_COM,
                        objeto=med_b,
                        tipo_objeto=TipoEntidade.MEDICAMENTO,
                        chunk_id=None,
                        origem=OrigemAresta.INFERIDA,
                        atributos={
                            "severidade": sev.value,
                            "mecanismo": mecanismo,
                            "efeito": efeito,
                            "manejo": manejo,
                            "evidencia": "moderada",
                            "regra": regra,
                            "enzima": enz,
                        },
                    )
                )
    return inferidas


# --------------------------------------------------------------------------
# Consulta: caminhos de interação entre dois fármacos
# --------------------------------------------------------------------------


def _chunks_da_aresta(G: nx.MultiDiGraph, u: str, v: str, relacao: str) -> list[str]:
    out = []
    for _, _, d in G.edges(nbunch=[u], data=True):
        pass
    for k, d in G.get_edge_data(u, v, default={}).items():
        if d.get("relacao") == relacao and d.get("chunk_id"):
            out.append(d["chunk_id"])
    return out


def _sev(d: dict) -> Severidade:
    try:
        return Severidade(d.get("severidade", "moderada"))
    except ValueError:
        return Severidade.MODERADA


def analisar_par(G: nx.MultiDiGraph, med_a: str, med_b: str) -> list[Achado]:
    """Todos os caminhos de interação entre dois princípios ativos.

    Retorna achados ordenados por severidade (decrescente) e, em empate,
    pelo menor número de saltos — a evidência mais direta primeiro.
    """
    na = no_id(TipoEntidade.MEDICAMENTO, med_a)
    nb = no_id(TipoEntidade.MEDICAMENTO, med_b)
    if na not in G or nb not in G:
        return []

    achados: list[Achado] = []

    def classes_de(n: str) -> list[str]:
        return [
            v for _, v, d in G.out_edges(n, data=True)
            if d["relacao"] == TipoRelacao.E_DA_CLASSE.value
        ]

    # ---- [1] direta (asserida ou inferida), nos dois sentidos ----
    for u, v in ((na, nb), (nb, na)):
        for d in G.get_edge_data(u, v, default={}).values():
            if d["relacao"] != TipoRelacao.INTERAGE_COM.value:
                continue
            inferida = d.get("origem") == OrigemAresta.INFERIDA.value
            achados.append(
                Achado(
                    tipo="mecanismo" if inferida else "direta",
                    a=G.nodes[u]["nome"], b=G.nodes[v]["nome"],
                    severidade=_sev(d),
                    mecanismo=d.get("mecanismo", ""),
                    efeito=d.get("efeito", ""),
                    manejo=d.get("manejo", ""),
                    evidencia=d.get("evidencia", "moderada"),
                    caminho=(
                        [u, no_id(TipoEntidade.ENZIMA, d["enzima"]), v]
                        if inferida and "enzima" in d
                        else [u, v]
                    ),
                    chunks=[d["chunk_id"]] if d.get("chunk_id") else _chunks_mecanismo(G, u, v, d),
                    regra=d.get("regra"),
                )
            )

    # ---- [2] fármaco -> classe do outro ----
    for origem, destino in ((na, nb), (nb, na)):
        for cls in classes_de(destino):
            for d in G.get_edge_data(origem, cls, default={}).values():
                if d["relacao"] != TipoRelacao.INTERAGE_COM_CLASSE.value:
                    continue
                achados.append(
                    Achado(
                        tipo="classe",
                        a=G.nodes[origem]["nome"], b=G.nodes[destino]["nome"],
                        severidade=_sev(d),
                        mecanismo=d.get("mecanismo", ""),
                        efeito=d.get("efeito", ""),
                        manejo=d.get("manejo", ""),
                        evidencia=d.get("evidencia", "moderada"),
                        caminho=[origem, cls, destino],
                        chunks=sorted(
                            {d["chunk_id"]} if d.get("chunk_id") else set()
                        )
                        + _chunks_da_aresta(G, destino, cls, TipoRelacao.E_DA_CLASSE.value),
                    )
                )

    # ---- [3] classe <-> classe ----
    for ca in classes_de(na):
        for cb in classes_de(nb):
            for u, v in ((ca, cb), (cb, ca)):
                for d in G.get_edge_data(u, v, default={}).values():
                    if d["relacao"] != TipoRelacao.INTERAGE_COM_CLASSE.value:
                        continue
                    achados.append(
                        Achado(
                            tipo="classe_classe",
                            a=G.nodes[na]["nome"], b=G.nodes[nb]["nome"],
                            severidade=_sev(d),
                            mecanismo=d.get("mecanismo", ""),
                            efeito=d.get("efeito", ""),
                            manejo=d.get("manejo", ""),
                            evidencia=d.get("evidencia", "moderada"),
                            caminho=[na, ca, cb, nb] if u == ca else [na, ca, cb, nb],
                            chunks=(
                                ([d["chunk_id"]] if d.get("chunk_id") else [])
                                + _chunks_da_aresta(G, na, ca, TipoRelacao.E_DA_CLASSE.value)
                                + _chunks_da_aresta(G, nb, cb, TipoRelacao.E_DA_CLASSE.value)
                            ),
                        )
                    )

    return _dedup_e_ordenar(achados)


def _chunks_mecanismo(G: nx.MultiDiGraph, u: str, v: str, d: dict) -> list[str]:
    """Evidência de uma aresta inferida: as duas bulas de metabolismo."""
    enz = d.get("enzima")
    if not enz:
        return []
    ne = no_id(TipoEntidade.ENZIMA, enz)
    chunks: list[str] = []
    for no in (u, v):
        for _, alvo, ed in G.out_edges(no, data=True):
            if alvo == ne and ed.get("chunk_id"):
                chunks.append(ed["chunk_id"])
    return sorted(set(chunks))


def _dedup_e_ordenar(achados: list[Achado]) -> list[Achado]:
    vistos: set[tuple] = set()
    saida: list[Achado] = []
    for a in achados:
        chave = (a.tipo, a.regra, tuple(sorted((a.a, a.b))), a.severidade.value)
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(a)
    ordem_tipo = {"direta": 0, "mecanismo": 1, "classe": 2, "classe_classe": 3}
    return sorted(
        saida,
        key=lambda x: (-x.severidade.peso, x.n_saltos, ordem_tipo.get(x.tipo, 9)),
    )


# --------------------------------------------------------------------------
# Vizinhança (usada na recuperação — passo 5)
# --------------------------------------------------------------------------


def vizinhanca(G: nx.MultiDiGraph, sementes: Iterable[str], hops: int = 2) -> set[str]:
    """Nós alcançáveis a até `hops` arestas das sementes (grafo não dirigido).

    Arestas MENCIONA são ignoradas na expansão: elas ligam chunks a
    entidades e, se percorridas, inundariam a vizinhança com o corpus
    inteiro.
    """
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d["relacao"] == TipoRelacao.MENCIONA.value:
            continue
        H.add_edge(u, v)

    alcancados: set[str] = set()
    fronteira = {s for s in sementes if s in H}
    alcancados |= fronteira
    for _ in range(hops):
        nova: set[str] = set()
        for n in fronteira:
            nova |= set(H.neighbors(n))
        nova -= alcancados
        alcancados |= nova
        fronteira = nova
        if not fronteira:
            break
    return alcancados


def chunks_de_entidades(G: nx.MultiDiGraph, nos: Iterable[str]) -> dict[str, set[str]]:
    """chunk_id -> conjunto de nós daquele conjunto que o chunk menciona."""
    alvo = set(nos)
    mapa: dict[str, set[str]] = {}
    for u, v, d in G.edges(data=True):
        if d["relacao"] != TipoRelacao.MENCIONA.value:
            continue
        if v in alvo:
            mapa.setdefault(u.replace(f"{TipoEntidade.CHUNK.value}::", ""), set()).add(v)
    return mapa


# --------------------------------------------------------------------------
# Persistência
# --------------------------------------------------------------------------


def salvar_grafo(G: nx.MultiDiGraph) -> None:
    garantir_diretorios()
    dados = nx.node_link_data(G, edges="links")
    GRAPH_JSON.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")


def carregar_grafo() -> nx.MultiDiGraph:
    dados = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    return nx.node_link_graph(dados, directed=True, multigraph=True, edges="links")


def estatisticas(G: nx.MultiDiGraph) -> dict[str, Any]:
    por_tipo: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        por_tipo[d.get("tipo", "?")] = por_tipo.get(d.get("tipo", "?"), 0) + 1
    por_rel: dict[str, int] = {}
    inferidas = 0
    for _, _, d in G.edges(data=True):
        por_rel[d["relacao"]] = por_rel.get(d["relacao"], 0) + 1
        if d.get("origem") == OrigemAresta.INFERIDA.value:
            inferidas += 1
    return {
        "nos": G.number_of_nodes(),
        "arestas": G.number_of_edges(),
        "nos_por_tipo": dict(sorted(por_tipo.items(), key=lambda x: -x[1])),
        "arestas_por_relacao": dict(sorted(por_rel.items(), key=lambda x: -x[1])),
        "arestas_inferidas": inferidas,
    }


def construir_completo(triplas: Iterable[Tripla], salvar: bool = True) -> nx.MultiDiGraph:
    """Constrói o grafo e aplica o motor de inferência."""
    triplas = list(triplas)
    G = construir_grafo(triplas)
    inferidas = inferir_mecanismos(G)
    for t in inferidas:
        s = no_id(t.tipo_sujeito, t.sujeito)
        o = no_id(t.tipo_objeto, t.objeto)
        G.add_edge(s, o, key=f"{t.relacao.value}|inferida|{t.atributos.get('enzima')}",
                   relacao=t.relacao.value, chunk_id=None,
                   origem=OrigemAresta.INFERIDA.value, **t.atributos)
    if salvar:
        salvar_grafo(G)
    return G


if __name__ == "__main__":
    from .extract import carregar_triplas

    G = construir_completo(carregar_triplas())
    est = estatisticas(G)
    print(f"grafo: {est['nos']} nós, {est['arestas']} arestas "
          f"({est['arestas_inferidas']} inferidas por mecanismo)")
    for k, v in est["nos_por_tipo"].items():
        print(f"  nó  {k:<14} {v}")
    for k, v in est["arestas_por_relacao"].items():
        print(f"  rel {k:<22} {v}")

    print("\n— exemplo de raciocínio multi-hop —")
    for a, b in [("varfarina", "ibuprofeno"), ("claritromicina", "sinvastatina"),
                 ("omeprazol", "clopidogrel")]:
        print(f"\n{a} + {b}:")
        for ach in analisar_par(G, a, b):
            caminho = " → ".join(n.split("::")[-1] for n in ach.caminho)
            print(f"  [{ach.tipo}/{ach.severidade.value}] {caminho}")
            print(f"     {ach.efeito}")
