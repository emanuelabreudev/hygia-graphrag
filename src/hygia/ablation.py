"""Estudo de ablação — isola a contribuição de cada componente do Graph RAG.

Responde: o ganho vem da caminhada no grafo, do motor de inferência, ou
de ambos? Remove-se um componente por vez e mede-se a cobertura de fatos
por categoria de raciocínio.

Configurações:
  completo         — Graph RAG integral (caminhada + inferência mecanística)
  sem_inferencia   — grafo só com arestas asseridas (sem regras R1–R4);
                     mede quanto do ganho depende de DERIVAR interações
  sem_caminhada    — sem expansão de vizinhança; recuperação textual + achados
  so_hibrido       — baseline de referência (nenhum uso de grafo)
"""

from __future__ import annotations

import json
from dataclasses import replace
from statistics import mean

from .config import DEFAULT_RETRIEVAL, RESULTS_DIR, garantir_diretorios
from .extract import carregar_triplas
from .gold import carregar_gold
from .graph import construir_completo, construir_grafo
from .index import Indice
from .resolver import ResolvedorEntidades
from .retriever import GraphRAGRetriever, HibridoRetriever
from .schema import TipoEntidade


def _cobertura(res, p, k):
    if p.tipo in ("mecanismo", "ponte_classe", "ponte_classe2"):
        meds = {n for t, n in p.entidades if TipoEntidade(t) is TipoEntidade.MEDICAMENTO}
        for a in res.achados_grafo:
            if {a.a, a.b} == meds:
                return 1.0
    return 1.0 if set(p.chunks_relevantes) <= set(res.chunks[:k]) else 0.0


def executar(salvar: bool = True) -> dict:
    cfg = DEFAULT_RETRIEVAL
    indice = Indice.carregar()
    triplas = carregar_triplas()
    resolvedor = ResolvedorEntidades()
    gold = carregar_gold()

    g_completo = construir_completo(triplas, salvar=False)   # com inferência
    g_asserido = construir_grafo(triplas)                    # sem inferência

    configs = {
        "so_hibrido": HibridoRetriever(indice, cfg),
        "sem_caminhada": GraphRAGRetriever(indice, g_completo, replace(cfg, max_hops=0)),
        "sem_inferencia": GraphRAGRetriever(indice, g_asserido, cfg),
        "completo": GraphRAGRetriever(indice, g_completo, cfg),
    }

    resultados: dict[str, dict] = {}
    for nome, r in configs.items():
        por_tipo: dict[str, list[float]] = {}
        geral: list[float] = []
        for p in gold:
            ents = resolvedor.resolver(p.texto)
            res = r.recuperar(p.texto, ents)
            cob = _cobertura(res, p, cfg.top_k)
            por_tipo.setdefault(p.tipo, []).append(cob)
            geral.append(cob)
        resultados[nome] = {
            "cobertura_geral": round(mean(geral), 4),
            "por_tipo": {t: round(mean(v), 4) for t, v in sorted(por_tipo.items())},
        }

    saida = {"ablacao": resultados}
    if salvar:
        garantir_diretorios()
        (RESULTS_DIR / "ablacao.json").write_text(
            json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return saida


def imprimir(saida: dict) -> None:
    res = saida["ablacao"]
    tipos = ["fato_direto", "mecanismo", "ponte_classe", "ponte_classe2"]
    ordem = ["so_hibrido", "sem_caminhada", "sem_inferencia", "completo"]
    print(f"\n{'='*72}\nESTUDO DE ABLAÇÃO — cobertura de fatos por configuração\n{'='*72}")
    print("configuração".ljust(16) + "geral".rjust(9) + "".join(t[:12].rjust(14) for t in tipos))
    for nome in ordem:
        d = res[nome]
        linha = nome.ljust(16) + f"{d['cobertura_geral']:.2f}".rjust(9)
        for t in tipos:
            linha += f"{d['por_tipo'].get(t, 0.0):.2f}".rjust(14)
        print(linha)


if __name__ == "__main__":
    imprimir(executar())
