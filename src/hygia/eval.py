"""PASSO 7 — Avaliação e comparação Graph RAG vs. baselines.

Métricas de RECUPERAÇÃO (independentes do gerador, portanto rodam sem
API key):

  recall@k     — fração dos chunks-gabarito recuperados no top-k
  precision@k  — fração do top-k que é gabarito
  MRR          — inverso da posição do primeiro chunk-gabarito
  hit@k        — a resposta é *respondível* (todos os chunks-gabarito no top-k)
  cobertura_fatos — todos os fatos necessários foram reunidos? (métrica-chave
                    para multi-hop: mede se a EVIDÊNCIA para compor a resposta
                    foi trazida, não só se um chunk relevante apareceu)

Rigor estatístico:
  - Resultados por categoria de raciocínio (fato_direto, ponte_classe, ...).
  - Intervalos de confiança de 95% por bootstrap sobre as perguntas.
  - Teste de significância pareado (bootstrap) entre Graph RAG e o melhor
    baseline, para hit@k e cobertura de fatos.

A geração (qualidade textual) é avaliada à parte em avaliar_geracao()
quando há ANTHROPIC_API_KEY, com Claude como juiz.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Callable

import numpy as np

from .config import DEFAULT_RETRIEVAL, RESULTS_DIR, RetrievalConfig, rng, garantir_diretorios
from .corpus import carregar_chunks
from .extract import carregar_triplas
from .gold import Pergunta, carregar_gold
from .graph import construir_completo
from .index import Indice
from .resolver import ResolvedorEntidades
from .retriever import Resultado, construir_recuperadores
from .schema import TipoEntidade


# --------------------------------------------------------------------------
# Métricas por pergunta
# --------------------------------------------------------------------------


def _recall_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    topo = set(recuperados[:k])
    return len(topo & relevantes) / len(relevantes) if relevantes else 0.0


def _precision_at_k(recuperados: list[str], relevantes: set[str], k: int) -> float:
    topo = recuperados[:k]
    if not topo:
        return 0.0
    return len([c for c in topo if c in relevantes]) / len(topo)


def _mrr(recuperados: list[str], relevantes: set[str]) -> float:
    for i, c in enumerate(recuperados):
        if c in relevantes:
            return 1.0 / (i + 1)
    return 0.0


def _hit_completo(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """1.0 se TODOS os chunks-gabarito estão no top-k. Mede respondibilidade."""
    return 1.0 if relevantes <= set(recuperados[:k]) else 0.0


def _cobertura_fatos(res: Resultado, p: Pergunta, k: int) -> float:
    """Todos os fatos necessários foram reunidos, por texto OU por grafo?

    Para o Graph RAG, um achado estruturado entre os dois fármacos da
    pergunta conta como cobertura completa — a evidência foi COMPOSTA,
    ainda que os chunks não estejam todos no top-k textual. Esta é a
    métrica que captura a vantagem real do raciocínio em grafo.
    """
    if p.tipo in ("mecanismo", "ponte_classe", "ponte_classe2"):
        meds = {nome for t, nome in _ents(p) if t is TipoEntidade.MEDICAMENTO}
        for a in res.achados_grafo:
            if {a.a, a.b} == meds:
                return 1.0
    return _hit_completo(res.chunks, set(p.chunks_relevantes), k)


def _ents(p: Pergunta) -> list[tuple[TipoEntidade, str]]:
    return [(TipoEntidade(t), n) for t, n in p.entidades]


@dataclass
class LinhaAvaliacao:
    pergunta: str
    tipo: str
    recuperador: str
    recall: float
    precision: float
    mrr: float
    hit: float
    cobertura: float


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------


def avaliar_recuperacao(
    cfg: RetrievalConfig = DEFAULT_RETRIEVAL, k: int | None = None
) -> list[LinhaAvaliacao]:
    k = k or cfg.top_k
    indice = Indice.carregar()
    grafo = construir_completo(carregar_triplas(), salvar=False)
    recuperadores = construir_recuperadores(indice, grafo, cfg)
    resolvedor = ResolvedorEntidades()
    gold = carregar_gold()

    linhas: list[LinhaAvaliacao] = []
    for p in gold:
        # o resolvedor extrai as âncoras do TEXTO da pergunta (mesma entrada p/ todos)
        ents = resolvedor.resolver(p.texto)
        rel = set(p.chunks_relevantes)
        for nome, r in recuperadores.items():
            res = r.recuperar(p.texto, ents)
            linhas.append(
                LinhaAvaliacao(
                    pergunta=p.id, tipo=p.tipo, recuperador=nome,
                    recall=_recall_at_k(res.chunks, rel, k),
                    precision=_precision_at_k(res.chunks, rel, k),
                    mrr=_mrr(res.chunks, rel),
                    hit=_hit_completo(res.chunks, rel, k),
                    cobertura=_cobertura_fatos(res, p, k),
                )
            )
    return linhas


# --------------------------------------------------------------------------
# Agregação + estatística
# --------------------------------------------------------------------------


def _bootstrap_ic(valores: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if not valores:
        return (0.0, 0.0)
    g = rng(7)
    arr = np.array(valores)
    medias = [g.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _teste_pareado(a: list[float], b: list[float], n_boot: int = 5000) -> float:
    """p-valor bootstrap (bicaudal) para H0: média(a) == média(b), pareado."""
    g = rng(11)
    d = np.array(a) - np.array(b)
    obs = d.mean()
    if np.allclose(d, 0):
        return 1.0
    centrado = d - obs
    conta = 0
    for _ in range(n_boot):
        amostra = g.choice(centrado, size=len(centrado), replace=True)
        if abs(amostra.mean()) >= abs(obs):
            conta += 1
    return (conta + 1) / (n_boot + 1)


def agregar(linhas: list[LinhaAvaliacao]) -> dict:
    por_rec: dict[str, list[LinhaAvaliacao]] = defaultdict(list)
    for l in linhas:
        por_rec[l.recuperador].append(l)

    resumo: dict[str, dict] = {}
    for rec, ls in por_rec.items():
        metr = {}
        for m in ("recall", "precision", "mrr", "hit", "cobertura"):
            vals = [getattr(l, m) for l in ls]
            lo, hi = _bootstrap_ic(vals)
            metr[m] = {"media": round(mean(vals), 4), "ic95": [round(lo, 4), round(hi, 4)]}
        # por categoria
        por_tipo: dict[str, dict] = {}
        tipos = sorted({l.tipo for l in ls})
        for t in tipos:
            sub = [l for l in ls if l.tipo == t]
            por_tipo[t] = {
                "n": len(sub),
                "cobertura": round(mean(l.cobertura for l in sub), 4),
                "hit": round(mean(l.hit for l in sub), 4),
                "recall": round(mean(l.recall for l in sub), 4),
            }
        resumo[rec] = {"geral": metr, "por_tipo": por_tipo}

    # significância: graphrag vs melhor baseline em cobertura e hit
    baselines = [r for r in por_rec if r != "graphrag"]
    testes = {}
    if "graphrag" in por_rec and baselines:
        idx_pergunta = lambda ls: {l.pergunta: l for l in ls}
        g_map = idx_pergunta(por_rec["graphrag"])
        for metrica in ("cobertura", "hit"):
            melhor_base, melhor_media = None, -1.0
            for b in baselines:
                mm = mean(getattr(l, metrica) for l in por_rec[b])
                if mm > melhor_media:
                    melhor_base, melhor_media = b, mm
            b_map = idx_pergunta(por_rec[melhor_base])
            perguntas = sorted(g_map)
            ga = [getattr(g_map[q], metrica) for q in perguntas]
            ba = [getattr(b_map[q], metrica) for q in perguntas]
            p = _teste_pareado(ga, ba)
            testes[metrica] = {
                "graphrag": round(mean(ga), 4),
                "melhor_baseline": melhor_base,
                "baseline_media": round(mean(ba), 4),
                "delta": round(mean(ga) - mean(ba), 4),
                "p_valor": round(p, 5),
                "significativo_5pct": p < 0.05,
            }
    return {"por_recuperador": resumo, "significancia": testes}


def executar(cfg: RetrievalConfig = DEFAULT_RETRIEVAL, salvar: bool = True) -> dict:
    linhas = avaliar_recuperacao(cfg)
    agg = agregar(linhas)
    saida = {
        "config": {
            "top_k": cfg.top_k, "n_componentes": cfg.n_componentes,
            "max_hops": cfg.max_hops, "rrf_k": cfg.rrf_k,
        },
        "n_perguntas": len({l.pergunta for l in linhas}),
        "resultados": agg,
        "linhas": [l.__dict__ for l in linhas],
    }
    if salvar:
        garantir_diretorios()
        (RESULTS_DIR / "avaliacao_recuperacao.json").write_text(
            json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return saida


def imprimir(saida: dict) -> None:
    print(f"\n{'='*70}\nAVALIAÇÃO DE RECUPERAÇÃO — {saida['n_perguntas']} perguntas, "
          f"top_k={saida['config']['top_k']}\n{'='*70}")
    res = saida["resultados"]["por_recuperador"]
    ordem = ["vetorial", "bm25", "hibrido", "graphrag"]
    print(f"\n{'recuperador':<12} {'cobertura':>18} {'hit@k':>16} {'recall@k':>16} {'MRR':>8}")
    for rec in ordem:
        if rec not in res:
            continue
        g = res[rec]["geral"]
        cob = g["cobertura"]; hit = g["hit"]; rc = g["recall"]
        print(f"{rec:<12} "
              f"{cob['media']:.3f} [{cob['ic95'][0]:.2f},{cob['ic95'][1]:.2f}]".rjust(18)
              + f"  {hit['media']:.3f} [{hit['ic95'][0]:.2f},{hit['ic95'][1]:.2f}]".rjust(16)
              + f"  {rc['media']:.3f}".rjust(10)
              + f"  {g['mrr']['media']:.3f}".rjust(8))

    print("\nCOBERTURA DE FATOS por categoria de raciocínio:")
    tipos = ["fato_direto", "mecanismo", "ponte_classe", "ponte_classe2"]
    cab = "categoria".ljust(16) + "".join(r.rjust(11) for r in ordem)
    print(cab)
    for t in tipos:
        linha = t.ljust(16)
        for rec in ordem:
            v = res.get(rec, {}).get("por_tipo", {}).get(t, {}).get("cobertura")
            linha += (f"{v:.2f}".rjust(11)) if v is not None else "—".rjust(11)
        print(linha)

    sig = saida["resultados"]["significancia"]
    if sig:
        print("\nSIGNIFICÂNCIA (Graph RAG vs. melhor baseline, bootstrap pareado):")
        for m, d in sig.items():
            estrela = " *" if d["significativo_5pct"] else ""
            print(f"  {m:<10} graphrag={d['graphrag']:.3f}  "
                  f"{d['melhor_baseline']}={d['baseline_media']:.3f}  "
                  f"Δ={d['delta']:+.3f}  p={d['p_valor']:.4f}{estrela}")


if __name__ == "__main__":
    imprimir(executar())
