"""Pipeline end-to-end reprodutível: da semente aos resultados.

Executa, em ordem, os sete passos do Graph RAG e grava todos os artefatos.
Determinístico (semente fixa em hygia.config.SEED). Roda sem chave de API:
a extração e a avaliação usam o extrator por gazetteer e o gerador offline.

Uso:
    python scripts/pipeline.py                # gazetteer (padrão, roda no CI)
    python scripts/pipeline.py --extrator llm # extração via Claude (requer API key)
    python scripts/pipeline.py --rapido       # pula figuras (smoke-test)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hygia import ablation, corpus, eval as avaliacao, extract, gold, index  # noqa: E402
from hygia.config import semear_tudo  # noqa: E402
from hygia.graph import construir_completo, estatisticas  # noqa: E402


def _passo(n: int, titulo: str) -> None:
    print(f"\n[{n}/7] {titulo}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extrator", choices=["gazetteer", "llm"], default="gazetteer")
    ap.add_argument("--rapido", action="store_true", help="pula geração de figuras")
    args = ap.parse_args()

    semear_tudo()
    t0 = time.time()

    print("=" * 72)
    print("HYGIA — Graph RAG para segurança medicamentosa | pipeline reprodutível")
    print("=" * 72)

    _passo(1, "Ontologia (esquema de entidades e relações) — declarativa em hygia.schema")
    from hygia.schema import ASSINATURAS
    print(f"      {len(ASSINATURAS)} assinaturas de tripla permitidas")

    _passo(2, "Corpus documental (bulas + monografias)")
    chunks = corpus.construir_corpus()
    print(f"      {len(chunks)} chunks | {len({c.doc_id for c in chunks})} documentos")

    _passo(3, "Extração de triplas (texto → conhecimento estruturado)")
    triplas = extract.executar(args.extrator)
    metr = extract.avaliar_extracao(triplas)
    print(f"      {len(triplas)} triplas | extração P={metr['precisao']} "
          f"R={metr['recall']} F1={metr['f1']}")

    _passo(4, "Grafo de conhecimento + motor de inferência mecanística")
    grafo = construir_completo(triplas)
    est = estatisticas(grafo)
    print(f"      {est['nos']} nós | {est['arestas']} arestas "
          f"({est['arestas_inferidas']} interações inferidas por mecanismo)")

    _passo(5, "Índice vetorial (LSA) + léxico (BM25)")
    idx = index.executar()
    print(f"      LSA d={idx.vetorial.matriz.shape[1]} | vocabulário BM25={len(idx.bm25.df)}")

    _passo(6, "Conjunto de avaliação (gold set)")
    perguntas = gold.gerar_gold()
    print(f"      {len(perguntas)} perguntas em 4 categorias de raciocínio")

    _passo(7, "Avaliação comparativa + ablação")
    res_eval = avaliacao.executar()
    res_abl = ablation.executar()
    avaliacao.imprimir(res_eval)
    ablation.imprimir(res_abl)

    if not args.rapido:
        print("\n[+] Gerando figuras…")
        import scripts.figuras as figuras  # noqa: E402
        figuras.main()

    print(f"\n{'='*72}\nConcluído em {time.time() - t0:.1f}s. "
          f"Artefatos em resultados/ e docs/img/.\n{'='*72}")
    return 0


if __name__ == "__main__":
    # permite 'import scripts.figuras'
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
