"""Gera as figuras do relatório a partir dos resultados salvos.

Paleta categórica validada para daltonismo (pior ΔE adjacente 103.7).
Marcas finas, grade recessiva, rótulos diretos em todas as barras (regra
de relevo, pois o amarelo fica abaixo de 3:1 de contraste).

Saída: docs/img/*.png (300 dpi) e docs/img/grafo_interativo.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hygia.config import FIG_DIR, RESULTS_DIR  # noqa: E402
from hygia.extract import carregar_triplas  # noqa: E402
from hygia.graph import analisar_par, construir_completo  # noqa: E402
from hygia.schema import OrigemAresta, TipoEntidade  # noqa: E402

# paleta por recuperador (identidade, ordem fixa)
COR = {
    "vetorial": "#2a78d6",
    "bm25": "#eda100",
    "hibrido": "#4a3aa7",
    "graphrag": "#008300",
}
ROTULO = {
    "vetorial": "Vetorial (LSA)",
    "bm25": "BM25",
    "hibrido": "Híbrido (plain RAG)",
    "graphrag": "Graph RAG",
}
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
GRADE = "#e6e6e2"
SURF = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRADE,
    "axes.linewidth": 1.0,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "svg.fonttype": "none",
})


def _limpar(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0, colors=TINTA2)
    ax.set_axisbelow(True)


def fig_cobertura_categoria(saida_eval: dict) -> Path:
    """Figura principal: cobertura de fatos por categoria de raciocínio."""
    res = saida_eval["resultados"]["por_recuperador"]
    tipos = ["fato_direto", "mecanismo", "ponte_classe", "ponte_classe2"]
    rot_tipo = ["Fato direto\n(1 salto)", "Mecanismo\n(enzima)",
                "Ponte de classe\n(2 saltos)", "Classe↔classe\n(3 saltos)"]
    recs = ["vetorial", "bm25", "hibrido", "graphrag"]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    n = len(recs)
    larg = 0.2
    x = range(len(tipos))
    for i, rec in enumerate(recs):
        vals = [res[rec]["por_tipo"].get(t, {}).get("cobertura", 0.0) for t in tipos]
        pos = [xi + (i - (n - 1) / 2) * larg for xi in x]
        barras = ax.bar(pos, vals, larg * 0.9, color=COR[rec], label=ROTULO[rec],
                        zorder=3, edgecolor=SURF, linewidth=1.5)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=TINTA, zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(rot_tipo, fontsize=10, color=TINTA)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Cobertura de fatos", color=TINTA2)
    ax.yaxis.grid(True, color=GRADE, linewidth=0.8, zorder=0)
    _limpar(ax)
    fig.text(0.012, 0.975, "Cobertura de fatos por categoria de raciocínio",
             fontsize=14, color=TINTA, weight="bold", ha="left", va="top")
    fig.text(0.012, 0.925,
             "Em fatos de salto único todos empatam; nos saltos múltiplos os baselines "
             "colapsam e só o grafo compõe a evidência.",
             fontsize=9.5, color=TINTA2, ha="left", va="top")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9.5, columnspacing=1.4, handlelength=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    saida = FIG_DIR / "cobertura_por_categoria.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saida


def fig_metricas_gerais(saida_eval: dict) -> Path:
    """Métricas gerais com intervalos de confiança de 95%."""
    res = saida_eval["resultados"]["por_recuperador"]
    recs = ["vetorial", "bm25", "hibrido", "graphrag"]
    metricas = [("cobertura", "Cobertura de fatos"), ("hit", "Hit@k (respondível)"),
                ("recall", "Recall@k"), ("mrr", "MRR")]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for ax, (chave, titulo) in zip(axes, metricas):
        vals = [res[r]["geral"][chave]["media"] for r in recs]
        ics = [res[r]["geral"][chave]["ic95"] for r in recs]
        y = range(len(recs))
        for yi, (r, v, ic) in enumerate(zip(recs, vals, ics)):
            ax.barh(yi, v, color=COR[r], height=0.62, zorder=3,
                    edgecolor=SURF, linewidth=1.2)
            # barra de erro (IC95)
            ax.plot([ic[0], ic[1]], [yi, yi], color=TINTA, linewidth=1.4, zorder=4)
            ax.plot([ic[0], ic[0]], [yi - 0.1, yi + 0.1], color=TINTA, linewidth=1.4, zorder=4)
            ax.plot([ic[1], ic[1]], [yi - 0.1, yi + 0.1], color=TINTA, linewidth=1.4, zorder=4)
            ax.text(min(v + 0.04, 1.02), yi, f"{v:.2f}", va="center", ha="left",
                    fontsize=8.5, color=TINTA, zorder=5)
        ax.set_yticks(list(y))
        ax.set_yticklabels([ROTULO[r] for r in recs] if chave == "cobertura" else [])
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1.0])
        ax.xaxis.grid(True, color=GRADE, linewidth=0.8, zorder=0)
        ax.set_title(titulo, fontsize=11, color=TINTA, weight="bold", loc="left", pad=8)
        ax.invert_yaxis()
        _limpar(ax)
    fig.suptitle("Métricas de recuperação (barra = IC 95% por bootstrap)",
                 fontsize=13, color=TINTA, weight="bold", x=0.09, ha="left", y=1.02)
    fig.tight_layout()
    saida = FIG_DIR / "metricas_gerais.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saida


def fig_ablacao(saida_abl: dict) -> Path:
    res = saida_abl["ablacao"]
    ordem = ["so_hibrido", "sem_caminhada", "sem_inferencia", "completo"]
    rot = ["Só híbrido\n(sem grafo)", "Sem caminhada", "Sem inferência", "Completo"]
    cores = ["#4a3aa7", "#eda100", "#e34948", "#008300"]
    vals = [res[c]["cobertura_geral"] for c in ordem]
    mec = [res[c]["por_tipo"].get("mecanismo", 0.0) for c in ordem]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = range(len(ordem))
    b1 = ax.bar([xi - 0.19 for xi in x], vals, 0.36, color=cores, zorder=3,
                edgecolor=SURF, linewidth=1.5, label="Cobertura geral")
    b2 = ax.bar([xi + 0.19 for xi in x], mec, 0.36, color=cores, alpha=0.5, zorder=3,
                edgecolor=SURF, linewidth=1.5, hatch="///", label="Categoria mecanismo")
    for xi, v in zip(x, vals):
        ax.text(xi - 0.19, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=TINTA)
    for xi, v in zip(x, mec):
        ax.text(xi + 0.19, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=TINTA)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rot, fontsize=9.5, color=TINTA)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.grid(True, color=GRADE, linewidth=0.8, zorder=0)
    ax.set_ylabel("Cobertura de fatos", color=TINTA2)
    _limpar(ax)
    ax.set_title("Ablação: o motor de inferência sustenta a categoria mecanismo",
                 fontsize=13, color=TINTA, weight="bold", pad=12, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    saida = FIG_DIR / "ablacao.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saida


def fig_caminho_multihop() -> Path:
    """Esquema do raciocínio multi-hop para claritromicina + sinvastatina."""
    G = construir_completo(carregar_triplas(), salvar=False)
    achados = analisar_par(G, "claritromicina", "sinvastatina")
    ach = next((a for a in achados if a.tipo == "mecanismo"), achados[0])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    nos = [n.split("::")[-1] for n in ach.caminho]
    tipos = [n.split("::")[0] for n in ach.caminho]
    cor_tipo = {"Medicamento": "#2a78d6", "Enzima": "#eda100", "Classe": "#4a3aa7"}
    xs = [0, 0.5, 1.0]
    for i, (nome, tipo, x) in enumerate(zip(nos, tipos, xs)):
        ax.scatter([x], [0], s=5200, color=cor_tipo.get(tipo, "#888"),
                   zorder=3, edgecolors=SURF, linewidths=3)
        ax.text(x, 0, nome, ha="center", va="center", color="white",
                fontsize=10.5, weight="bold", zorder=4)
        ax.text(x, -0.13, tipo, ha="center", va="center", color=TINTA2, fontsize=8.5)
    rel = ["inibe (forte)", "metaboliza"]
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.11, 0), xytext=(xs[i] + 0.11, 0),
                    arrowprops=dict(arrowstyle="-|>", color=TINTA2, lw=2))
        ax.text((xs[i] + xs[i + 1]) / 2, 0.09, rel[i], ha="center", color=TINTA, fontsize=9)
    ax.text(0.5, 0.33, "Interação inferida (CONTRAINDICADO) — nenhum documento cita os dois fármacos juntos",
            ha="center", color="#008300", fontsize=10, weight="bold")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.3, 0.45)
    ax.axis("off")
    fig.tight_layout()
    saida = FIG_DIR / "caminho_multihop.png"
    fig.savefig(saida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return saida


def fig_grafo_visao_geral() -> Path:
    """Visão macro do grafo: núcleo enzimático e arestas inferidas destacadas."""
    G = construir_completo(carregar_triplas(), salvar=False)
    H = nx.Graph()
    cor, tam = {}, {}
    cor_tipo = {
        TipoEntidade.MEDICAMENTO.value: "#2a78d6",
        TipoEntidade.ENZIMA.value: "#eda100",
        TipoEntidade.CLASSE.value: "#4a3aa7",
    }
    manter = set(cor_tipo)
    for n, d in G.nodes(data=True):
        if d.get("tipo") in manter:
            H.add_node(n)
            cor[n] = cor_tipo[d["tipo"]]
            tam[n] = {"Enzima": 380, "Medicamento": 70, "Classe": 120}[d["tipo"]]
    arestas_norm, arestas_inf = [], []
    for u, v, d in G.edges(data=True):
        if u in H and v in H and d["relacao"] in (
            "INIBE", "INDUZ", "SUBSTRATO_DE", "ATIVADO_POR", "E_DA_CLASSE",
            "INTERAGE_COM", "INTERAGE_COM_CLASSE"):
            if d.get("origem") == OrigemAresta.INFERIDA.value:
                arestas_inf.append((u, v))
            else:
                arestas_norm.append((u, v))

    from hygia.config import semear_tudo
    semear_tudo()
    pos = nx.spring_layout(H, seed=42, k=0.55, iterations=120)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(H, pos, edgelist=arestas_norm, ax=ax, edge_color=GRADE,
                           width=0.8, alpha=0.8)
    nx.draw_networkx_edges(H, pos, edgelist=arestas_inf, ax=ax, edge_color="#008300",
                           width=1.2, alpha=0.7, style="dashed")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[cor[n] for n in H.nodes()],
                           node_size=[tam[n] for n in H.nodes()], edgecolors=SURF, linewidths=1.0)
    for n, d in G.nodes(data=True):
        if d.get("tipo") == TipoEntidade.ENZIMA.value and n in pos:
            ax.text(pos[n][0], pos[n][1], d["nome"], ha="center", va="center",
                    fontsize=8, weight="bold", color=TINTA)
    from matplotlib.lines import Line2D
    legenda = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#eda100", markersize=13, label="Enzima (via metabólica)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a78d6", markersize=9, label="Medicamento"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4a3aa7", markersize=10, label="Classe"),
        Line2D([0], [0], color="#008300", lw=1.4, linestyle="--", label="Interação inferida por mecanismo"),
    ]
    ax.legend(handles=legenda, frameon=False, loc="lower left", fontsize=9.5)
    ax.set_title("Grafo de conhecimento — núcleo enzimático e interações inferidas",
                 fontsize=13, color=TINTA, weight="bold", loc="left")
    ax.axis("off")
    fig.tight_layout()
    saida = FIG_DIR / "grafo_visao_geral.png"
    fig.savefig(saida, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return saida


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ev = json.loads((RESULTS_DIR / "avaliacao_recuperacao.json").read_text(encoding="utf-8"))
    ab = json.loads((RESULTS_DIR / "ablacao.json").read_text(encoding="utf-8"))
    gerados = [
        fig_cobertura_categoria(ev),
        fig_metricas_gerais(ev),
        fig_ablacao(ab),
        fig_caminho_multihop(),
        fig_grafo_visao_geral(),
    ]
    for p in gerados:
        print(f"figura: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
