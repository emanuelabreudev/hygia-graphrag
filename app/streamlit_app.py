"""HYGIA — interface de verificação de interações medicamentosas.

Três abas:
  1. Verificador — informe 2+ medicamentos (por princípio ativo OU marca) e
     veja as interações com o CAMINHO de raciocínio no grafo, gravidade e
     conduta. Compara Graph RAG × plain RAG lado a lado.
  2. Explorar o grafo — subgrafo interativo em torno de um fármaco.
  3. Benchmark — os resultados da avaliação comparativa.

Roda sem chave de API (gerador offline determinístico). Se ANTHROPIC_API_KEY
estiver definida, a resposta em linguagem natural é redigida por Claude.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hygia.answer import gerar_resposta_llm, gerar_resposta_offline
from hygia.corpus import carregar_chunks, construir_corpus
from hygia.extract import Gazetteer, carregar_triplas, executar as extrair
from hygia.graph import analisar_par, construir_completo, estatisticas
from hygia.index import Indice
from hygia.resolver import ResolvedorEntidades
from hygia.retriever import GraphRAGRetriever, HibridoRetriever
from hygia.schema import Severidade, TipoEntidade, no_id

st.set_page_config(page_title="HYGIA · Segurança Medicamentosa",
                   page_icon="⚕️", layout="wide")

COR_SEV = {
    "contraindicado": "#b3261e",
    "grave": "#e8710a",
    "moderada": "#f2b100",
    "leve": "#1e8e3e",
}
ICONE_SEV = {"contraindicado": "🚫", "grave": "⚠️", "moderada": "⚡", "leve": "ℹ️"}


@st.cache_resource(show_spinner="Carregando base de conhecimento…")
def carregar_tudo():
    if not (ROOT / "data" / "corpus" / "chunks.json").exists():
        construir_corpus()
    try:
        triplas = carregar_triplas()
    except FileNotFoundError:
        triplas = extrair("gazetteer")
    grafo = construir_completo(triplas, salvar=False)
    indice = Indice.treinar()
    chunks = {c.id: c for c in carregar_chunks()}
    gaz = Gazetteer.a_partir_da_semente()
    return grafo, indice, chunks, gaz


grafo, indice, chunks, gaz = carregar_tudo()
resolvedor = ResolvedorEntidades(gaz)
tem_api = bool(os.getenv("ANTHROPIC_API_KEY"))

# nomes de medicamentos para o seletor (princípios ativos)
MEDS = sorted({d["nome"] for _, d in grafo.nodes(data=True)
               if d.get("tipo") == TipoEntidade.MEDICAMENTO.value})

st.title("⚕️ HYGIA — Verificador de Interações Medicamentosas")
st.caption("Graph RAG com raciocínio multi-hop auditável · protótipo educacional, "
           "não substitui avaliação clínica")

aba1, aba2, aba3 = st.tabs(["🔍 Verificador", "🕸️ Explorar o grafo", "📊 Benchmark"])


# --------------------------------------------------------------------------
# Aba 1 — Verificador
# --------------------------------------------------------------------------

with aba1:
    col_in, col_ex = st.columns([3, 2])
    with col_in:
        selecionados = st.multiselect(
            "Medicamentos em uso (princípio ativo)",
            options=MEDS,
            default=["varfarina", "ibuprofeno"],
            help="Você também pode digitar nomes comerciais no campo abaixo.",
        )
    with col_ex:
        texto_livre = st.text_input(
            "…ou digite livremente (aceita marcas)",
            placeholder="ex.: Marevan, Advil, Zocor",
        )

    meds_final = list(dict.fromkeys(selecionados))
    if texto_livre.strip():
        for _, nome in resolvedor.resolver(texto_livre):
            if nome not in meds_final:
                meds_final.append(nome)

    if st.button("Verificar interações", type="primary", use_container_width=True):
        if len(meds_final) < 2:
            st.warning("Informe pelo menos dois medicamentos.")
        else:
            st.markdown(f"**Analisando:** {', '.join(meds_final)}")
            consulta = "Há interações entre " + " e ".join(meds_final) + "?"
            entidades = [(TipoEntidade.MEDICAMENTO, m) for m in meds_final]

            g_ret = GraphRAGRetriever(indice, grafo)
            res = g_ret.recuperar(consulta, entidades)

            achados = res.achados_grafo
            if not achados:
                st.success("✅ Nenhuma interação registrada na base entre os itens informados. "
                           "(Ausência de registro não é garantia de segurança — a base é curada e limitada.)")
            else:
                pior = max(achados, key=lambda a: a.severidade.peso)
                cor = COR_SEV[pior.severidade.value]
                st.markdown(
                    f"<div style='padding:14px;border-radius:10px;background:{cor}22;"
                    f"border-left:6px solid {cor}'>"
                    f"<h3 style='margin:0;color:{cor}'>{ICONE_SEV[pior.severidade.value]} "
                    f"Gravidade máxima: {pior.severidade.value.upper()}</h3>"
                    f"<p style='margin:4px 0 0'>{len(achados)} interação(ões) identificada(s).</p></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("### Interações encontradas")
                for a in achados:
                    cor = COR_SEV[a.severidade.value]
                    via = {
                        "direta": "documentada diretamente na bula",
                        "mecanismo": f"inferida pelo mecanismo enzimático — regra {a.regra}",
                        "classe": "inferida pela classe do segundo fármaco",
                        "classe_classe": "inferida pela interação entre as classes",
                    }.get(a.tipo, a.tipo)
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            st.markdown(
                                f"<h2 style='margin:0'>{ICONE_SEV[a.severidade.value]}</h2>"
                                f"<span style='color:{cor};font-weight:700'>"
                                f"{a.severidade.value.upper()}</span>",
                                unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"**{a.a} + {a.b}** — _{via}_")
                            caminho = "  →  ".join(
                                f"`{n.split('::')[-1]}`" for n in a.caminho)
                            st.markdown(f"🧭 **Caminho de raciocínio:** {caminho}")
                            st.markdown(f"**Efeito:** {a.efeito}")
                            with st.expander("Mecanismo e conduta"):
                                st.markdown(f"**Mecanismo:** {a.mecanismo}")
                                st.markdown(f"**Conduta:** {a.manejo}")
                                if a.chunks:
                                    st.caption("Evidência (trechos): " + ", ".join(a.chunks))

            # --- resposta redigida + comparação com baseline ---
            st.divider()
            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### 🟢 Resposta — Graph RAG")
                if tem_api:
                    with st.spinner("Redigindo com Claude…"):
                        r = gerar_resposta_llm(consulta, res, chunks)
                    st.info(r.texto)
                else:
                    r = gerar_resposta_offline(consulta, res, chunks)
                    st.info(r.texto)
                    st.caption("Modo offline determinístico. Defina ANTHROPIC_API_KEY "
                               "para respostas redigidas por Claude.")
            with colB:
                st.markdown("#### ⚪ Baseline — Plain RAG (sem grafo)")
                h_ret = HibridoRetriever(indice)
                res_h = h_ret.recuperar(consulta, entidades)
                rh = gerar_resposta_offline(consulta, res_h, chunks)
                st.warning(rh.texto if rh.texto.strip() else
                           "O baseline não reúne evidência suficiente: nenhum trecho "
                           "recuperado cita os dois fármacos simultaneamente.")
                st.caption("Trechos recuperados: " + ", ".join(res_h.chunks[:5]))


# --------------------------------------------------------------------------
# Aba 2 — Explorar o grafo
# --------------------------------------------------------------------------

with aba2:
    st.markdown("Explore a vizinhança de interações de um medicamento no grafo de conhecimento.")
    foco = st.selectbox("Medicamento", MEDS, index=MEDS.index("varfarina") if "varfarina" in MEDS else 0)
    hops = st.slider("Profundidade (saltos)", 1, 3, 2)

    from pyvis.network import Network
    import networkx as nx

    from hygia.graph import vizinhanca
    from hygia.schema import OrigemAresta

    semente = no_id(TipoEntidade.MEDICAMENTO, foco)
    nos = vizinhanca(grafo, [semente], hops=hops)
    sub = grafo.subgraph(nos)

    cor_tipo = {
        "Medicamento": "#2a78d6", "Classe": "#4a3aa7", "Enzima": "#eda100",
        "Condicao": "#9aa0a6", "Efeito": "#c58af9", "Marca": "#80868b",
    }
    net = Network(height="560px", width="100%", bgcolor="#ffffff", font_color="#202124",
                  directed=True, notebook=False)
    net.barnes_hut(gravity=-8000, spring_length=120)
    for n, d in sub.nodes(data=True):
        tipo = d.get("tipo", "?")
        if tipo in ("Chunk",):
            continue
        net.add_node(n, label=d.get("nome", n), color=cor_tipo.get(tipo, "#888"),
                     size=26 if n == semente else (18 if tipo == "Enzima" else 12),
                     title=f"{tipo}: {d.get('nome')}")
    ids = {n["id"] for n in net.nodes}
    for u, v, d in sub.edges(data=True):
        if u in ids and v in ids and d["relacao"] != "MENCIONA":
            inferida = d.get("origem") == OrigemAresta.INFERIDA.value
            net.add_edge(u, v, label=d["relacao"].lower().replace("_", " "),
                         color="#1e8e3e" if inferida else "#cfcfcf",
                         dashes=inferida, width=2 if inferida else 1)
    net.set_options('{"physics": {"stabilization": {"iterations": 150}}}')
    html = net.generate_html()
    st.components.v1.html(html, height=580, scrolling=False)
    st.caption("Arestas verdes tracejadas = interações inferidas por mecanismo enzimático "
               "(não constam de nenhum documento isolado).")


# --------------------------------------------------------------------------
# Aba 3 — Benchmark
# --------------------------------------------------------------------------

with aba3:
    import json

    res_path = ROOT / "resultados" / "avaliacao_recuperacao.json"
    est = estatisticas(grafo)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nós no grafo", est["nos"])
    c2.metric("Arestas", est["arestas"])
    c3.metric("Interações inferidas", est["arestas_inferidas"])
    c4.metric("Documentos no corpus", len({c.doc_id for c in chunks.values()}))

    if res_path.exists():
        dados = json.loads(res_path.read_text(encoding="utf-8"))
        rec = dados["resultados"]["por_recuperador"]
        import pandas as pd

        st.markdown("### Cobertura de fatos por categoria de raciocínio")
        tipos = ["fato_direto", "mecanismo", "ponte_classe", "ponte_classe2"]
        linhas = []
        for r in ["vetorial", "bm25", "hibrido", "graphrag"]:
            linha = {"recuperador": r}
            for t in tipos:
                linha[t] = rec[r]["por_tipo"].get(t, {}).get("cobertura", 0.0)
            linhas.append(linha)
        df = pd.DataFrame(linhas).set_index("recuperador")
        st.dataframe(df.style.format("{:.2f}").background_gradient(cmap="Greens", vmin=0, vmax=1),
                     use_container_width=True)

        st.markdown("### Significância estatística (Graph RAG vs. melhor baseline)")
        sig = dados["resultados"]["significancia"]
        for m, d in sig.items():
            st.write(f"**{m}**: Graph RAG {d['graphrag']:.2f} vs "
                     f"{d['melhor_baseline']} {d['baseline_media']:.2f} · "
                     f"Δ = {d['delta']:+.2f} · p = {d['p_valor']:.4f}"
                     + (" ✅ significativo" if d['significativo_5pct'] else ""))

        fig_path = ROOT / "docs" / "img" / "cobertura_por_categoria.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Rode `make pipeline` para gerar os resultados do benchmark.")
