"""Testes de contrato e regressão do pipeline.

Rápidos (segundos) e sem rede — servem de smoke-test no CI. Fixam os
resultados-chave para que uma regressão silenciosa no motor de inferência
ou na recuperação quebre o build, não passe despercebida.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hygia import ablation, corpus, extract, gold
from hygia.eval import avaliar_recuperacao, agregar
from hygia.graph import analisar_par, construir_completo, estatisticas
from hygia.index import Indice
from hygia.resolver import ResolvedorEntidades
from hygia.retriever import GraphRAGRetriever, HibridoRetriever
from hygia.schema import Severidade, TipoEntidade


@pytest.fixture(scope="module")
def artefatos():
    corpus.construir_corpus()
    triplas = extract.executar("gazetteer")
    gold.gerar_gold()
    grafo = construir_completo(triplas, salvar=False)
    indice = Indice.treinar()
    return {"triplas": triplas, "grafo": grafo, "indice": indice}


# --- Corpus ---------------------------------------------------------------


def test_corpus_completude(artefatos):
    """Toda interação da semente vira um chunk (checagem interna já roda)."""
    chunks = corpus.carregar_chunks()
    assert len(chunks) > 300
    assert all(c.texto for c in chunks)


def test_corpus_assimetria():
    """Cada interação fármaco-fármaco é documentada em apenas um dos lados."""
    import json

    from hygia.config import INTERACOES_JSON, MEDICAMENTOS_JSON

    inter = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))
    pas = {m["principio_ativo"] for m in
           json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))["medicamentos"]}
    for i in inter["interacoes_farmaco_farmaco"]:
        assert i["documentado_em"] in (i["a"], i["b"])
        assert i["documentado_em"] in pas


# --- Extração -------------------------------------------------------------


def test_extracao_perfeita(artefatos):
    """A extração por gazetteer deve reproduzir a semente (F1 = 1.0)."""
    m = extract.avaliar_extracao(artefatos["triplas"])
    assert m["precisao"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_extracao_valida_esquema(artefatos):
    assert all(t.valida() for t in artefatos["triplas"])


# --- Grafo e inferência ---------------------------------------------------


def test_grafo_tem_inferencias(artefatos):
    est = estatisticas(artefatos["grafo"])
    assert est["arestas_inferidas"] >= 40


def test_inferencia_mecanismo_pro_farmaco(artefatos):
    """omeprazol inibe CYP2C19; clopidogrel é ativado por ela → contraindicado.

    Nenhum documento cita os dois juntos: é raciocínio de grafo puro.
    """
    achados = analisar_par(artefatos["grafo"], "omeprazol", "clopidogrel")
    mec = [a for a in achados if a.tipo == "mecanismo"]
    assert mec, "interação mecanística omeprazol+clopidogrel não foi inferida"
    assert mec[0].severidade is Severidade.CONTRAINDICADO
    assert "cyp2c19" in " → ".join(mec[0].caminho).lower()


def test_inferencia_estatina_macrolideo(artefatos):
    """claritromicina inibe CYP3A4; sinvastatina é substrato → grave/contraindicado."""
    achados = analisar_par(artefatos["grafo"], "claritromicina", "sinvastatina")
    assert achados
    assert achados[0].severidade.peso >= Severidade.GRAVE.peso


def test_ponte_de_classe(artefatos):
    """varfarina interage com a classe dos AINEs; ibuprofeno é um AINE."""
    achados = analisar_par(artefatos["grafo"], "varfarina", "ibuprofeno")
    assert any(a.tipo == "classe" for a in achados)
    assert achados[0].severidade.peso >= Severidade.GRAVE.peso


def test_par_sem_interacao(artefatos):
    """Par sem qualquer caminho de interação retorna vazio (sem falso positivo)."""
    achados = analisar_par(artefatos["grafo"], "paracetamol", "levotiroxina")
    assert achados == []


# --- Recuperação e avaliação ----------------------------------------------


def test_resolvedor_reconhece_marca():
    r = ResolvedorEntidades()
    ents = r.resolver("posso dar Marevan com Aspirina?")
    nomes = {n for _, n in ents}
    assert "varfarina" in nomes
    assert "ácido acetilsalicílico" in nomes


def test_graphrag_supera_baseline_em_multihop(artefatos):
    """Contrato central: em multi-hop, Graph RAG cobre e o híbrido não."""
    indice, grafo = artefatos["indice"], artefatos["grafo"]
    g = GraphRAGRetriever(indice, grafo)
    h = HibridoRetriever(indice)
    resolvedor = ResolvedorEntidades()
    q = "Há risco em associar claritromicina e sinvastatina?"
    ents = resolvedor.resolver(q)
    rg = g.recuperar(q, ents)
    rh = h.recuperar(q, ents)
    assert rg.achados_grafo, "Graph RAG não produziu achados estruturados"
    assert not rh.achados_grafo, "híbrido não deveria ter achados de grafo"


def test_avaliacao_cobertura_regressao(artefatos):
    """Congela os números-chave da avaliação para pegar regressões."""
    linhas = avaliar_recuperacao()
    agg = agregar(linhas)["por_recuperador"]
    # Graph RAG: cobertura perfeita
    assert agg["graphrag"]["geral"]["cobertura"]["media"] == 1.0
    # baselines colapsam em multi-hop
    for base in ("vetorial", "bm25", "hibrido"):
        assert agg[base]["por_tipo"]["mecanismo"]["cobertura"] == 0.0
        assert agg[base]["por_tipo"]["ponte_classe2"]["cobertura"] == 0.0


def test_significancia(artefatos):
    linhas = avaliar_recuperacao()
    sig = agregar(linhas)["significancia"]
    assert sig["cobertura"]["significativo_5pct"]
    assert sig["cobertura"]["delta"] > 0.5


def test_ablacao_inferencia_importa(artefatos):
    """Remover a inferência deve derrubar a categoria mecanismo."""
    saida = ablation.executar(salvar=False)["ablacao"]
    completo = saida["completo"]["por_tipo"]["mecanismo"]
    sem_inf = saida["sem_inferencia"]["por_tipo"]["mecanismo"]
    assert completo == 1.0
    assert sem_inf < completo


# --- Determinismo ---------------------------------------------------------


def test_determinismo_indice():
    a = Indice.treinar().vetorial.matriz
    b = Indice.treinar().vetorial.matriz
    assert (a == b).all()
