"""Conjunto de avaliação (gold set) gerado a partir da semente.

Cada pergunta carrega:
  - texto            : a consulta em linguagem natural
  - entidades        : nós âncora (resolvidos) — o que o resolvedor deveria achar
  - chunks_relevantes: gabarito de recuperação (ids de chunk que sustentam a resposta)
  - tipo             : categoria do salto de raciocínio exigido
  - severidade       : rótulo esperado (quando aplicável)

As categorias por dificuldade de raciocínio:
  fato_direto     — resposta está num único chunk (bula do próprio fármaco)
  ponte_classe    — exige compor bula (cita classe) + monografia (membro da classe)
  ponte_classe2   — exige duas pontes de classe (classe↔classe)
  mecanismo       — exige cruzar duas bulas via enzima (aresta inferida)

As três últimas são o "multi-hop A→B→C" onde o RAG por similaridade quebra:
NENHUM chunk contém os dois fármacos da pergunta simultaneamente.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import GOLD_JSON, INTERACOES_JSON, MEDICAMENTOS_JSON, garantir_diretorios
from .corpus import _slug, carregar_chunks
from .schema import TipoEntidade


def _mapa_fato_chunk() -> dict[str, str]:
    """fato (ex.: 'dci:dci-001') -> id do chunk que o contém."""
    mapa: dict[str, str] = {}
    for c in carregar_chunks():
        for f in c.fatos:
            mapa[f] = c.id
    return mapa


@dataclass
class Pergunta:
    id: str
    texto: str
    tipo: str
    entidades: list[list[str]]        # [[tipo, nome], ...]
    chunks_relevantes: list[str]
    severidade: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ids() -> dict[str, str]:
    meds = json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))["medicamentos"]
    return {m["principio_ativo"]: m["id"] for m in meds}


def gerar_gold(salvar: bool = True) -> list[Pergunta]:
    meds = json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))["medicamentos"]
    inter = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))
    id_por_pa = {m["principio_ativo"]: m["id"] for m in meds}
    classe_por_pa = {m["principio_ativo"]: m["classe"] for m in meds}
    membros_por_classe: dict[str, list[str]] = {}
    for m in meds:
        membros_por_classe.setdefault(m["classe"], []).append(m["principio_ativo"])
    fato2chunk = _mapa_fato_chunk()

    perguntas: list[Pergunta] = []
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        perguntas.append(Pergunta(id=f"q{n:03d}", **kw))

    # ---- fato_direto: contraindicações e reações (single-hop) ----
    diretas = [
        ("varfarina", "contraindicacoes", "Posso prescrever varfarina para uma gestante?"),
        ("sinvastatina", "reacoes", "Quais as reações adversas mais comuns da sinvastatina?"),
        ("metformina", "contraindicacoes", "Em que situações a metformina é contraindicada?"),
        ("propranolol", "contraindicacoes", "O propranolol pode ser usado em paciente asmático?"),
        ("digoxina", "reacoes", "Quais são os efeitos adversos comuns da digoxina?"),
        ("fluoxetina", "contraindicacoes", "A fluoxetina tem alguma contraindicação absoluta?"),
    ]
    for pa, secao, texto in diretas:
        cid = f"{id_por_pa[pa]}#{secao}"
        add(texto=texto, tipo="fato_direto",
            entidades=[[TipoEntidade.MEDICAMENTO.value, pa]],
            chunks_relevantes=[cid],
            meta={"secao": secao})

    # ---- mecanismo: pares que só se conectam via enzima (aresta inferida) ----
    # cada bula de metabolismo cita a enzima, nunca o outro fármaco
    pares_mecanismo = [
        ("claritromicina", "sinvastatina", "CYP3A4", "contraindicado"),
        ("cetoconazol", "sildenafila", "CYP3A4", "contraindicado"),
        ("omeprazol", "clopidogrel", "CYP2C19", "contraindicado"),
        ("fluoxetina", "tramadol", "CYP2D6", "contraindicado"),
        ("ciprofloxacino", "teofilina", "CYP1A2", "contraindicado"),
        ("carbamazepina", "sinvastatina", "CYP3A4", "grave"),
        ("fluconazol", "varfarina", "CYP2C9", "contraindicado"),
        ("alopurinol", "azatioprina", "XO", "contraindicado"),
    ]
    for a, b, enz, sev in pares_mecanismo:
        # gabarito: as duas seções de metabolismo (uma de cada bula) + monografia da enzima
        chunks = [
            f"{id_por_pa[a]}#metabolismo",
            f"{id_por_pa[b]}#metabolismo",
        ]
        add(texto=f"Há risco em associar {a} e {b}?",
            tipo="mecanismo",
            entidades=[[TipoEntidade.MEDICAMENTO.value, a], [TipoEntidade.MEDICAMENTO.value, b]],
            chunks_relevantes=chunks,
            severidade=sev,
            meta={"enzima": enz})

    # ---- ponte_classe: a interação está na bula de um lado (cita CLASSE),
    #      o outro fármaco é membro dessa classe (monografia) ----
    for i in inter["interacoes_farmaco_classe"]:
        if i["documentado_em"].startswith("classe:"):
            continue
        pa = i["documentado_em"]
        classe_alvo = i["classe"]
        membros = [m for m in membros_por_classe.get(classe_alvo, []) if m != pa]
        if not membros:
            continue
        outro = sorted(membros)[0]
        # gabarito multi-hop: o chunk de interação (cita a CLASSE, não 'outro')
        # + a monografia da classe (lista 'outro' como membro).
        chunks = [
            fato2chunk[f"dci:{i['id']}"],
            f"classe-{_slug(classe_alvo)}#membros",
        ]
        add(texto=f"Existe interação entre {pa} e {outro}?",
            tipo="ponte_classe",
            entidades=[[TipoEntidade.MEDICAMENTO.value, pa], [TipoEntidade.MEDICAMENTO.value, outro]],
            chunks_relevantes=chunks,
            severidade=i["severidade"],
            meta={"classe": classe_alvo, "ddi_id": i["id"]})

    # ---- ponte_classe2: interação classe↔classe, perguntada por dois fármacos ----
    for i in inter["interacoes_classe_classe"][:8]:
        ca, cb = i["classe_a"], i["classe_b"]
        ma = sorted(membros_por_classe.get(ca, []))
        mb = sorted(membros_por_classe.get(cb, []))
        if not ma or not mb:
            continue
        fa, fb = ma[0], mb[0]
        # gabarito: o chunk da interação classe↔classe + as duas monografias
        # de composição (cada uma lista um dos fármacos como membro).
        chunks = [
            fato2chunk[f"cci:{i['id']}"],
            f"classe-{_slug(ca)}#membros",
            f"classe-{_slug(cb)}#membros",
        ]
        add(texto=f"É seguro combinar {fa} com {fb}?",
            tipo="ponte_classe2",
            entidades=[[TipoEntidade.MEDICAMENTO.value, fa], [TipoEntidade.MEDICAMENTO.value, fb]],
            chunks_relevantes=chunks,
            severidade=i["severidade"],
            meta={"classe_a": ca, "classe_b": cb, "cci_id": i["id"]})

    if salvar:
        garantir_diretorios()
        GOLD_JSON.write_text(
            json.dumps([p.to_dict() for p in perguntas], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return perguntas


def carregar_gold() -> list[Pergunta]:
    return [Pergunta(**d) for d in json.loads(GOLD_JSON.read_text(encoding="utf-8"))]


if __name__ == "__main__":
    ps = gerar_gold()
    por_tipo: dict[str, int] = {}
    for p in ps:
        por_tipo[p.tipo] = por_tipo.get(p.tipo, 0) + 1
    print(f"gold set: {len(ps)} perguntas")
    for t, c in sorted(por_tipo.items()):
        print(f"  {t:<16} {c}")
