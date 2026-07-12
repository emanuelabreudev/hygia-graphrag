"""PASSO 0 — Construção do corpus documental (bulas e monografias).

Princípio de projeto (leia antes de criticar o benchmark):

    O corpus é COMPLETO. Todo fato necessário para responder qualquer
    pergunta do conjunto de avaliação está escrito, em português, em
    algum chunk. Os baselines recebem exatamente o mesmo corpus.

O que torna o problema difícil não é informação faltante, e sim
informação *dispersa*: a interação entre varfarina e ibuprofeno não
aparece em documento nenhum. O que existe é (a) a bula da varfarina
dizendo que ela interage com anti-inflamatórios não esteroidais e (b) a
monografia da classe dizendo que o ibuprofeno é um anti-inflamatório não
esteroidal. Compor (a) + (b) é trabalho de grafo, não de similaridade de
cosseno.

O mesmo vale, com um salto a mais, para as interações mecanísticas: a
bula da claritromicina diz que ela inibe a CYP3A4; a da sinvastatina diz
que ela é metabolizada pela CYP3A4. Nenhuma das duas cita a outra.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import CHUNKS_JSON, INTERACOES_JSON, MEDICAMENTOS_JSON, garantir_diretorios


@dataclass
class Chunk:
    id: str
    doc_id: str
    titulo: str
    secao: str
    texto: str
    # rótulos de proveniência — usados na avaliação de extração, NUNCA na recuperação
    fatos: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t


def _listar(itens: list[str]) -> str:
    """Junta itens em português: 'a, b e c'."""
    itens = [i for i in itens if i]
    if not itens:
        return ""
    if len(itens) == 1:
        return itens[0]
    return ", ".join(itens[:-1]) + " e " + itens[-1]


def carregar_sementes() -> tuple[dict, dict]:
    meds = json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))
    inter = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))
    return meds, inter


# --------------------------------------------------------------------------
# Geração de bulas
# --------------------------------------------------------------------------

_PAPEL_TEXTO = {
    "inibidor": (
        "É um inibidor de intensidade {intensidade} da {enzima} ({enzima_nome}). "
        "Ao reduzir a atividade dessa enzima, pode elevar a concentração plasmática "
        "de fármacos que dependem dela para sua eliminação."
    ),
    "indutor": (
        "É um indutor de intensidade {intensidade} da {enzima} ({enzima_nome}). "
        "Ao acelerar a atividade dessa enzima, pode reduzir a concentração plasmática "
        "e a eficácia de fármacos que dependem dela para sua eliminação."
    ),
    "substrato": (
        "É metabolizado pela {enzima} ({enzima_nome}), da qual é substrato de "
        "sensibilidade {intensidade}. Fármacos que inibam ou induzam essa enzima podem "
        "alterar de forma clinicamente relevante sua concentração plasmática."
    ),
    "pro_farmaco": (
        "É um pró-fármaco: precisa ser bioativado pela {enzima} ({enzima_nome}) para "
        "gerar seu metabólito ativo. A inibição dessa enzima bloqueia a bioativação e "
        "leva à perda de efeito terapêutico, ainda que o paciente esteja tomando o "
        "comprimido corretamente."
    ),
}


def _chunk_identificacao(med: dict) -> Chunk:
    pa = med["principio_ativo"]
    marcas = _listar(med["nomes_comerciais"])
    sin = _listar(med.get("sinonimos", []))
    texto = (
        f"Bula de {pa}. Princípio ativo: {pa}. "
        f"Classe farmacológica: {med['classe']}. "
        f"Código ATC: {med['atc']}. "
        f"Comercializado no Brasil sob os nomes {marcas}."
    )
    if sin:
        texto += f" Também referido como {sin}."
    return Chunk(
        id=f"{med['id']}#identificacao",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Identificação",
        texto=texto,
        fatos=[f"classe:{med['id']}"] + [f"marca:{med['id']}:{m}" for m in med["nomes_comerciais"]],
    )


def _chunk_indicacoes(med: dict) -> Chunk:
    pa = med["principio_ativo"]
    texto = (
        f"Indicações de {pa}. Este medicamento é indicado para {_listar(med['indicacoes'])}. "
        f"A prescrição deve considerar o quadro clínico completo do paciente."
    )
    return Chunk(
        id=f"{med['id']}#indicacoes",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Indicações",
        texto=texto,
        fatos=[f"indicacao:{med['id']}"],
    )


def _chunk_contraindicacoes(med: dict) -> Chunk:
    pa = med["principio_ativo"]
    texto = (
        f"Contraindicações de {pa}. O uso é contraindicado nas seguintes situações: "
        f"{_listar(med['contraindicacoes'])}."
    )
    return Chunk(
        id=f"{med['id']}#contraindicacoes",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Contraindicações",
        texto=texto,
        fatos=[f"contraindicacao:{med['id']}"],
    )


def _chunk_reacoes(med: dict) -> Chunk:
    pa = med["principio_ativo"]
    texto = (
        f"Reações adversas de {pa}. As reações mais frequentemente relatadas são "
        f"{_listar(med['reacoes_comuns'])}. Reações raras e graves devem ser notificadas "
        f"ao sistema de farmacovigilância."
    )
    return Chunk(
        id=f"{med['id']}#reacoes",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Reações adversas",
        texto=texto,
        fatos=[f"reacao:{med['id']}"],
    )


def _chunk_conservacao(med: dict) -> Chunk:
    """Seção deliberadamente irrelevante — distrator de recuperação.

    Bulas reais são cheias de texto assim. Se o recuperador traz este chunk
    no top-k, ele está gastando contexto com ruído.
    """
    pa = med["principio_ativo"]
    texto = (
        f"Conservação e armazenamento de {pa}. Conservar em temperatura ambiente, entre "
        f"15 °C e 30 °C, protegido da luz e da umidade. Manter na embalagem original e fora "
        f"do alcance de crianças. Não utilizar o medicamento após a data de validade impressa "
        f"na embalagem. Descarte o medicamento vencido em ponto de coleta apropriado."
    )
    return Chunk(
        id=f"{med['id']}#conservacao",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Conservação",
        texto=texto,
        fatos=[],
    )


def _chunk_metabolismo(med: dict, metab: list[dict], enzimas: dict[str, dict]) -> Chunk | None:
    """Seção de metabolismo — a fonte dos saltos mecanísticos.

    Note o que este chunk NÃO diz: ele não nomeia nenhum outro fármaco.
    Só declara o papel deste medicamento sobre uma enzima. A interação
    mecanística nasce do cruzamento de dois chunks como este, feito pelo
    grafo — nunca por similaridade textual.
    """
    pa = med["principio_ativo"]
    linhas: list[str] = []
    fatos: list[str] = []
    for m in metab:
        if m["medicamento"] != pa:
            continue
        enz = enzimas[m["enzima"]]
        linhas.append(
            _PAPEL_TEXTO[m["papel"]].format(
                intensidade=m["intensidade"].replace("sensivel", "sensível"),
                enzima=m["enzima"],
                enzima_nome=enz["nome"],
            )
        )
        fatos.append(f"metabolismo:{pa}:{m['papel']}:{m['enzima']}")
    if not linhas:
        return None
    texto = f"Metabolismo e vias de eliminação de {pa}. " + " ".join(linhas)
    return Chunk(
        id=f"{med['id']}#metabolismo",
        doc_id=f"bula:{med['id']}",
        titulo=f"Bula — {pa}",
        secao="Metabolismo",
        texto=texto,
        fatos=fatos,
    )


def _texto_interacao(alvo: str, i: dict, tipo_alvo: str = "medicamento") -> str:
    prefixo = {
        "medicamento": f"o uso concomitante com {alvo}",
        "classe": f"o uso concomitante com medicamentos da classe dos {alvo}",
    }[tipo_alvo]
    sev = {
        "contraindicado": "Esta associação é CONTRAINDICADA.",
        "grave": "Esta é uma interação de gravidade ALTA.",
        "moderada": "Esta é uma interação de gravidade MODERADA.",
        "leve": "Esta é uma interação de gravidade LEVE.",
    }[i["severidade"]]
    return (
        f"Interação: {prefixo}. {sev} "
        f"Mecanismo: {i['mecanismo']} "
        f"Consequência clínica: {i['efeito']} "
        f"Conduta: {i['manejo']} "
        f"Nível de evidência: {i['evidencia']}."
    )


def _chunks_interacoes_bula(
    med: dict, ddi: list[dict], dci: list[dict], id_por_pa: dict[str, str]
) -> list[Chunk]:
    """Interações que a bula DESTE medicamento documenta.

    Assimetria intencional e realista: cada interação fármaco-fármaco é
    descrita na bula de apenas um dos dois (campo `documentado_em`).
    Perguntar pelo outro lado exige percorrer a aresta no grafo.
    """
    pa = med["principio_ativo"]
    chunks: list[Chunk] = []
    n = 0

    for i in ddi:
        if i["documentado_em"] != pa:
            continue
        outro = i["b"] if i["a"] == pa else i["a"]
        n += 1
        chunks.append(
            Chunk(
                id=f"{med['id']}#interacao-{n}",
                doc_id=f"bula:{med['id']}",
                titulo=f"Bula — {pa}",
                secao="Interações medicamentosas",
                texto=f"Interações medicamentosas de {pa}. " + _texto_interacao(outro, i, "medicamento"),
                fatos=[f"ddi:{i['id']}"],
            )
        )

    for i in dci:
        if i["documentado_em"] != pa:
            continue
        n += 1
        chunks.append(
            Chunk(
                id=f"{med['id']}#interacao-{n}",
                doc_id=f"bula:{med['id']}",
                titulo=f"Bula — {pa}",
                secao="Interações medicamentosas",
                texto=f"Interações medicamentosas de {pa}. "
                + _texto_interacao(i["classe"], i, "classe"),
                fatos=[f"dci:{i['id']}"],
            )
        )
    return chunks


# --------------------------------------------------------------------------
# Monografias de classe
# --------------------------------------------------------------------------


def _chunks_monografia_classe(
    classe: str, membros: list[dict], cci: list[dict]
) -> list[Chunk]:
    """Monografia de classe.

    Contém a lista de membros — de propósito. É esta lista que dá ao
    baseline a chance justa de resolver as perguntas de classe: a
    informação existe no corpus. O que o baseline não tem é o mecanismo
    para COMPOR essa lista com a bula que cita a classe.
    """
    slug = _slug(classe)
    doc_id = f"monografia:classe:{slug}"
    nomes = _listar(sorted(m["principio_ativo"] for m in membros))
    chunks = [
        Chunk(
            id=f"classe-{slug}#membros",
            doc_id=doc_id,
            titulo=f"Monografia de classe — {classe}",
            secao="Composição da classe",
            texto=(
                f"Monografia da classe {classe}. Pertencem a esta classe farmacológica os "
                f"seguintes princípios ativos disponíveis no Brasil: {nomes}. "
                f"Fármacos de uma mesma classe compartilham mecanismo de ação e, em regra, "
                f"o mesmo perfil de interações medicamentosas."
            ),
            fatos=[f"membros_classe:{classe}"],
        )
    ]
    n = 0
    for i in cci:
        if i["documentado_em"] != f"classe:{classe}":
            continue
        outra = i["classe_b"] if i["classe_a"] == classe else i["classe_a"]
        n += 1
        chunks.append(
            Chunk(
                id=f"classe-{slug}#interacao-{n}",
                doc_id=doc_id,
                titulo=f"Monografia de classe — {classe}",
                secao="Interações de classe",
                texto=(
                    f"Interações da classe {classe}. "
                    + _texto_interacao(outra, i, "classe")
                ),
                fatos=[f"cci:{i['id']}"],
            )
        )
    return chunks


def _chunk_enzima(enz: dict) -> Chunk:
    return Chunk(
        id=f"enzima-{_slug(enz['id'])}#descricao",
        doc_id=f"monografia:enzima:{_slug(enz['id'])}",
        titulo=f"Monografia de via metabólica — {enz['id']}",
        secao="Descrição",
        texto=(
            f"Via metabólica {enz['id']} — {enz['nome']}. Trata-se de uma enzima {enz['tipo']}. "
            f"{enz['nota']} A inibição desta via reduz a depuração de seus substratos e eleva "
            f"a concentração plasmática deles; a indução produz o efeito oposto. Quando o "
            f"substrato é um pró-fármaco, a inibição da via impede sua ativação e causa perda "
            f"de eficácia."
        ),
        fatos=[f"enzima:{enz['id']}"],
    )


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------


def construir_corpus(salvar: bool = True) -> list[Chunk]:
    meds_raw, inter = carregar_sementes()
    meds = meds_raw["medicamentos"]
    enzimas = {e["id"]: e for e in inter["enzimas"]}
    metab = inter["metabolismo"]
    ddi = inter["interacoes_farmaco_farmaco"]
    dci = inter["interacoes_farmaco_classe"]
    cci = inter["interacoes_classe_classe"]

    id_por_pa = {m["principio_ativo"]: m["id"] for m in meds}

    chunks: list[Chunk] = []
    for med in meds:
        chunks.append(_chunk_identificacao(med))
        chunks.append(_chunk_indicacoes(med))
        chunks.append(_chunk_contraindicacoes(med))
        chunks.append(_chunk_reacoes(med))
        cm = _chunk_metabolismo(med, metab, enzimas)
        if cm:
            chunks.append(cm)
        chunks.extend(_chunks_interacoes_bula(med, ddi, dci, id_por_pa))
        chunks.append(_chunk_conservacao(med))

    classes: dict[str, list[dict]] = {}
    for med in meds:
        classes.setdefault(med["classe"], []).append(med)
    for classe, membros in sorted(classes.items()):
        chunks.extend(_chunks_monografia_classe(classe, membros, cci))

    for enz in inter["enzimas"]:
        chunks.append(_chunk_enzima(enz))

    _checar_integridade(chunks, ddi, dci, cci)

    if salvar:
        garantir_diretorios()
        CHUNKS_JSON.write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return chunks


def _checar_integridade(
    chunks: list[Chunk], ddi: list[dict], dci: list[dict], cci: list[dict]
) -> None:
    """Garante que toda interação declarada virou exatamente um chunk.

    Um `documentado_em` com nome errado silenciosamente sumiria do corpus
    e inflaria artificialmente a diferença entre grafo e baseline. Isso é
    exatamente o tipo de bug que invalidaria o benchmark, então falha alto.
    """
    fatos = {f for c in chunks for f in c.fatos}
    faltando = []
    for i in ddi:
        if f"ddi:{i['id']}" not in fatos:
            faltando.append(f"ddi:{i['id']} (documentado_em={i['documentado_em']!r})")
    for i in dci:
        if f"dci:{i['id']}" not in fatos:
            faltando.append(f"dci:{i['id']} (documentado_em={i['documentado_em']!r})")
    for i in cci:
        if f"cci:{i['id']}" not in fatos:
            faltando.append(f"cci:{i['id']} (documentado_em={i['documentado_em']!r})")
    if faltando:
        raise ValueError(
            "Interações declaradas na semente que não geraram chunk — "
            "'documentado_em' provavelmente não casa com nenhum princípio ativo "
            "ou classe existente:\n  " + "\n  ".join(faltando)
        )


def carregar_chunks() -> list[Chunk]:
    dados = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    return [Chunk(**d) for d in dados]


if __name__ == "__main__":
    cs = construir_corpus()
    docs = {c.doc_id for c in cs}
    tokens = sum(len(c.texto.split()) for c in cs)
    print(f"corpus: {len(cs)} chunks | {len(docs)} documentos | ~{tokens} tokens")
