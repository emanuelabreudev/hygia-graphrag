"""PASSO 6 — Geração da resposta a partir do contexto recuperado.

Dois geradores:

  gerar_resposta_llm    — Claude recebe (a) os chunks de texto e (b), no
                          caso do Graph RAG, o subgrafo de interações
                          estruturado. Exige ANTHROPIC_API_KEY.

  gerar_resposta_offline— Extrai a resposta de forma determinística do
                          contexto (achados do grafo + regras textuais).
                          Sem rede: garante que o pipeline e o CI rodem
                          ponta a ponta e que a avaliação de RECUPERAÇÃO
                          seja independente da disponibilidade da API.

A avaliação principal (recall@k, MRR, cobertura de fatos) mede a
RECUPERAÇÃO e não depende do gerador. A qualidade da geração é avaliada
à parte quando há API key (ver eval.avaliar_geracao).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import MODELO_LLM
from .corpus import Chunk
from .graph import Achado
from .retriever import Resultado
from .schema import Severidade


@dataclass
class Resposta:
    texto: str
    severidade_max: Severidade | None
    fontes: list[str]
    modo: str  # 'llm' | 'offline'


_ORDEM_SEV = [Severidade.CONTRAINDICADO, Severidade.GRAVE, Severidade.MODERADA, Severidade.LEVE]


def _sev_max(achados: list[Achado]) -> Severidade | None:
    if not achados:
        return None
    return max((a.severidade for a in achados), key=lambda s: s.peso)


def _texto_dos_chunks(res: Resultado, mapa: dict[str, Chunk], limite: int) -> str:
    linhas = []
    for cid in res.chunks[:limite]:
        c = mapa.get(cid)
        if c:
            linhas.append(f"[{cid}] ({c.secao} — {c.titulo})\n{c.texto}")
    return "\n\n".join(linhas)


def _texto_do_subgrafo(achados: list[Achado], limite: int) -> str:
    if not achados:
        return "(nenhuma interação estruturada encontrada entre as entidades)"
    linhas = []
    for a in achados[:limite]:
        caminho = " → ".join(n.split("::")[-1] for n in a.caminho)
        linhas.append(
            f"- [{a.severidade.value.upper()} | via {a.tipo} | {caminho}] "
            f"{a.a} + {a.b}: {a.efeito} Conduta: {a.manejo}"
            + (f" (regra {a.regra})" if a.regra else "")
        )
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# Gerador offline (determinístico)
# --------------------------------------------------------------------------


def gerar_resposta_offline(consulta: str, res: Resultado, mapa: dict[str, Chunk]) -> Resposta:
    sev = _sev_max(res.achados_grafo)
    partes: list[str] = []

    if res.achados_grafo:
        cab = {
            Severidade.CONTRAINDICADO: "ASSOCIAÇÃO CONTRAINDICADA.",
            Severidade.GRAVE: "Interação de gravidade ALTA identificada.",
            Severidade.MODERADA: "Interação de gravidade MODERADA identificada.",
            Severidade.LEVE: "Interação de gravidade LEVE identificada.",
        }[sev]
        partes.append(cab)
        for a in res.achados_grafo[:4]:
            via = {
                "direta": "documentada diretamente",
                "mecanismo": f"inferida pelo mecanismo enzimático (regra {a.regra})",
                "classe": "por pertencer o segundo fármaco à classe citada",
                "classe_classe": "por interação entre as classes dos dois fármacos",
            }.get(a.tipo, a.tipo)
            partes.append(f"• {a.a} + {a.b} — {via}. {a.efeito} {a.mecanismo} Conduta: {a.manejo}")
    else:
        # sem par: responde com os melhores chunks (mesmo caminho dos baselines)
        for cid in res.chunks[:3]:
            c = mapa.get(cid)
            if c:
                partes.append(c.texto)

    texto = "\n".join(partes) if partes else "Não foram encontradas informações suficientes."
    return Resposta(texto, sev, res.chunks[:6], "offline")


# --------------------------------------------------------------------------
# Gerador com Claude
# --------------------------------------------------------------------------

_SISTEMA = """Você é um assistente de segurança medicamentosa para profissionais de saúde.
Responda SOMENTE com base no contexto fornecido (trechos de bulas e o subgrafo de interações).
Regras:
- Se houver interação, declare a gravidade (contraindicado, grave, moderada ou leve) logo no início.
- Explique o mecanismo e a conduta prática.
- Cite os identificadores de trecho entre colchetes que sustentam cada afirmação.
- Se o contexto não sustentar a resposta, diga isso — não invente.
- Este é um sistema de apoio à decisão; a conduta final é do profissional responsável."""

_TEMPLATE = """CONSULTA: {consulta}

SUBGRAFO DE INTERAÇÕES (relações estruturadas extraídas do grafo de conhecimento):
{subgrafo}

TRECHOS RECUPERADOS:
{chunks}

Responda à consulta."""


def gerar_resposta_llm(
    consulta: str, res: Resultado, mapa: dict[str, Chunk],
    max_chunks: int = 8, max_fatos: int = 12, modelo: str | None = None, usar_grafo: bool = True,
) -> Resposta:
    import anthropic

    cliente = anthropic.Anthropic()
    subgrafo = _texto_do_subgrafo(res.achados_grafo, max_fatos) if usar_grafo else \
        "(recuperador sem grafo)"
    prompt = _TEMPLATE.format(
        consulta=consulta,
        subgrafo=subgrafo,
        chunks=_texto_dos_chunks(res, mapa, max_chunks),
    )
    resp = cliente.messages.create(
        model=modelo or MODELO_LLM,
        max_tokens=1500,
        system=_SISTEMA,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        return Resposta("(resposta recusada pelo modelo)", _sev_max(res.achados_grafo),
                        res.chunks[:max_chunks], "llm")
    texto = "".join(b.text for b in resp.content if b.type == "text")
    return Resposta(texto, _sev_max(res.achados_grafo), res.chunks[:max_chunks], "llm")
