"""PASSO 2 — Extração de entidades e relações (texto → triplas).

A extração roda sobre o TEXTO dos chunks, não sobre o JSON semente. Isso
não é um detalhe estético: é o que permite medir a qualidade da própria
extração (precisão/recall/F1) contra a semente, que funciona como
gabarito. Um pipeline que "extraísse" lendo o JSON estaria se avaliando
contra si mesmo.

Dois extratores intercambiáveis:

  gazetteer  (padrão)  — NER por dicionário + regras sobre padrões
                          linguísticos. Determinístico, sem custo, roda
                          no CI.
  llm                  — Claude faz a extração de triplas. Mais robusto a
                          variação linguística; exige ANTHROPIC_API_KEY.

Ambos produzem `list[Tripla]` e passam pelo mesmo validador de esquema.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from .config import INTERACOES_JSON, MEDICAMENTOS_JSON, TRIPLAS_JSON, garantir_diretorios
from .corpus import Chunk, carregar_chunks
from .schema import (
    ASSINATURAS,
    OrigemAresta,
    Severidade,
    TipoEntidade,
    TipoRelacao,
    Tripla,
)


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Gazetteer (léxico do domínio)
# --------------------------------------------------------------------------


@dataclass
class Gazetteer:
    """Léxico de superfícies → entidade canônica.

    Construído a partir da semente: princípios ativos, sinônimos, nomes
    comerciais, classes e enzimas. É o único ponto em que a semente é
    consultada durante a extração — para *reconhecer* nomes, não para
    inferir relações. As relações saem do texto.
    """

    superficie_para_med: dict[str, str]
    superficie_para_classe: dict[str, str]
    superficie_para_enzima: dict[str, str]
    marcas: dict[str, str]  # marca normalizada → princípio ativo

    @classmethod
    def a_partir_da_semente(cls) -> "Gazetteer":
        meds = json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))["medicamentos"]
        inter = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))

        sm: dict[str, str] = {}
        sc: dict[str, str] = {}
        marcas: dict[str, str] = {}
        for m in meds:
            pa = m["principio_ativo"]
            sm[_norm(pa)] = pa
            for s in m.get("sinonimos", []):
                sm[_norm(s)] = pa
            for b in m["nomes_comerciais"]:
                sm[_norm(b)] = pa
                marcas[_norm(b)] = pa
            sc[_norm(m["classe"])] = m["classe"]

        for i in inter["interacoes_farmaco_classe"]:
            sc[_norm(i["classe"])] = i["classe"]
        for i in inter["interacoes_classe_classe"]:
            sc[_norm(i["classe_a"])] = i["classe_a"]
            sc[_norm(i["classe_b"])] = i["classe_b"]

        se = {}
        for e in inter["enzimas"]:
            se[_norm(e["id"])] = e["id"]
            se[_norm(e["nome"])] = e["id"]
        return cls(sm, sc, se, marcas)

    def _achar(self, texto_norm: str, mapa: dict[str, str]) -> list[str]:
        """Casamento por superfície com fronteira de palavra, mais longo primeiro."""
        achados: list[str] = []
        for sup in sorted(mapa, key=len, reverse=True):
            if re.search(rf"(?<![\w-]){re.escape(sup)}(?![\w-])", texto_norm):
                canon = mapa[sup]
                if canon not in achados:
                    achados.append(canon)
        return achados

    def medicamentos(self, texto: str) -> list[str]:
        return self._achar(_norm(texto), self.superficie_para_med)

    def classes(self, texto: str) -> list[str]:
        return self._achar(_norm(texto), self.superficie_para_classe)

    def enzimas(self, texto: str) -> list[str]:
        return self._achar(_norm(texto), self.superficie_para_enzima)

    def resolver(self, termo: str) -> tuple[TipoEntidade, str] | None:
        """Resolve um termo livre (marca, sinônimo, classe, enzima) ao canônico.

        É o que permite ao usuário digitar 'Marevan' e o sistema entender
        'varfarina'.
        """
        n = _norm(termo)
        if n in self.superficie_para_med:
            return (TipoEntidade.MEDICAMENTO, self.superficie_para_med[n])
        if n in self.superficie_para_classe:
            return (TipoEntidade.CLASSE, self.superficie_para_classe[n])
        if n in self.superficie_para_enzima:
            return (TipoEntidade.ENZIMA, self.superficie_para_enzima[n])
        return None


# --------------------------------------------------------------------------
# Extrator por regras
# --------------------------------------------------------------------------

_RE_SEVERIDADE = {
    Severidade.CONTRAINDICADO: re.compile(r"associacao e contraindicada"),
    Severidade.GRAVE: re.compile(r"gravidade alta"),
    Severidade.MODERADA: re.compile(r"gravidade moderada"),
    Severidade.LEVE: re.compile(r"gravidade leve"),
}
_RE_EVIDENCIA = re.compile(r"nivel de evidencia:\s*(alta|moderada|baixa)")
_RE_MECANISMO = re.compile(r"Mecanismo:\s*(.+?)\s*Consequência clínica:", re.S)
_RE_EFEITO = re.compile(r"Consequência clínica:\s*(.+?)\s*Conduta:", re.S)
_RE_MANEJO = re.compile(r"Conduta:\s*(.+?)\s*Nível de evidência:", re.S)
_RE_CLASSE_ALVO = re.compile(r"uso concomitante com medicamentos da classe dos (.+?)\.")
_RE_MED_ALVO = re.compile(r"uso concomitante com (?!medicamentos da classe)(.+?)\.")


def _severidade(texto: str) -> Severidade:
    n = _norm(texto)
    for sev, rx in _RE_SEVERIDADE.items():
        if rx.search(n):
            return sev
    return Severidade.MODERADA


def _atributos_interacao(texto: str) -> dict:
    ev = _RE_EVIDENCIA.search(_norm(texto))
    return {
        "severidade": _severidade(texto).value,
        "mecanismo": (m.group(1).strip() if (m := _RE_MECANISMO.search(texto)) else ""),
        "efeito": (m.group(1).strip() if (m := _RE_EFEITO.search(texto)) else ""),
        "manejo": (m.group(1).strip() if (m := _RE_MANEJO.search(texto)) else ""),
        "evidencia": ev.group(1) if ev else "moderada",
    }


def _sujeito_do_chunk(chunk: Chunk, gaz: Gazetteer) -> tuple[TipoEntidade, str] | None:
    """Identifica de quem é o documento a partir do doc_id."""
    if chunk.doc_id.startswith("bula:"):
        # o título carrega o princípio ativo canônico
        pa = chunk.titulo.split("—", 1)[-1].strip()
        r = gaz.resolver(pa)
        return r
    if chunk.doc_id.startswith("monografia:classe:"):
        cl = chunk.titulo.split("—", 1)[-1].strip()
        return gaz.resolver(cl)
    if chunk.doc_id.startswith("monografia:enzima:"):
        en = chunk.titulo.split("—", 1)[-1].strip()
        return gaz.resolver(en)
    return None


def extrair_gazetteer(chunks: Iterable[Chunk], gaz: Gazetteer) -> list[Tripla]:
    triplas: list[Tripla] = []

    for c in chunks:
        suj = _sujeito_do_chunk(c, gaz)

        # --- MENCIONA: liga chunks às entidades (usado no passo 4) ---
        for m in gaz.medicamentos(c.texto):
            triplas.append(
                Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, m,
                       TipoEntidade.MEDICAMENTO, chunk_id=c.id)
            )
        for cl in gaz.classes(c.texto):
            triplas.append(
                Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, cl,
                       TipoEntidade.CLASSE, chunk_id=c.id)
            )
        for en in gaz.enzimas(c.texto):
            triplas.append(
                Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, en,
                       TipoEntidade.ENZIMA, chunk_id=c.id)
            )

        if suj is None:
            continue
        tipo_suj, nome_suj = suj

        # --- Identificação: classe e marcas ---
        if c.secao == "Identificação" and tipo_suj is TipoEntidade.MEDICAMENTO:
            m = re.search(r"Classe farmacológica:\s*(.+?)\.", c.texto)
            if m and (r := gaz.resolver(m.group(1))):
                triplas.append(
                    Tripla(nome_suj, TipoEntidade.MEDICAMENTO, TipoRelacao.E_DA_CLASSE,
                           r[1], TipoEntidade.CLASSE, chunk_id=c.id)
                )
            m = re.search(r"sob os nomes (.+?)\.", c.texto)
            if m:
                for marca in re.split(r",| e ", m.group(1)):
                    marca = marca.strip()
                    if marca:
                        triplas.append(
                            Tripla(nome_suj, TipoEntidade.MEDICAMENTO, TipoRelacao.TEM_MARCA,
                                   marca, TipoEntidade.MARCA, chunk_id=c.id)
                        )

        # --- Indicações / contraindicações / reações ---
        if c.secao == "Indicações" and tipo_suj is TipoEntidade.MEDICAMENTO:
            m = re.search(r"indicado para (.+?)\.", c.texto)
            if m:
                for cond in re.split(r",| e ", m.group(1)):
                    cond = cond.strip()
                    if cond:
                        triplas.append(
                            Tripla(nome_suj, TipoEntidade.MEDICAMENTO, TipoRelacao.INDICADO_PARA,
                                   cond, TipoEntidade.CONDICAO, chunk_id=c.id)
                        )
        if c.secao == "Contraindicações" and tipo_suj is TipoEntidade.MEDICAMENTO:
            m = re.search(r"situações:\s*(.+?)\.$", c.texto.strip())
            if m:
                for cond in re.split(r",| e (?![^(]*\))", m.group(1)):
                    cond = cond.strip()
                    if cond:
                        triplas.append(
                            Tripla(nome_suj, TipoEntidade.MEDICAMENTO,
                                   TipoRelacao.CONTRAINDICADO_EM, cond,
                                   TipoEntidade.CONDICAO, chunk_id=c.id)
                        )
        if c.secao == "Reações adversas" and tipo_suj is TipoEntidade.MEDICAMENTO:
            m = re.search(r"relatadas são (.+?)\.", c.texto)
            if m:
                for ef in re.split(r",| e ", m.group(1)):
                    ef = ef.strip()
                    if ef:
                        triplas.append(
                            Tripla(nome_suj, TipoEntidade.MEDICAMENTO, TipoRelacao.CAUSA,
                                   ef, TipoEntidade.EFEITO, chunk_id=c.id)
                        )

        # --- Metabolismo: inibe / induz / substrato / pró-fármaco ---
        if c.secao == "Metabolismo" and tipo_suj is TipoEntidade.MEDICAMENTO:
            for frase in re.split(r"(?<=\.)\s+", c.texto):
                enzs = gaz.enzimas(frase)
                if not enzs:
                    continue
                nf = _norm(frase)
                if "e um inibidor" in nf:
                    rel = TipoRelacao.INIBE
                elif "e um indutor" in nf:
                    rel = TipoRelacao.INDUZ
                elif "e um pro-farmaco" in nf or "bioativado" in nf:
                    rel = TipoRelacao.ATIVADO_POR
                elif "metabolizado" in nf or "substrato" in nf:
                    rel = TipoRelacao.SUBSTRATO_DE
                else:
                    continue
                inten = "moderada"
                if mi := re.search(r"intensidade (\w+)", nf):
                    inten = mi.group(1)
                elif re.search(r"sensibilidade (\w+)", nf):
                    inten = "sensivel"
                for e in enzs:
                    triplas.append(
                        Tripla(nome_suj, TipoEntidade.MEDICAMENTO, rel, e,
                               TipoEntidade.ENZIMA, chunk_id=c.id,
                               atributos={"intensidade": inten})
                    )

        # --- Interações asseridas ---
        if c.secao in ("Interações medicamentosas", "Interações de classe"):
            attrs = _atributos_interacao(c.texto)
            mc = _RE_CLASSE_ALVO.search(c.texto)
            if mc and (r := gaz.resolver(mc.group(1))):
                rel = (
                    TipoRelacao.INTERAGE_COM_CLASSE
                    if tipo_suj is TipoEntidade.MEDICAMENTO
                    else TipoRelacao.INTERAGE_COM_CLASSE
                )
                triplas.append(
                    Tripla(nome_suj, tipo_suj, rel, r[1], TipoEntidade.CLASSE,
                           chunk_id=c.id, atributos=attrs)
                )
            else:
                mm = _RE_MED_ALVO.search(c.texto)
                if mm and (r := gaz.resolver(mm.group(1))):
                    if r[0] is TipoEntidade.MEDICAMENTO and tipo_suj is TipoEntidade.MEDICAMENTO:
                        triplas.append(
                            Tripla(nome_suj, TipoEntidade.MEDICAMENTO, TipoRelacao.INTERAGE_COM,
                                   r[1], TipoEntidade.MEDICAMENTO, chunk_id=c.id,
                                   atributos=attrs)
                        )

    return validar_triplas(triplas)


# --------------------------------------------------------------------------
# Extrator por LLM (opcional)
# --------------------------------------------------------------------------

_PROMPT_EXTRACAO = """Você é um extrator de conhecimento farmacológico. Extraia triplas do texto abaixo.

Tipos de entidade permitidos: Medicamento, Classe, Marca, Enzima, Condicao, Efeito.
Relações permitidas e suas assinaturas:
  E_DA_CLASSE          Medicamento -> Classe
  TEM_MARCA            Medicamento -> Marca
  INIBE                Medicamento -> Enzima
  INDUZ                Medicamento -> Enzima
  SUBSTRATO_DE         Medicamento -> Enzima
  ATIVADO_POR          Medicamento -> Enzima   (apenas pró-fármacos)
  INTERAGE_COM         Medicamento -> Medicamento
  INTERAGE_COM_CLASSE  Medicamento -> Classe | Classe -> Classe
  INDICADO_PARA        Medicamento -> Condicao
  CONTRAINDICADO_EM    Medicamento -> Condicao
  CAUSA                Medicamento -> Efeito

Regras:
- Extraia SOMENTE o que está literalmente afirmado no texto. Não infira interações.
- Use o princípio ativo canônico como sujeito, nunca o nome comercial.
- Em relações de interação, inclua os atributos severidade (contraindicado|grave|moderada|leve),
  mecanismo, efeito, manejo e evidencia (alta|moderada|baixa) quando presentes.

Responda apenas o JSON no formato do esquema fornecido.

TEXTO (seção "{secao}" de "{titulo}"):
{texto}"""

_ESQUEMA_SAIDA = {
    "type": "object",
    "properties": {
        "triplas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sujeito": {"type": "string"},
                    "tipo_sujeito": {"type": "string", "enum": [t.value for t in TipoEntidade]},
                    "relacao": {"type": "string", "enum": [r.value for r in TipoRelacao]},
                    "objeto": {"type": "string"},
                    "tipo_objeto": {"type": "string", "enum": [t.value for t in TipoEntidade]},
                    "atributos": {
                        "type": "object",
                        "properties": {
                            "severidade": {"type": "string"},
                            "mecanismo": {"type": "string"},
                            "efeito": {"type": "string"},
                            "manejo": {"type": "string"},
                            "evidencia": {"type": "string"},
                            "intensidade": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["sujeito", "tipo_sujeito", "relacao", "objeto", "tipo_objeto"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triplas"],
    "additionalProperties": False,
}


def extrair_llm(chunks: Iterable[Chunk], gaz: Gazetteer, modelo: str | None = None) -> list[Tripla]:
    """Extração via Claude. Requer ANTHROPIC_API_KEY."""
    import anthropic

    from .config import MODELO_LLM

    cliente = anthropic.Anthropic()
    modelo = modelo or MODELO_LLM
    triplas: list[Tripla] = []

    for c in chunks:
        if c.secao == "Conservação":
            continue  # sem conteúdo farmacológico
        resp = cliente.messages.create(
            model=modelo,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _ESQUEMA_SAIDA}},
            messages=[{
                "role": "user",
                "content": _PROMPT_EXTRACAO.format(secao=c.secao, titulo=c.titulo, texto=c.texto),
            }],
        )
        if resp.stop_reason == "refusal":
            continue
        bruto = next((b.text for b in resp.content if b.type == "text"), "{}")
        for t in json.loads(bruto).get("triplas", []):
            # canonicaliza sujeito/objeto pelo gazetteer quando possível
            rs = gaz.resolver(t["sujeito"])
            ro = gaz.resolver(t["objeto"])
            triplas.append(
                Tripla(
                    sujeito=rs[1] if rs else t["sujeito"],
                    tipo_sujeito=TipoEntidade(t["tipo_sujeito"]),
                    relacao=TipoRelacao(t["relacao"]),
                    objeto=ro[1] if ro else t["objeto"],
                    tipo_objeto=TipoEntidade(t["tipo_objeto"]),
                    chunk_id=c.id,
                    origem=OrigemAresta.ASSERIDA,
                    atributos=t.get("atributos", {}),
                )
            )
        # arestas MENCIONA continuam vindo do gazetteer (barato e exaustivo)
        for m in gaz.medicamentos(c.texto):
            triplas.append(Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, m,
                                  TipoEntidade.MEDICAMENTO, chunk_id=c.id))
        for cl in gaz.classes(c.texto):
            triplas.append(Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, cl,
                                  TipoEntidade.CLASSE, chunk_id=c.id))
        for en in gaz.enzimas(c.texto):
            triplas.append(Tripla(c.id, TipoEntidade.CHUNK, TipoRelacao.MENCIONA, en,
                                  TipoEntidade.ENZIMA, chunk_id=c.id))

    return validar_triplas(triplas)


# --------------------------------------------------------------------------
# Validação e avaliação da extração
# --------------------------------------------------------------------------


def validar_triplas(triplas: Iterable[Tripla]) -> list[Tripla]:
    """Descarta triplas fora do esquema e deduplica."""
    vistos: set[tuple] = set()
    saida: list[Tripla] = []
    for t in triplas:
        if not t.valida():
            continue
        chave = (t.sujeito, t.relacao.value, t.objeto, t.chunk_id)
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(t)
    return saida


def gabarito_da_semente() -> set[tuple[str, str, str]]:
    """Triplas farmacológicas esperadas, derivadas diretamente da semente.

    Serve de gabarito para medir a extração. Arestas MENCIONA e
    INDICADO_PARA/CAUSA (listas livres, sujeitas a segmentação) ficam de
    fora: o alvo aqui são as relações estruturais que sustentam o
    raciocínio (classe, metabolismo, interações).
    """
    meds = json.loads(MEDICAMENTOS_JSON.read_text(encoding="utf-8"))["medicamentos"]
    inter = json.loads(INTERACOES_JSON.read_text(encoding="utf-8"))
    g: set[tuple[str, str, str]] = set()

    for m in meds:
        g.add((m["principio_ativo"], TipoRelacao.E_DA_CLASSE.value, m["classe"]))
        for b in m["nomes_comerciais"]:
            g.add((m["principio_ativo"], TipoRelacao.TEM_MARCA.value, b))

    papel_rel = {
        "inibidor": TipoRelacao.INIBE.value,
        "indutor": TipoRelacao.INDUZ.value,
        "substrato": TipoRelacao.SUBSTRATO_DE.value,
        "pro_farmaco": TipoRelacao.ATIVADO_POR.value,
    }
    for m in inter["metabolismo"]:
        g.add((m["medicamento"], papel_rel[m["papel"]], m["enzima"]))

    for i in inter["interacoes_farmaco_farmaco"]:
        suj = i["documentado_em"]
        obj = i["b"] if i["a"] == suj else i["a"]
        g.add((suj, TipoRelacao.INTERAGE_COM.value, obj))
    for i in inter["interacoes_farmaco_classe"]:
        if i["documentado_em"].startswith("classe:"):
            continue
        g.add((i["documentado_em"], TipoRelacao.INTERAGE_COM_CLASSE.value, i["classe"]))
    for i in inter["interacoes_classe_classe"]:
        suj = i["documentado_em"].removeprefix("classe:")
        obj = i["classe_b"] if i["classe_a"] == suj else i["classe_a"]
        g.add((suj, TipoRelacao.INTERAGE_COM_CLASSE.value, obj))
    return g


_RELACOES_AVALIADAS = {
    TipoRelacao.E_DA_CLASSE.value,
    TipoRelacao.TEM_MARCA.value,
    TipoRelacao.INIBE.value,
    TipoRelacao.INDUZ.value,
    TipoRelacao.SUBSTRATO_DE.value,
    TipoRelacao.ATIVADO_POR.value,
    TipoRelacao.INTERAGE_COM.value,
    TipoRelacao.INTERAGE_COM_CLASSE.value,
}


def avaliar_extracao(triplas: list[Tripla]) -> dict:
    """Precisão / recall / F1 da extração contra o gabarito da semente."""
    gab = gabarito_da_semente()
    pred = {
        (t.sujeito, t.relacao.value, t.objeto)
        for t in triplas
        if t.relacao.value in _RELACOES_AVALIADAS and t.origem is OrigemAresta.ASSERIDA
    }
    vp = len(pred & gab)
    fp = len(pred - gab)
    fn = len(gab - pred)
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "verdadeiros_positivos": vp,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "precisao": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "exemplos_fp": sorted(pred - gab)[:5],
        "exemplos_fn": sorted(gab - pred)[:5],
    }


def executar(extrator: Literal["gazetteer", "llm"] = "gazetteer", salvar: bool = True) -> list[Tripla]:
    chunks = carregar_chunks()
    gaz = Gazetteer.a_partir_da_semente()
    triplas = extrair_gazetteer(chunks, gaz) if extrator == "gazetteer" else extrair_llm(chunks, gaz)
    if salvar:
        garantir_diretorios()
        TRIPLAS_JSON.write_text(
            json.dumps([t.to_dict() for t in triplas], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return triplas


def carregar_triplas() -> list[Tripla]:
    return [Tripla.from_dict(d) for d in json.loads(TRIPLAS_JSON.read_text(encoding="utf-8"))]


if __name__ == "__main__":
    import sys

    ext = sys.argv[1] if len(sys.argv) > 1 else "gazetteer"
    ts = executar(ext)  # type: ignore[arg-type]
    por_rel: dict[str, int] = defaultdict(int)
    for t in ts:
        por_rel[t.relacao.value] += 1
    print(f"extração ({ext}): {len(ts)} triplas")
    for r, n in sorted(por_rel.items(), key=lambda x: -x[1]):
        print(f"  {r:<22} {n}")
    print("\nqualidade da extração vs. semente:")
    for k, v in avaliar_extracao(ts).items():
        print(f"  {k}: {v}")
