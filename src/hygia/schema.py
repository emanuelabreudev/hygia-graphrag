"""PASSO 1 do Graph RAG — Ontologia: tipos de entidade e de relação.

Um esquema limpo é 80% de um grafo útil. Aqui ele é declarado de forma
explícita e validável, não implícito no código de extração.

Modelo de domínio (segurança medicamentosa):

    (Medicamento)-[:E_DA_CLASSE]->(Classe)
    (Medicamento)-[:TEM_MARCA]->(Marca)
    (Medicamento)-[:INIBE|INDUZ|SUBSTRATO_DE|ATIVADO_POR]->(Enzima)
    (Medicamento)-[:INTERAGE_COM]->(Medicamento)
    (Medicamento)-[:INTERAGE_COM_CLASSE]->(Classe)
    (Classe)-[:INTERAGE_COM_CLASSE]->(Classe)
    (Medicamento)-[:INDICADO_PARA|CONTRAINDICADO_EM]->(Condicao)
    (Medicamento)-[:CAUSA]->(Efeito)
    (Interacao)-[:EVIDENCIADA_POR]->(Chunk)

A aresta INTERAGE_COM pode ser *asserida* (existe um documento que a
declara) ou *inferida* (derivada por regra mecanística sobre o grafo).
Essa distinção é carregada no atributo `origem` e é o que permite o
raciocínio multi-hop auditável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TipoEntidade(str, Enum):
    MEDICAMENTO = "Medicamento"
    CLASSE = "Classe"
    MARCA = "Marca"
    ENZIMA = "Enzima"
    CONDICAO = "Condicao"
    EFEITO = "Efeito"
    INTERACAO = "Interacao"
    CHUNK = "Chunk"


class TipoRelacao(str, Enum):
    E_DA_CLASSE = "E_DA_CLASSE"
    TEM_MARCA = "TEM_MARCA"
    INIBE = "INIBE"
    INDUZ = "INDUZ"
    SUBSTRATO_DE = "SUBSTRATO_DE"
    ATIVADO_POR = "ATIVADO_POR"
    INTERAGE_COM = "INTERAGE_COM"
    INTERAGE_COM_CLASSE = "INTERAGE_COM_CLASSE"
    INDICADO_PARA = "INDICADO_PARA"
    CONTRAINDICADO_EM = "CONTRAINDICADO_EM"
    CAUSA = "CAUSA"
    EVIDENCIADA_POR = "EVIDENCIADA_POR"
    MENCIONA = "MENCIONA"


class Severidade(str, Enum):
    CONTRAINDICADO = "contraindicado"
    GRAVE = "grave"
    MODERADA = "moderada"
    LEVE = "leve"

    @property
    def peso(self) -> int:
        return {"contraindicado": 4, "grave": 3, "moderada": 2, "leve": 1}[self.value]


class OrigemAresta(str, Enum):
    """De onde veio a aresta — chave para explicabilidade."""

    ASSERIDA = "asserida"  # extraída literalmente de um documento
    INFERIDA = "inferida"  # derivada por regra mecanística sobre o grafo


# Assinaturas permitidas: (tipo_origem, relacao, tipo_destino).
# Extrações que violem o esquema são rejeitadas em extract.validar_triplas.
ASSINATURAS: set[tuple[TipoEntidade, TipoRelacao, TipoEntidade]] = {
    (TipoEntidade.MEDICAMENTO, TipoRelacao.E_DA_CLASSE, TipoEntidade.CLASSE),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.TEM_MARCA, TipoEntidade.MARCA),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.INIBE, TipoEntidade.ENZIMA),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.INDUZ, TipoEntidade.ENZIMA),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.SUBSTRATO_DE, TipoEntidade.ENZIMA),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.ATIVADO_POR, TipoEntidade.ENZIMA),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.INTERAGE_COM, TipoEntidade.MEDICAMENTO),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.INTERAGE_COM_CLASSE, TipoEntidade.CLASSE),
    (TipoEntidade.CLASSE, TipoRelacao.INTERAGE_COM_CLASSE, TipoEntidade.CLASSE),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.INDICADO_PARA, TipoEntidade.CONDICAO),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.CONTRAINDICADO_EM, TipoEntidade.CONDICAO),
    (TipoEntidade.MEDICAMENTO, TipoRelacao.CAUSA, TipoEntidade.EFEITO),
    (TipoEntidade.INTERACAO, TipoRelacao.EVIDENCIADA_POR, TipoEntidade.CHUNK),
    (TipoEntidade.CHUNK, TipoRelacao.MENCIONA, TipoEntidade.MEDICAMENTO),
    (TipoEntidade.CHUNK, TipoRelacao.MENCIONA, TipoEntidade.CLASSE),
    (TipoEntidade.CHUNK, TipoRelacao.MENCIONA, TipoEntidade.ENZIMA),
}


@dataclass(frozen=True)
class Tripla:
    """(sujeito, predicado, objeto) + proveniência.

    `chunk_id` é a evidência textual. Uma tripla sem chunk_id só é aceita
    se `origem == INFERIDA` (nasceu de uma regra, não de um texto).
    """

    sujeito: str
    tipo_sujeito: TipoEntidade
    relacao: TipoRelacao
    objeto: str
    tipo_objeto: TipoEntidade
    chunk_id: str | None = None
    origem: OrigemAresta = OrigemAresta.ASSERIDA
    atributos: dict[str, Any] = field(default_factory=dict)

    @property
    def assinatura(self) -> tuple[TipoEntidade, TipoRelacao, TipoEntidade]:
        return (self.tipo_sujeito, self.relacao, self.tipo_objeto)

    def valida(self) -> bool:
        if self.assinatura not in ASSINATURAS:
            return False
        if self.origem is OrigemAresta.ASSERIDA and not self.chunk_id:
            return False
        return bool(self.sujeito and self.objeto)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sujeito": self.sujeito,
            "tipo_sujeito": self.tipo_sujeito.value,
            "relacao": self.relacao.value,
            "objeto": self.objeto,
            "tipo_objeto": self.tipo_objeto.value,
            "chunk_id": self.chunk_id,
            "origem": self.origem.value,
            "atributos": self.atributos,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Tripla":
        return cls(
            sujeito=d["sujeito"],
            tipo_sujeito=TipoEntidade(d["tipo_sujeito"]),
            relacao=TipoRelacao(d["relacao"]),
            objeto=d["objeto"],
            tipo_objeto=TipoEntidade(d["tipo_objeto"]),
            chunk_id=d.get("chunk_id"),
            origem=OrigemAresta(d.get("origem", "asserida")),
            atributos=d.get("atributos", {}),
        )


def no_id(tipo: TipoEntidade, nome: str) -> str:
    """Identificador canônico de nó: 'Tipo::nome-normalizado'."""
    return f"{tipo.value}::{nome.strip().lower()}"
