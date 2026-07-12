"""Resolução de entidades da pergunta (linguagem natural → nós do grafo).

Compartilhada por todos os recuperadores, para que a comparação seja
justa: a diferença medida entre baseline e Graph RAG vem do uso do grafo,
não de reconhecer melhor os nomes na pergunta.
"""

from __future__ import annotations

from .extract import Gazetteer
from .schema import TipoEntidade


class ResolvedorEntidades:
    def __init__(self, gaz: Gazetteer | None = None):
        self.gaz = gaz or Gazetteer.a_partir_da_semente()

    def resolver(self, texto: str) -> list[tuple[TipoEntidade, str]]:
        """Extrai medicamentos e classes citados na pergunta (canonicalizados)."""
        ents: list[tuple[TipoEntidade, str]] = []
        vistos: set[str] = set()
        for nome in self.gaz.medicamentos(texto):
            chave = f"med::{nome}"
            if chave not in vistos:
                vistos.add(chave)
                ents.append((TipoEntidade.MEDICAMENTO, nome))
        for nome in self.gaz.classes(texto):
            chave = f"cls::{nome}"
            if chave not in vistos:
                vistos.add(chave)
                ents.append((TipoEntidade.CLASSE, nome))
        return ents
