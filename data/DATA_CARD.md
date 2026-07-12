# Data Card — Base de conhecimento HYGIA

## Resumo

Base curada de **medicamentos, vias metabólicas e interações medicamentosas**
de uso corrente no Brasil, construída para demonstrar raciocínio multi-hop em
Graph RAG. A partir dela é gerado um **corpus documental sintético** (bulas e
monografias) sobre o qual todo o pipeline opera.

> ⚠️ **Uso exclusivamente educacional e de pesquisa.** Não é um dispositivo
> médico, não foi validada clinicamente e não deve orientar decisões de
> prescrição. A conduta clínica é sempre do profissional responsável.

## Origem e licença

| Item | Descrição |
|------|-----------|
| **Fontes de referência** | Conteúdo farmacológico consolidado a partir de fontes públicas de referência: Bulário Eletrônico da ANVISA, Formulário Terapêutico Nacional (Ministério da Saúde) e literatura farmacológica clássica (Goodman & Gilman; Rang & Dale). |
| **Natureza** | Os arquivos-semente (`data/seed/*.json`) foram **redigidos manualmente** consolidando fatos amplamente documentados; não reproduzem texto proprietário de bula. |
| **Corpus** | O corpus (`data/corpus/chunks.json`) é **100% sintético**, gerado por template a partir da semente (`src/hygia/corpus.py`). Nenhum texto de bula real foi copiado. |
| **Licença dos dados** | Os arquivos-semente e o corpus gerado são liberados sob **CC BY 4.0**, junto ao código (MIT). |

## Composição

| Arquivo | Conteúdo | Volume |
|---------|----------|--------|
| `data/seed/medicamentos.json` | Princípios ativos, sinônimos, marcas, classe, ATC, indicações, contraindicações, reações | 52 medicamentos |
| `data/seed/interacoes.json` | Enzimas (7), papéis metabólicos (29), regras de inferência (4), interações fármaco-fármaco (15), fármaco-classe (15), classe-classe (15) | 45 interações + 4 regras |
| `data/corpus/chunks.json` | Corpus documental gerado (bulas + monografias de classe + monografias de enzima) | 391 chunks / 101 documentos |
| `data/gold/perguntas.json` | Conjunto de avaliação com gabarito de recuperação | 35 perguntas |

### Cobertura por área terapêutica

Anticoagulantes/antiagregantes, anti-inflamatórios, antidepressivos (ISRS,
tricíclicos, IMAO), antibióticos (macrolídeos, quinolonas, sulfonamidas),
antifúngicos, antidiabéticos, anti-hipertensivos (IECA, BRA, diuréticos,
betabloqueadores, bloqueadores de cálcio), estatinas, anticonvulsivantes,
benzodiazepínicos, hormônios tireoidianos, imunossupressores.

## Decisão de projeto: documentação assimétrica

Cada interação **fármaco-fármaco é documentada na bula de apenas um dos dois
lados** (campo `documentado_em`). Isso reproduz a realidade — a bula da
varfarina cita AINEs, a do ibuprofeno não cita varfarina — e é o que torna o
problema um teste genuíno de raciocínio multi-hop, e não de recuperação por
similaridade. O corpus **é completo**: todo fato necessário está escrito em
algum chunk; o desafio é *compor* fatos dispersos.

## Qualidade e integridade

- **Verificação automática de completude** (`corpus._checar_integridade`): o
  build falha se qualquer interação da semente não gerar um chunk.
- **Extração validada contra a semente**: precisão/recall/F1 medidos em
  `extract.avaliar_extracao` — atualmente **F1 = 1.0** (extrator gazetteer).
- **Gold set validado**: todos os chunks-gabarito existem no corpus (testado).

## Vieses e limitações conhecidos

- **Escopo curado e limitado**: 52 fármacos. Ausência de interação na base
  **não** significa segurança — apenas ausência de registro.
- **Severidades simplificadas**: quatro níveis (contraindicado/grave/moderada/
  leve); a prática clínica considera dose, via, comorbidades e genética.
- **Regras mecanísticas didáticas**: o motor de inferência (R1–R4) modela CYP,
  UGT e xantina oxidase de forma simplificada; não cobre glicoproteína-P,
  transportadores, farmacogenômica nem interações farmacodinâmicas completas.
- **Idioma único** (português) e **contexto Brasil** (marcas comerciais).
- **Sem dados de pacientes**: nenhuma informação pessoal ou identificável.

## Reprodutibilidade

Todo o dado derivado é regenerável de forma determinística (semente fixa em
`hygia.config.SEED = 42`):

```bash
make pipeline   # semente -> corpus -> triplas -> grafo -> índice -> gold -> avaliação
```

SHA256 dos artefatos-semente (referência):

```
a283eaeb6de477498423e840eedb44ebecad01d23298fd12eb214020da283a1b  data/seed/medicamentos.json
fab88980f2347251db9b4d84b6b010dbe7b54b46f35c11cd8a41a4c72aa007d6  data/seed/interacoes.json
```

Confira com `sha256sum data/seed/*.json`.
