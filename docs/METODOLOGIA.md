# Metodologia

Este documento descreve o pipeline em profundidade, formaliza o motor de
inferência e justifica as escolhas de projeto que tornam a comparação honesta.

---

## 1. Ontologia (Passo 1) — `schema.py`

O grafo modela seis tipos de entidade e treze de relação, declarados como enums
e restritos por um conjunto de **assinaturas permitidas** `(tipo_sujeito,
relação, tipo_objeto)`. Toda tripla extraída é validada contra esse conjunto;
uma violação é descartada em `extract.validar_triplas`. Isso mantém o grafo
consistente por construção — "um esquema limpo é 80% de um grafo útil".

Distinção-chave: a aresta `INTERAGE_COM` carrega o atributo **`origem`**:
- `ASSERIDA` — extraída literalmente de um documento (tem `chunk_id`);
- `INFERIDA` — derivada por regra mecanística (tem o caminho que a justifica).

Essa distinção é o que permite auditar cada resposta.

---

## 2. Corpus e extração (Passos 0 e 2)

### Geração do corpus — `corpus.py`

O corpus é sintético e gerado por template a partir da semente. Três decisões
tornam o benchmark defensável:

1. **Completude.** Todo fato necessário para responder qualquer pergunta do gold
   set está escrito em algum chunk. Uma verificação automática
   (`_checar_integridade`) falha o build se qualquer interação da semente não
   virar um chunk. Isso impede que a vantagem do grafo venha de informação
   escondida do baseline.
2. **Assimetria realista.** Cada interação fármaco-fármaco aparece na bula de
   apenas um dos lados (`documentado_em`). É o que cria o multi-hop.
3. **Distratores.** Cada bula tem uma seção de "Conservação" irrelevante, como
   bulas reais — ruído que o recuperador precisa evitar.

### Extração — `extract.py`

Duas implementações sob a mesma saída (`list[Tripla]`):
- **Gazetteer** (padrão): NER por dicionário do domínio + regras sobre padrões
  linguísticos. Determinístico, sem custo, roda no CI.
- **LLM**: Claude extrai triplas com saída estruturada (`output_config.format`).
  Mais robusto a variação linguística; requer `ANTHROPIC_API_KEY`.

**A extração roda sobre o texto, não sobre o JSON semente.** Isso permite
medir sua qualidade contra a semente como gabarito (`avaliar_extracao`):
precisão, recall e F1 sobre as relações estruturais. Resultado atual do
gazetteer: **P = 1.00, R = 1.00, F1 = 1.00** (240 triplas estruturais).

---

## 3. Grafo e motor de inferência (Passo 3) — `graph.py`

O grafo é um `networkx.MultiDiGraph`. Sobre as arestas asseridas, um **motor de
inferência** deriva interações mecanísticas percorrendo pares
(modulador, alvo) que compartilham uma enzima.

### Formalização das regras (R1–R4)

Sejam `A` e `B` medicamentos e `E` uma enzima. Notação: `A --r--> E` é uma aresta
de relação `r`.

| Regra | Antecedente | Consequente (interação `A`↔`B`) | Severidade |
|-------|-------------|-------------------------------|------------|
| **R1** | `A --INIBE--> E` (forte/moderada) ∧ `B --SUBSTRATO_DE--> E` | Acúmulo de `B` por queda de depuração | grave; **contraindicado** se inibição forte ∧ substrato sensível |
| **R2** | `A --INDUZ--> E` (forte) ∧ `B --SUBSTRATO_DE--> E` | Aceleração da depuração de `B` → perda de eficácia | grave |
| **R3** | `A --INIBE--> E` ∧ `B --ATIVADO_POR--> E` | Bloqueio da bioativação do pró-fármaco `B` → falha terapêutica | **contraindicado** (inibição forte) / grave |
| **R4** | `A --INIBE--> XO` ∧ `B --SUBSTRATO_DE/ATIVADO_POR--> XO` | Acúmulo do metabólito citotóxico | **contraindicado** |

As regras são declaradas em `data/seed/interacoes.json` (`regras_mecanismo`) e
aplicadas em `inferir_mecanismos`. Cada aresta inferida guarda a enzima, a regra
e o mecanismo textual — é o que alimenta o caminho de raciocínio.

**Controle de falsos positivos.** R1 exige intensidade **forte ou moderada** de
inibição (inibições fracas são ignoradas); R2 exige indução **forte**. Sem esse
filtro, cada substrato compartilhado geraria um alerta, inundando o usuário de
ruído clinicamente irrelevante.

### Consulta de par — `analisar_par`

Dado (A, B), a função busca **quatro tipos de caminho**, em ordem de evidência:

```
[1] direta        A --INTERAGE_COM--> B                      (asserida)
[2] mecanismo     A --INIBE/INDUZ--> E <--SUBSTRATO_DE-- B   (inferida)
[3] classe        A --INTERAGE_COM_CLASSE--> C <--E_DA_CLASSE-- B
[4] classe×classe A --E_DA_CLASSE--> Ca --...--> Cb <--E_DA_CLASSE-- B
```

Os achados são deduplicados e ordenados por severidade (decrescente) e nº de
saltos (crescente). Cada um carrega `caminho` e `chunks` (evidência), garantindo
explicabilidade.

---

## 4. Indexação (Passo 4) — `index.py`

Duas representações autocontidas (sem chamada de rede, para reprodutibilidade e
justiça):
- **Vetorial**: TF-IDF (uni+bigramas) → LSA por SVD truncada (d=128), cosseno.
- **Léxica**: BM25 clássico (k1=1.5, b=0.75).

O índice é **determinístico** (semente fixa); `Indice.carregar()` retreina bit a
bit em vez de desserializar objetos frágeis. Um teste verifica o determinismo.

---

## 5. Recuperação híbrida (Passo 5) — `retriever.py`

Quatro recuperadores sob a interface `recuperar(consulta, entidades) -> Resultado`:

| Recuperador | Fontes | Papel |
|-------------|--------|-------|
| `vetorial` | LSA | baseline denso |
| `bm25` | BM25 | baseline léxico |
| `hibrido` | LSA + BM25 (RRF) | **plain RAG** — baseline forte |
| `graphrag` | LSA + BM25 + caminhada no grafo (RRF) | proposta |

**Fusão por Reciprocal Rank Fusion** (`rrf`): robusta a escalas incompatíveis
(cosseno vs. BM25 vs. prioridade de grafo) — só a ordem importa.
`escore(d) = Σ_i w_i / (k + rank_i(d))`, com `k = 60`.

**Justiça da comparação.** As entidades âncora da pergunta são resolvidas por um
componente **compartilhado** (`resolver.py`); todos recebem a mesma consulta
textual. A única diferença entre `hibrido` e `graphrag` é o uso do grafo — é isso
que a comparação isola.

**O que o grafo acrescenta:**
1. A caminhada de vizinhança recupera chunks que evidenciam arestas do subgrafo
   — inclusive a monografia da classe do *outro* fármaco, que nenhuma
   representação textual aproximaria da consulta.
2. `analisar_par` produz achados estruturados (o "graph context" do Passo 6).

---

## 6. Avaliação (Passo 7) — `eval.py`

### Conjunto de avaliação — `gold.py`

35 perguntas geradas da semente, com **gabarito de recuperação** derivado do mapa
`fato → chunk` do próprio corpus (robusto a mudanças de ID). Quatro categorias:
`fato_direto` (6), `mecanismo` (8), `ponte_classe` (13), `ponte_classe2` (8).

### Métricas

- **recall@k / precision@k / MRR**: recuperação clássica.
- **hit@k**: todos os chunks-gabarito no top-k (respondibilidade).
- **cobertura de fatos** (métrica principal): a evidência para compor a resposta
  foi reunida? Para o Graph RAG, um achado estruturado entre os dois fármacos da
  pergunta conta como cobertura — captura a vantagem real do raciocínio, que é
  *compor*, não apenas trazer um chunk relevante.

### Rigor estatístico

- Resultados **por categoria** de raciocínio.
- **IC 95% por bootstrap** (2000 reamostragens) sobre as perguntas.
- **Teste de significância pareado** (bootstrap, 5000 reamostragens) entre Graph
  RAG e o melhor baseline, para cobertura e hit@k.

### Estudo de ablação — `ablation.py`

Remove um componente por vez:

| Configuração | Cobertura geral | Categoria mecanismo |
|--------------|:---:|:---:|
| Só híbrido (sem grafo) | 0.20 | 0.00 |
| Sem caminhada | 1.00 | 1.00 |
| **Sem inferência** | 0.86 | **0.38** |
| Completo | 1.00 | 1.00 |

**Leitura.** O motor de inferência é o que sustenta a categoria mecanismo (0.38
→ 1.00). A caminhada, isoladamente, contribui para o recall textual mas a
análise estruturada de pares carrega a cobertura — daí "sem caminhada" ainda
atingir 1.00 na cobertura, mas o recall@k textual do Graph RAG completo (0.605)
ser maior que o do híbrido (0.386).

### Análise de erros

- **BM25 acerta 8% das pontes de classe**: quando o nome do fármaco-membro por
  acaso aparece na monografia recuperada por sobreposição léxica. É coincidência
  de vocabulário, não raciocínio — e não escala para os outros tipos (0% em
  mecanismo e classe↔classe).
- **Risco de falso positivo mecanístico**: mitigado pelo filtro de intensidade
  (R1/R2). Uma extensão natural seria calibrar um limiar por evidência clínica.
- **Sensibilidade a `top_k`**: com `k` menor, o hit@k dos baselines cai mais
  rápido; a cobertura do Graph RAG é estável porque não depende do ranqueamento
  textual para os achados estruturados.

---

## 7. Reprodutibilidade

- Toda aleatoriedade deriva de `hygia.config.SEED = 42`.
- Dependências pinadas (`requirements.txt`); ambiente isolado por `uv`.
- Artefatos versionados em `data/processed/` e `resultados/`.
- CI roda o pipeline em modo rápido e a suíte de 15 testes a cada push.
- `make pipeline` reconstrói tudo, da semente aos resultados, em ~4 s.

---

## Decisões de projeto — perguntas frequentes

**"O corpus não foi feito para o grafo ganhar?"** Não. O corpus é *completo* — o
baseline tem toda a informação. A verificação de integridade garante isso. A
dificuldade é composicional, e replica a realidade das bulas (documentação
assimétrica). O empate em `fato_direto` é a prova: onde não há composição, todos
acertam.

**"Por que LSA e BM25 em vez de embeddings neurais?"** Para manter o pipeline
autocontido, determinístico e sem dependência de rede (CI reprodutível). O ponto
do experimento não é o embedding — é a *composição*. Um embedding melhor
aumentaria o recall de chunks isolados, mas não resolveria multi-hop: nenhum
trecho contém os dois fármacos. Trocar o índice denso por embeddings é uma
extensão trivial (`IndiceVetorial`), sem alterar a conclusão.

**"Por que gazetteer se há extrator por LLM?"** Para que a *extração* seja
medível e reprodutível no CI (F1 = 1.0 contra a semente). O extrator por LLM está
implementado e é selecionável; o gazetteer é o default determinístico.
