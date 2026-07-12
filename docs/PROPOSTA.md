# Proposta de projeto — HYGIA

## Problema

Interações medicamentosas potencialmente perigosas (DDI, *drug–drug
interactions*) são uma das causas evitáveis mais frequentes de eventos adversos
a medicamentos. A informação para preveni-las **existe** — está nas bulas e em
compêndios farmacológicos — mas é **fragmentada**: distribuída por documentos
diferentes, descrita de forma assimétrica (a bula de A cita B, a de B não cita
A) e, no caso de interações mecanísticas (via enzimas do citocromo P450), muitas
vezes **não está escrita em documento algum** — precisa ser derivada.

### Por que é um problema de IA (e não uma consulta a tabela)

1. **Composição de evidência dispersa.** Responder "posso associar varfarina e
   ibuprofeno?" exige unir a bula da varfarina (que cita a *classe* dos AINEs) à
   monografia da classe (que lista o ibuprofeno). Nenhum documento único contém
   os dois nomes.
2. **Raciocínio mecanístico.** "Claritromicina + sinvastatina" é contraindicado
   porque a primeira inibe a CYP3A4 e a segunda depende dela — uma inferência
   sobre a estrutura do conhecimento, não uma correspondência de texto.
3. **Explicabilidade obrigatória.** Em apoio à decisão clínica, uma resposta sem
   o *porquê* rastreável é inutilizável. O sistema precisa mostrar o caminho.

Esses três pontos são exatamente o que **Graph RAG** endereça e o que RAG por
similaridade não consegue.

## Motivação

- **Impacto.** DDIs estão entre as causas mais comuns e evitáveis de
  hospitalização por evento adverso; ferramentas de apoio que expliquem o
  raciocínio aumentam a confiança e a adesão do prescritor.
- **Adequação técnica.** O domínio é naturalmente um grafo (medicamento–classe–
  enzima–condição), com relações tipadas e regras mecanísticas bem estabelecidas
  — um caso-modelo para demonstrar a diferença entre recuperar *chunks* e
  recuperar *relações*.
- **Diferencial de portfólio.** Em vez de "mais um RAG", o projeto isola e
  **mede** com rigor estatístico *onde* e *por quê* o grafo ganha, com ablação e
  baselines honestos.

## Escopo

**Dentro do escopo:**
- 52 medicamentos de uso corrente no Brasil, 7 vias enzimáticas, 45 interações.
- Quatro categorias de raciocínio: fato direto, mecanismo enzimático, ponte de
  classe (2 saltos), classe↔classe (3 saltos).
- Pipeline completo dos 7 passos do Graph RAG, comparado a 3 baselines.
- Interface de verificação com visualização do caminho de raciocínio.

**Fora do escopo (limitações assumidas):**
- Interações farmacodinâmicas complexas, glicoproteína-P, transportadores,
  farmacogenômica individual.
- Ajuste por dose, via de administração, comorbidades, função renal/hepática.
- Validação clínica — este é um protótipo de pesquisa.

## Hipóteses (explícitas e testáveis)

- **H1.** Em perguntas multi-hop (mecanismo, ponte de classe, classe↔classe), o
  Graph RAG tem cobertura de fatos significativamente maior que o melhor baseline
  (plain RAG). → *Confirmada*: 1.00 vs 0.20, _p_ = 0.0002.
- **H2.** Em perguntas de salto único, o Graph RAG **não** é pior que o baseline
  (não há custo de adicionar o grafo). → *Confirmada*: empate em 1.00.
- **H3.** O ganho em interações mecanísticas depende do motor de inferência, não
  apenas da caminhada no grafo. → *Confirmada por ablação*: sem inferência, a
  categoria mecanismo cai de 1.00 para 0.38.

## Metodologia (resumo)

Ver [`METODOLOGIA.md`](METODOLOGIA.md) para o detalhamento. Em síntese:
ontologia declarativa → extração de triplas validada contra a semente →
construção do grafo com motor de inferência mecanística (regras R1–R4) →
indexação híbrida (LSA + BM25) → recuperação por Reciprocal Rank Fusion sobre
texto **e** grafo → geração com Claude → avaliação por categoria com bootstrap e
teste de significância, mais estudo de ablação.

## Cronograma (execução do protótipo)

| Etapa | Entregável |
|-------|-----------|
| 1. Modelagem do domínio | Ontologia + base curada (semente) + Data Card |
| 2. Corpus e extração | Gerador de corpus + extrator (F1 medido vs. semente) |
| 3. Grafo e inferência | Grafo `networkx` + motor de regras mecanísticas |
| 4. Recuperação | Baselines + Graph RAG sob interface comum |
| 5. Avaliação | Gold set, métricas, bootstrap, significância, ablação |
| 6. Interface e reprodutibilidade | Streamlit, Makefile, CI, testes, documentação |

## Resultados esperados × obtidos

| Esperado | Obtido |
|----------|--------|
| Ganho multi-hop mensurável e significativo | Δ cobertura +0.80, _p_ = 0.0002 |
| Sem regressão em salto único | Empate em 1.00 |
| Respostas auditáveis (caminho de raciocínio) | Todo achado carrega `A → via → B` |
| Pipeline reprodutível ponta a ponta | `make pipeline` determinístico em ~4 s; CI verde |
