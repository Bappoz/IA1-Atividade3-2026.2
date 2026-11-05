# Portfólio de IA 1 — Tratando Incerteza

Portfólio reproduzível sobre **monitoramento inteligente e diagnóstico de falhas em sistemas industriais**. Os três projetos são independentes, mas usam o mesmo cenário conceitual de uma máquina instrumentada:

**Relatório principal:** `report_geral.pdf`, com identificação acadêmica, proposta integrada, metodologia, resultados dos três projetos, discussão comparativa e referências. Os PDFs dentro dos subdiretórios são relatórios técnicos auxiliares de cada projeto.

1. `projeto_1_redes_bayesianas`: diagnóstico probabilístico por enumeração exata em uma Rede Bayesiana.
2. `projeto_2_hmm`: identificação temporal de estados operacionais por HMM, com Forward escalonado, Viterbi e Baum–Welch.
3. `projeto_3_kalman`: estimação de posição e velocidade por Filtro de Kalman linear com atualização de Joseph.

## Execução

```bash
python3 projeto_1_redes_bayesianas/main.py
python3 projeto_2_hmm/main.py
python3 projeto_3_kalman/main.py
pytest -q
```

Cada `main.py` grava figuras em `results/` e métricas em `metrics.json`. Para recompilar o relatório geral:

```bash
pdflatex -interaction=nonstopmode report_geral.tex
```

As simulações são sintéticas, documentadas e usam sementes fixas. Os PDFs entregues foram gerados a partir dos valores registrados pelos próprios experimentos.
