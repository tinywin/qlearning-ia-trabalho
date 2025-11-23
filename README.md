# Q-Learning – Implementação Completa para CliffWalking

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-arrays-013243?logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-visualizations-informational)](https://matplotlib.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20environment-brightgreen?logo=openaigym)](https://gymnasium.farama.org/)
[![Imageio](https://img.shields.io/badge/Imageio-GIF%20generation-orange)](https://imageio.readthedocs.io/)

---

Este projeto implementa o algoritmo **Q-Learning** aplicado ao ambiente **CliffWalking-v0** do Gymnasium/Gym.

O sistema usa técnicas de **aprendizado por reforço** para treinar um agente que deve navegar em uma grade evitando precipícios e alcançando o objetivo. O código inclui **política epsilon-greedy** com decaimento adaptativo, **visualização da Q-table**, **geração de GIF animado** e **avaliação completa de desempenho**.

---

## O que foi feito (explicação simples)

1. Foi implementado o algoritmo **Q-Learning** com política epsilon-greedy
2. O agente aprende a navegar no ambiente **CliffWalking** (grade 4x12)
3. A **Q-table** é construída iterativamente durante 500 episódios
4. São geradas **visualizações da política** aprendida com setas direcionais
5. Um **GIF animado** mostra a trajetória do agente treinado
6. A **taxa de sucesso** é avaliada em 100 episódios de teste

---

## O que é Q-Learning

O **Q-Learning** é um algoritmo de aprendizado por reforço que aprende a política ótima sem conhecer o modelo do ambiente.
Neste projeto, ele treina um agente para navegar de uma posição inicial **S** até o objetivo **G**, evitando células perigosas **C** (precipício).

A **Q-table** armazena valores de qualidade Q(s,a) para cada par estado-ação, sendo atualizada pela equação de Bellman.

---

## Métricas e avaliação

| Métrica                    | Explicação                                                     |
| :------------------------- | :------------------------------------------------------------- |
| **Taxa de Sucesso**       | Percentual de episódios que alcançam o objetivo sem falhar.    |
| **Epsilon Decay**         | Redução gradual da exploração (1.0 → 0.01) ao longo do tempo. |
| **Convergência Q-table**  | Estabilização dos valores Q após treinamento suficiente.       |
| **Política Ótima**        | Conjunto de ações que maximizam a recompensa esperada.         |

---

## Como usar

```powershell
pip install -r requirements.txt
python q_learning_cliffwalking.py
```

O script treina o agente, exibe a Q-table, mostra a política com setas, gera um GIF da trajetória e avalia o desempenho final.

---

## Estrutura do projeto

```
q_learning_cliffwalking.py
README.md
requirements.txt
trajeto_agente.gif          # Gerado após execução
```

---

## Parâmetros configuráveis

| Parâmetro         | Descrição                                      | Valor Padrão  |
| :---------------- | :--------------------------------------------- | :------------ |
| `NUM_EPISODIOS`   | Número de episódios de treinamento             | `500`         |
| `EPS_INICIAL`     | Taxa de exploração inicial                     | `1.0`         |
| `EPS_MINIMO`      | Taxa mínima de exploração                      | `0.01`        |
| `EPS_DECAY`       | Fator de decaimento do epsilon                 | `0.995`       |
| `ALPHA`           | Taxa de aprendizado (learning rate)           | `0.1`         |
| `GAMMA`           | Fator de desconto (discount factor)           | `0.99`        |

---

## Exemplos de uso

```powershell
# Execução padrão
python q_learning_cliffwalking.py

# Com ambiente virtual (recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python q_learning_cliffwalking.py
```

---

## Ambiente CliffWalking

O ambiente consiste em uma **grade 4x12** onde:

* **S**: Posição inicial (canto inferior esquerdo)
* **G**: Objetivo (canto inferior direito)  
* **C**: Precipício (células perigosas que causam reinício)
* **·**: Células seguras para navegação

O agente deve encontrar o caminho ótimo de S para G evitando cair no precipício.

---

## Pipeline de execução

### 1. Inicialização

* Cria o ambiente CliffWalking-v0 (compatível com Gym/Gymnasium)
* Inicializa Q-table com zeros para todos os estados e ações
* Define parâmetros de aprendizado e exploração

### 2. Treinamento (Q-Learning)

* Loop de 500 episódios de treinamento
* Política epsilon-greedy para balancear exploração/exploração
* Atualização da Q-table usando a equação de Bellman
* Decaimento gradual do epsilon ao longo do tempo

### 3. Visualização da Q-table

* Exibe valores Q formatados para cada estado
* Mostra a evolução do aprendizado
* Identifica estados críticos e valores convergidos

### 4. Política aprendida

* Gera visualização com setas direcionais
* Cada célula mostra a ação ótima (↑↓←→)
* Destaca o caminho ótimo da política final

### 5. Geração do GIF

* Simula episódio usando política aprendida
* Cria frames animados com símbolos universais
* Aplica efeitos visuais (cores, animações, legendas)
* Salva como `trajeto_agente.gif`

### 6. Avaliação final

* Testa o agente em 100 episódios sem exploração
* Calcula taxa de sucesso e estatísticas de desempenho
* Gera relatório completo dos resultados

---

## Visualizações geradas

| Saída                      | Descrição                                    | Interpretação                           |
| :------------------------- | :------------------------------------------- | :-------------------------------------- |
| **Q-table formatada**     | Valores Q para cada estado e ação           | Qualidade das decisões aprendidas       |
| **Política com setas**    | Direções ótimas visualizadas                | Caminho ideal descoberto pelo agente    |
| **GIF animado**           | Trajetória do agente treinado                | Demonstração visual do comportamento    |
| **Relatório de avaliação**| Taxa de sucesso e estatísticas              | Métricas de desempenho final           |

---

## Funções principais

| Função                            | Responsabilidade                                     |
| :-------------------------------- | :--------------------------------------------------- |
| `criar_ambiente()`               | Inicialização do ambiente CliffWalking              |
| `escolher_acao_epsilon_greedy()` | Implementação da política de exploração             |
| `treinar_q_learning()`           | Loop principal de treinamento do algoritmo          |
| `imprimir_q_table()`             | Formatação e exibição da Q-table aprendida          |
| `avaliar_politica()`             | Teste de desempenho sem exploração                  |
| `mostrar_politica()`             | Visualização da política como setas direcionais     |
| `salvar_gif_trajeto()`           | Geração do GIF animado da trajetória                |

---

## Observações técnicas

* **Epsilon-greedy:** Balanceamento automático entre exploração e exploração
* **Q-table:** Representação tabular de valores estado-ação (48 estados × 4 ações)
* **Compatibilidade:** Funciona com Gym v0.21+ e Gymnasium v0.26+
* **GIF animado:** Símbolos universais (●◆▲★) para compatibilidade cross-platform
* **Matplotlib:** Visualizações robustas com fallback para diferentes versões da API

---

## Autoria e créditos

* **Autora:** Laura Barbosa Henrique (`@tinywin`)
* **Instituição:** Universidade Federal do Tocantins (UFT)
* **Disciplina:** Inteligência Artificial — 2025/02
* **Docente:** Prof. Dr. Alexandre Rossini
* **Contato:** `laura.henrique@mail.uft.edu.br`

**Ambiente:**
CliffWalking-v1 — [Gymnasium](https://gymnasium.farama.org/) / [OpenAI Gym](https://github.com/openai/gym)

---

## Configuração do ambiente

### Windows / PowerShell

1. Criar e ativar ambiente virtual (opcional, mas recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependências:

```powershell
pip install -r requirements.txt
```

3. Executar o programa:

```powershell
python q_learning_cliffwalking.py
```

---

## Licença e uso

Projeto **educacional**, sem fins comerciais.
Código e experimentos liberados para **aprendizado e pesquisa**.

---

## Resumo simples

> "Implementei o algoritmo Q-Learning para treinar um agente no ambiente CliffWalking.
> O sistema aprende automaticamente a política ótima usando exploração epsilon-greedy e gera visualizações animadas da trajetória.
> Os resultados mostram convergência para 100% de taxa de sucesso após 500 episódios de treinamento."

---

## Conclusão

O modelo Q-Learning alcançou **convergência completa** no ambiente CliffWalking.
A análise revelou:

* **Política ótima:** Caminho mais seguro evitando o precipício
* **Taxa de sucesso:** 100% após treinamento completo  
* **Convergência:** Q-table estabilizada com valores consistentes
* **Visualização:** GIF animado demonstra comportamento aprendido

O epsilon-greedy permitiu balancear exploração e exploração eficientemente, garantindo descoberta da política ótima com alta taxa de sucesso.
