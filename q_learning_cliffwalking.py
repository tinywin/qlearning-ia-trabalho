# -*- coding: utf-8 -*-
"""
Projeto: Q-Learning no ambiente CliffWalking-v0/v1 (Gym/Gymnasium)

Objetivo do projeto
- Implementar, do zero, o algoritmo de Aprendizado por Reforço Q-Learning
  aplicado ao ambiente CliffWalking.

Descrição do ambiente CliffWalking
- O CliffWalking é um grid 4x12 (4 linhas, 12 colunas). O agente começa no
  estado inicial S (canto inferior esquerdo) e deve chegar ao objetivo G
  (canto inferior direito).
- A borda inferior, entre S e G, contém um “penhasco” (cliff). Se o agente
  entrar no cliff, recebe uma grande penalidade e o episódio termina.

Recompensas
- Cada passo: recompensa de -1.
- Cair do cliff: recompensa de -100 e término do episódio.
- Alcançar o objetivo (G): recompensa 0 e término do episódio.

Estados e ações
- Estados: cada célula do grid corresponde a um estado discreto, totalizando
  4 * 12 = 48 estados. Geralmente representados como inteiros [0..47].
- Ações: 4 ações possíveis por estado (quando válidas):
  0 = CIMA, 1 = DIREITA, 2 = BAIXO, 3 = ESQUERDA.

Resumo do Q-Learning
- Mantemos uma tabela Q(s, a) com os valores esperados de recompensa futura ao
  tomar ação a no estado s e seguir a política atual.
- Atualização Q-Learning:
    target = recompensa + gamma * max_{a'} Q(s', a')     (se não terminou)
    target = recompensa                                  (se terminou)
    Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

Escolha de ação (política epsilon-greedy)
- Com probabilidade epsilon escolhemos uma ação aleatória (exploração).
- Caso contrário, escolhemos a ação com maior Q (exploração do conhecimento).
- Epsilon decai ao longo dos episódios.

Compatibilidade Gym/Gymnasium
- Tenta importar gymnasium; se não existir, usa gym.
- Normaliza as saídas de reset/step para uma interface única.

Como executar
- Após instalar as dependências (ver README), execute:
    python q_learning_cliffwalking.py
"""

from __future__ import annotations

# Importações padrão
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

# NumPy para manipular a Q-table
import numpy as np

# Matplotlib (configura backend ANTES de importar pyplot)
import matplotlib
matplotlib.use("Agg")  # backend para gerar figuras sem janela gráfica

import matplotlib.pyplot as plt

# GIF para visualização do trajeto
import imageio.v2 as imageio  # imageio>=2

# Tratamento automático para funcionar com gymnasium OU gym
try:
    import gymnasium as gym  # type: ignore
    _USANDO_GYMN = True
except Exception:
    import gym  # type: ignore
    _USANDO_GYMN = False

# Tentar ter acesso ao wrapper TimeLimit (para garantir limite de passos)
try:
    if _USANDO_GYMN:
        from gymnasium.wrappers import TimeLimit as GymTimeLimit
    else:
        from gym.wrappers import TimeLimit as GymTimeLimit  # type: ignore
except Exception:
    GymTimeLimit = None  # type: ignore

# Mapa de nomes legíveis das ações: 0=CIMA, 1=DIREITA, 2=BAIXO, 3=ESQUERDA
NOME_ACOES_PT = {
    0: "CIMA",
    1: "DIREITA",
    2: "BAIXO",
    3: "ESQUERDA",
}

# Mapa de setas para visualização da política
SETAS = {
    0: "^",  # CIMA
    1: ">",  # DIREITA
    2: "v",  # BAIXO
    3: "<",  # ESQUERDA
}


def _reset_normalizado(env):
    """Reset compatível com gym e gymnasium.

    Retorna sempre (estado, info) conforme a API do Gymnasium.
    """
    resultado = env.reset()
    if isinstance(resultado, tuple) and len(resultado) == 2:
        obs, info = resultado
        return obs, info
    else:
        obs = resultado
        info = {}
        return obs, info


def _step_normalizado(env, acao: int):
    """Step compatível com gym e gymnasium.

    Retorna sempre (proximo_estado, recompensa, terminado, truncado, info).
    """
    resultado = env.step(acao)

    if isinstance(resultado, tuple) and len(resultado) == 5:
        # Gymnasium
        obs, reward, terminated, truncated, info = resultado
        return obs, reward, terminated, truncated, info
    elif isinstance(resultado, tuple) and len(resultado) == 4:
        # Gym antigo
        obs, reward, done, info = resultado
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info
    else:
        raise RuntimeError(
            "Formato inesperado retornado por env.step(); verifique a versão do Gym/Gymnasium."
        )


@dataclass
class HistoricoTreino:
    """Métricas do treinamento."""
    recompensas_por_episodio: List[float]
    medias_moveis_100: List[float]


@dataclass
class ResultadoAvaliacao:
    """Resultados da avaliação da política."""
    recompensas: List[float]
    taxa_sucesso: float


def criar_ambiente(seed: Optional[int] = 123) -> Tuple["gym.Env", int, int]:
    """Cria e retorna o ambiente CliffWalking-v0/v1, além de suas dimensões.

    - Tenta 'CliffWalking-v0', cai para 'CliffWalking-v1' se necessário.
    - Aplica TimeLimit se não houver um limite de passos razoável.
    """

    def _tenta_make(env_id: str):
        try:
            return gym.make(env_id, render_mode=None)
        except TypeError:
            return gym.make(env_id)

    try:
        env = _tenta_make("CliffWalking-v0")
    except Exception:
        env = _tenta_make("CliffWalking-v1")

    # Descobrir id do ambiente (v0 ou v1) para log
    env_id = getattr(getattr(env, "spec", None), "id", "desconhecido")
    print(f"Ambiente criado: {env_id}")

    # Garantir limite razoável de passos por episódio (TimeLimit)
    if GymTimeLimit is not None and not isinstance(env, GymTimeLimit):
        # por padrão, CliffWalking termina rápido, mas aqui garantimos um teto
        max_steps = 500
        env = GymTimeLimit(env, max_episode_steps=max_steps)
        print(f"TimeLimit aplicado: max_episode_steps = {max_steps}")

    # Ajuste de semente
    try:
        _ = env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass
    except TypeError:
        try:
            env.seed(seed)  # type: ignore
        except Exception:
            pass

    # Obter nrow, ncol
    nrow = getattr(getattr(env, "unwrapped", env), "nrow", None)
    ncol = getattr(getattr(env, "unwrapped", env), "ncol", None)
    if nrow is None or ncol is None:
        nrow, ncol = 4, 12  # padrão do CliffWalking

    return env, int(nrow), int(ncol)


def escolher_acao_epsilon_greedy(
    Q: np.ndarray, estado: int, epsilon: float, rng: random.Random
) -> int:
    """Seleciona uma ação usando política epsilon-greedy."""
    if rng.random() < epsilon:
        # Exploração
        n_acoes = Q.shape[1]
        return rng.randrange(n_acoes)
    else:
        # Exploração do conhecimento (exploitação)
        valores = Q[estado]
        max_q = np.max(valores)
        acoes_maximas = np.where(np.isclose(valores, max_q))[0]
        return int(rng.choice(acoes_maximas))


def treinar_q_learning(
    env: "gym.Env",
    nrow: int,
    ncol: int,
    num_episodios: int = 20000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_inicial: float = 1.0,
    epsilon_minimo: float = 0.05,
    epsilon_decay: float = 0.9995,
    log_intervalo: int = 100,
    qtable_log_intervalo: int = 5000,
    seed: Optional[int] = 123,
) -> Tuple[np.ndarray, HistoricoTreino]:
    """Treinamento Q-Learning propriamente dito."""

    n_estados = env.observation_space.n
    n_acoes = env.action_space.n

    Q = np.zeros((n_estados, n_acoes), dtype=np.float64)

    recompensas_por_episodio: List[float] = []
    medias_moveis_100: List[float] = []

    rng = random.Random(seed)
    epsilon = float(epsilon_inicial)

    for episodio in range(1, num_episodios + 1):
        estado, _info = _reset_normalizado(env)
        estado = int(estado)

        recompensa_total = 0.0

        while True:
            acao = escolher_acao_epsilon_greedy(Q, estado, epsilon, rng)

            prox_estado, recompensa, terminado, truncado, _info = _step_normalizado(env, acao)
            prox_estado = int(prox_estado)

            recompensa_total += float(recompensa)

            if terminado or truncado:
                alvo = recompensa
            else:
                alvo = recompensa + gamma * np.max(Q[prox_estado])

            Q[estado, acao] = Q[estado, acao] + alpha * (alvo - Q[estado, acao])

            estado = prox_estado

            if terminado or truncado:
                break

        recompensas_por_episodio.append(recompensa_total)

        janela = 100
        ini = max(0, len(recompensas_por_episodio) - janela)
        media_100 = float(np.mean(recompensas_por_episodio[ini:]))
        medias_moveis_100.append(media_100)

        # Decaimento de epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_minimo)

        if episodio % log_intervalo == 0:
            print(
                f"Episódios concluídos: {episodio} | "
                f"Recompensa média (últimos 100): {medias_moveis_100[-1]:.3f} | "
                f"epsilon: {epsilon:.4f}"
            )

        if episodio % qtable_log_intervalo == 0:
            print(f"\nQ-table (parcial) após {episodio} episódios:")
            imprimir_q_table(Q, nrow, ncol, max_estados=24)
            print("-")

    historico = HistoricoTreino(
        recompensas_por_episodio=recompensas_por_episodio,
        medias_moveis_100=medias_moveis_100,
    )
    return Q, historico


def _indice_para_rc(indice: int, ncol: int) -> Tuple[int, int]:
    """Converte índice de estado (0..N-1) para (linha, coluna)."""
    r = indice // ncol
    c = indice % ncol
    return int(r), int(c)


def _rc_para_indice(r: int, c: int, ncol: int) -> int:
    """Converte (linha, coluna) em índice único de estado."""
    return int(r * ncol + c)


def imprimir_q_table(Q: np.ndarray, nrow: int, ncol: int, max_estados: Optional[int] = None) -> None:
    """Imprime a Q-table com ações nomeadas em PT."""
    n_estados = Q.shape[0]
    ate = n_estados if max_estados is None else min(n_estados, max_estados)

    print("Legenda ações: 0=CIMA, 1=DIREITA, 2=BAIXO, 3=ESQUERDA")
    print("Estado (r,c): Q[CIMA] | Q[DIREITA] | Q[BAIXO] | Q[ESQUERDA]")

    for s in range(ate):
        r, c = _indice_para_rc(s, ncol)
        q_vals = Q[s]
        linha = (
            f"s={s:02d} ({r},{c}): "
            f"{q_vals[0]:.3f} | {q_vals[1]:.3f} | {q_vals[2]:.3f} | {q_vals[3]:.3f}"
        )
        print(linha)

    if ate < n_estados:
        print(f"... ({n_estados - ate} estados restantes não impressos)")


def _celula_eh_cliff(r: int, c: int, nrow: int, ncol: int) -> bool:
    """True se a célula (r,c) for parte do cliff."""
    return r == (nrow - 1) and 1 <= c <= (ncol - 2)


def _estado_objetivo(nrow: int, ncol: int) -> int:
    """Índice do estado objetivo (G)."""
    return _rc_para_indice(nrow - 1, ncol - 1, ncol)


def mostrar_politica(Q: np.ndarray, nrow: int, ncol: int) -> None:
    """Exibe a política aprendida como setas em um grid 2D."""
    estado_start = _rc_para_indice(nrow - 1, 0, ncol)
    estado_goal = _estado_objetivo(nrow, ncol)

    print("\nPolítica aprendida (setas):")
    for r in range(nrow):
        linha_str = []
        for c in range(ncol):
            s = _rc_para_indice(r, c, ncol)

            if s == estado_start:
                linha_str.append("S")
            elif s == estado_goal:
                linha_str.append("G")
            elif _celula_eh_cliff(r, c, nrow, ncol):
                linha_str.append("C")
            else:
                a_max = int(np.argmax(Q[s]))
                linha_str.append(SETAS[a_max])
        print(" ".join(linha_str))

    print("Legenda: S=Início, G=Objetivo, C=Cliff, ^=CIMA, >=DIREITA, v=BAIXO, <=ESQUERDA\n")


def extrair_trajeto_greedy(Q: np.ndarray, nrow: int, ncol: int, max_passos: int = 200) -> List[int]:
    """
    Usa a política greedy (argmax da Q-table) para gerar um trajeto
    a partir do estado inicial S até o objetivo G, ou até atingir
    o número máximo de passos (para evitar loops).

    Retorna:
    - lista de índices de estados visitados na ordem.
    """
    estado_start = _rc_para_indice(nrow - 1, 0, ncol)
    estado_goal = _estado_objetivo(nrow, ncol)

    trajeto: List[int] = [estado_start]
    estado_atual = estado_start

    for _ in range(max_passos):
        if estado_atual == estado_goal:
            break

        a = int(np.argmax(Q[estado_atual]))
        r, c = _indice_para_rc(estado_atual, ncol)

        # aplica ação no grid
        if a == 0:      # CIMA
            r -= 1
        elif a == 1:    # DIREITA
            c += 1
        elif a == 2:    # BAIXO
            r += 1
        elif a == 3:    # ESQUERDA
            c -= 1

        # mantém dentro dos limites
        r = max(0, min(nrow - 1, r))
        c = max(0, min(ncol - 1, c))

        novo_estado = _rc_para_indice(r, c, ncol)
        trajeto.append(novo_estado)
        estado_atual = novo_estado

        # se caiu no cliff, pode parar (só por segurança)
        if _celula_eh_cliff(r, c, nrow, ncol):
            break

    return trajeto


def salvar_gif_trajeto(
    trajeto: List[int],
    nrow: int,
    ncol: int,
    nome_arquivo: str = "trajeto_cliffwalking.gif",
    duracao_frame: float = 0.6,
) -> None:
    """
    Gera um GIF estilizado mostrando o agente se movendo pelo grid CliffWalking
    com base em uma lista de estados (trajeto).

    Melhorias visuais:
    - Cores mais atrativas e gradientes
    - Agente animado com diferentes símbolos
    - Rastro do caminho percorrido
    - Bordas e sombras elegantes
    - Título dinâmico com contador de passos
    """
    import matplotlib.patches as patches
    
    frames = []
    estados_visitados = []  # Para mostrar o rastro

    estado_start = _rc_para_indice(nrow - 1, 0, ncol)
    estado_goal = _estado_objetivo(nrow, ncol)
    
    # Símbolos do agente para criar animação (usando caracteres universais)
    simbolos_agente = ["●", "◆", "▲", "★"]
    cores_agente = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    tamanhos_agente = [18, 16, 18, 20]  # Variação no tamanho para animação

    for i, estado in enumerate(trajeto):
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#2C3E50')
        ax.set_facecolor('#34495E')

        # Adicionar estados visitados ao rastro
        if estado not in estados_visitados:
            estados_visitados.append(estado)

        # Desenha o grid com estilo aprimorado
        for r in range(nrow):
            for c in range(ncol):
                s = _rc_para_indice(r, c, ncol)
                
                # Definir cores e estilos
                if s == estado_start:
                    cor = "#2ECC71"  # Verde vibrante
                    cor_borda = "#27AE60"
                    texto = "S"
                    texto_cor = "white"
                    fontsize = 18
                    fontweight = "bold"
                elif s == estado_goal:
                    cor = "#3498DB"  # Azul vibrante
                    cor_borda = "#2980B9"
                    texto = "G"
                    texto_cor = "white"
                    fontsize = 18
                    fontweight = "bold"
                elif _celula_eh_cliff(r, c, nrow, ncol):
                    cor = "#E74C3C"  # Vermelho vibrante
                    cor_borda = "#C0392B"
                    texto = "X"
                    texto_cor = "white"
                    fontsize = 16
                    fontweight = "bold"
                elif s in estados_visitados and s != estado:
                    # Rastro do caminho (gradiente baseado na ordem)
                    idade_rastro = estados_visitados.index(s) / max(1, len(estados_visitados) - 1)
                    cor = "#F39C12"  # Laranja para rastro
                    cor_borda = "#E67E22"
                    texto = "·"
                    texto_cor = "#2C3E50"
                    fontsize = 12
                    fontweight = "bold"
                else:
                    cor = "#ECF0F1"  # Cinza claro
                    cor_borda = "#BDC3C7"
                    texto = ""
                    texto_cor = "#2C3E50"
                    fontsize = 10
                    fontweight = "normal"

                # Criar retângulo com sombra
                shadow = patches.Rectangle(
                    (c + 0.02, nrow - 1 - r - 0.02),
                    0.96, 0.96,
                    facecolor='#1A252F', alpha=0.3, zorder=1
                )
                ax.add_patch(shadow)
                
                rect = patches.Rectangle(
                    (c, nrow - 1 - r),
                    1, 1,
                    edgecolor=cor_borda,
                    facecolor=cor,
                    linewidth=2,
                    zorder=2
                )
                ax.add_patch(rect)

                if texto:
                    ax.text(
                        c + 0.5, nrow - 1 - r + 0.5,
                        texto,
                        ha="center", va="center",
                        fontsize=fontsize,
                        fontweight=fontweight,
                        color=texto_cor,
                        zorder=3
                    )

        # Desenha o agente com animação
        r_ag, c_ag = _indice_para_rc(estado, ncol)
        simbolo_atual = simbolos_agente[i % len(simbolos_agente)]
        cor_atual = cores_agente[i % len(cores_agente)]
        tamanho_atual = tamanhos_agente[i % len(tamanhos_agente)]
        
        # Círculo de destaque ao redor do agente (pulsante)
        raio_base = 0.35
        pulso = 0.05 * (1 + math.sin(i * 0.5))  # Efeito pulsante
        circle = patches.Circle(
            (c_ag + 0.5, nrow - 1 - r_ag + 0.5),
            raio_base + pulso, 
            facecolor=cor_atual, alpha=0.8,
            edgecolor='white', linewidth=4, zorder=4
        )
        ax.add_patch(circle)
        
        # Agente com símbolos universais
        ax.text(
            c_ag + 0.5, nrow - 1 - r_ag + 0.5,
            simbolo_atual,
            ha="center", va="center",
            fontsize=tamanho_atual, 
            fontweight="bold",
            color="white", zorder=5
        )

        # Configurações do plot
        ax.set_xlim(-0.5, ncol + 0.5)
        ax.set_ylim(-0.5, nrow + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        
        # Título dinâmico com informações
        titulo = f"★ Q-Learning CliffWalking - Passo {i+1}/{len(trajeto)} ★"
        if estado == estado_goal:
            titulo += " - ✓ OBJETIVO ALCANÇADO!"
        
        ax.set_title(titulo, fontsize=16, fontweight='bold', 
                    color='white', pad=25)

        # Adicionar legenda elegante
        legenda_texto = "S = Início  |  G = Objetivo  |  X = Penhasco  |  · = Rastro"
        ax.text(ncol/2, -0.4, legenda_texto, ha='center', va='center',
               fontsize=11, color='#BDC3C7', style='italic', fontweight='bold')

        # Remover espinhas
        for spine in ax.spines.values(): 
            spine.set_visible(False)

        # converte figura para array de imagem (compatível com versões do matplotlib)
        fig.canvas.draw()
        try:
            # Método moderno (matplotlib >= 3.3.0)
            imagem = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            imagem = imagem.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Converter RGBA para RGB removendo canal alpha
            imagem = imagem[:, :, :3]
        except AttributeError:
            try:
                # Método alternativo para versões intermediárias
                imagem = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                imagem = imagem.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Fallback usando print_to_buffer (mais compatível)
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                from PIL import Image
                pil_img = Image.open(buf)
                imagem = np.array(pil_img.convert('RGB'))
                buf.close()
        frames.append(imagem)

        plt.close(fig)

    imageio.mimsave(nome_arquivo, frames, duration=duracao_frame)
    print(f"GIF salvo em: {nome_arquivo}")


def avaliar_politica(
    env: "gym.Env", Q: np.ndarray, nrow: int, ncol: int, n_episodios_avaliacao: int = 100
) -> ResultadoAvaliacao:
    """Avalia a política greedy por alguns episódios."""
    recompensas: List[float] = []
    sucessos = 0

    estado_goal = _estado_objetivo(nrow, ncol)

    for _ in range(n_episodios_avaliacao):
        estado, _info = _reset_normalizado(env)
        estado = int(estado)

        recompensa_total = 0.0

        while True:
            a = int(np.argmax(Q[estado]))
            prox_estado, reward, terminated, truncated, _info = _step_normalizado(env, a)
            prox_estado = int(prox_estado)

            recompensa_total += float(reward)
            estado = prox_estado

            if terminated or truncated:
                if terminated and not truncated and estado == estado_goal:
                    sucessos += 1
                break

        recompensas.append(recompensa_total)

    taxa_sucesso = sucessos / float(max(1, n_episodios_avaliacao))
    return ResultadoAvaliacao(recompensas=recompensas, taxa_sucesso=taxa_sucesso)


def main() -> None:
    """Função principal: cria ambiente, treina, mostra resultados e avalia."""
    NUM_EPISODIOS = 50000          # episódios de treinamento
    ALPHA = 0.1
    GAMMA = 0.99
    EPS_INICIAL = 1.0
    EPS_MINIMO = 0.10
    EPS_DECAY = 0.9995

    LOG_INTERVALO = 500
    QTABLE_LOG_INTERVALO = 5000

    SEED = 123

    env, nrow, ncol = criar_ambiente(seed=SEED)

    print("Biblioteca de ambiente detectada:", "gymnasium" if _USANDO_GYMN else "gym")
    print(f"Dimensões do grid: {nrow} linhas x {ncol} colunas")

    Q, historico = treinar_q_learning(
        env=env,
        nrow=nrow,
        ncol=ncol,
        num_episodios=NUM_EPISODIOS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_inicial=EPS_INICIAL,
        epsilon_minimo=EPS_MINIMO,
        epsilon_decay=EPS_DECAY,
        log_intervalo=LOG_INTERVALO,
        qtable_log_intervalo=QTABLE_LOG_INTERVALO,
        seed=SEED,
    )

    print("\nQ-table final completa:")
    imprimir_q_table(Q, nrow, ncol, max_estados=None)

    mostrar_politica(Q, nrow, ncol)

    # Gera GIF do trajeto greedy aprendido
    trajeto = extrair_trajeto_greedy(Q, nrow, ncol, max_passos=200)
    salvar_gif_trajeto(trajeto, nrow, ncol, nome_arquivo="trajeto_cliffwalking.gif")

    # Avaliação da política greedy
    resultado = avaliar_politica(env, Q, nrow, ncol, n_episodios_avaliacao=200)
    media_recompensa = float(np.mean(resultado.recompensas)) if resultado.recompensas else 0.0
    desvio_recompensa = float(np.std(resultado.recompensas)) if resultado.recompensas else 0.0

    print("Resultados da avaliação final (política greedy):")
    print(f" - Episódios de avaliação: {len(resultado.recompensas)}")
    print(f" - Recompensa média: {media_recompensa:.3f} ± {desvio_recompensa:.3f}")
    print(f" - Taxa de sucesso (atingiu G): {resultado.taxa_sucesso * 100:.1f}%")

    print("\nTreinamento e avaliação concluídos.")


if __name__ == "__main__":
    main()