# Deep Q-Learning en Demon Attack (RAM)

Agente DQN en PyTorch que aprende a jugar a **Demon Attack** de Atari (`ALE/DemonAttack-v5`) usando como
estado la **RAM de la consola (128 bytes)** en lugar de los píxeles de la pantalla.

El código adapta la guía *"DQN PyTorch Beginners Tutorial: Flappy Bird"* a un juego de Atari y añade
varios experimentos para estabilizar el aprendizaje y una forma más fiable de elegir el mejor modelo.

## Montaje

| Elemento | Valor |
|---|---|
| Estado | RAM de 128 bytes (uint8, normalizada a [0, 1]); con `frame_stack: 4`, 4 estados apilados = 512 entradas |
| Red | MLP 128 → 256 → 256 → 6 (ReLU), un valor Q por acción (512 → 256 → 256 → 6 con 4 estados) |
| Replay memory | 100.000 transiciones; se empieza a entrenar con 10.000 |
| Optimización | lotes de 32, una actualización cada 4 pasos |
| Descuento | γ = 0,99 |
| Optimizador | Adam, lr = 1e-4 |
| Exploración | ε-greedy, ε de 1 a 0,05 multiplicando por 0,995 en cada episodio (llega al mínimo hacia el episodio 600) |
| Recompensa | recortada a ±1 (`sign(r)`) |
| Red objetivo | copia de la red de política cada 1.000 pasos (10.000 en los experimentos `stable`) |
| Pérdida | MSE (Huber en los experimentos `stable`) |

El objetivo de aprendizaje es el habitual de DQN: `y = r + γ · (1 − terminal) · max_a' Q_objetivo(s', a')`.

Entrenado en un MacBook Air M2, solo CPU, a unos 2.500 pasos por segundo.

## Experimentos

Media de la puntuación por episodio (sin recortar) durante el entrenamiento, ± error estándar.
Una semilla por experimento.

| Experimento | Cambio | Media ep. 500–1000 | Media ep. 1000–1500 |
|---|---|---|---|
| `long` (DQN base) | — | 269 ± 9 | 301 ± 11 |
| `ddqn` | Double DQN | 267 ± 9 | 269 ± 11 |
| `stack4` | 4 estados RAM apilados | 275 ± 9 | 307 ± 12 |
| `stable` | perder vida = terminal, Huber, red objetivo cada 10.000 pasos | 312 ± 12 | 341 ± 13 |
| `stable_stack4` | `stable` + 4 estados (solo 1.000 episodios) | 358 ± 14 | — |

Cada experimento corresponde a un bloque de `hyperparameters.yml` con el prefijo `demonattack_`
(p. ej. `demonattack_stable`). `long` se configuró para 5.000 episodios y se paró a mano en el 3.935.

Al reevaluar los snapshots con ε = 0,05 (20 partidas cada uno), `stable_stack4` promedia ~398 puntos en los
episodios 500–1000, frente a ~266 de `stable`. El mejor snapshot de `stable_stack4` (episodio 900) hace
606 ± 151 y jugó una partida de 3.150 puntos.

> El ~266 de `stable` (media de 377,5 / 192,2 / 330,0 / 291,8 / 183,0 / 224,0 en los episodios 500–1000)
> sale de la salida de terminal de esa evaluación, no de un CSV. El CSV se sobrescribió después con una
> evaluación de 10 partidas que tenía el mismo nombre de archivo. Por eso `eval_snapshots.py` incluye ahora el
> número de partidas en el nombre de sus resultados.

## Mejor agente

El mejor agente es el **snapshot del episodio 400 del experimento `stack4`**:
**1.218 ± 80 puntos** (media ± error estándar) en 30 partidas con ε = 0.

Como referencia, Mnih et al. (2015) dan para Demon Attack 152 puntos a un jugador aleatorio y 3.401 a un
humano. Así que el agente está muy por encima del azar y lejos del nivel humano.

El modelo está en [`models/demonattack_stack4_best_confirmed.pt`](models/demonattack_stack4_best_confirmed.pt)
(< 1 MB).

## Hallazgo principal

**La red encuentra buenas políticas pronto, pero no las conserva.** El agente que mejor juega sale en el
episodio 400 de `stack4`, mucho antes del final del entrenamiento. Después, el rendimiento sube y baja sin
quedarse en el máximo.

Por eso el mejor modelo no es el último. Se guarda un snapshot cada 50 episodios (configurable con
`snapshot_every`) y se reevalúan después con
muchas partidas, sin entrenar (`eval_snapshots.py`). Las evaluaciones de 10 partidas durante el entrenamiento
sirven para preseleccionar candidatos, y la evaluación posterior con 30 partidas y las mismas semillas para
todos confirma cuál es mejor de verdad. En `stack4`, el episodio 200 también parecía prometedor
(923 ± 131 en 30 partidas), pero el 400 es más alto y bastante más regular.

## Por qué "perder vida = terminal"

En los experimentos `stable`, perder una vida se trata como estado terminal **solo para el objetivo de
aprendizaje**. La partida sigue, pero esa transición no suma el valor del estado siguiente.

Así, morir cuesta exactamente el valor futuro que se pierde:

- no se premia sobrevivir (una recompensa por seguir vivo lleva al agente a esconderse en lugar de disparar);
- no se castiga de más (una penalización fija se suma a lo que ya se pierde y distorsiona la escala de Q).

`agent.py` también admite `life_loss: terminal_penalty` (terminal y además −1), aunque no se usa en ningún
experimento de la tabla.

## Cómo reproducir

```bash
conda create -n dql python=3.11
conda activate dql
pip install "gymnasium[atari]" torch pyyaml matplotlib "imageio[ffmpeg]"
```

Todos los comandos se ejecutan desde la carpeta del proyecto. Los resultados (logs, gráficas, checkpoints,
snapshots y vídeos) se guardan en `runs/`, que no está en el repositorio.

```bash
# Entrenar un experimento (nombre = bloque de hyperparameters.yml)
python agent.py demonattack_stable_stack4 --train

# Ver jugar al mejor agente (abre una ventana y juega con ε = 0 hasta que lo pares con Ctrl+C)
python agent.py demonattack_stack4 --model models/demonattack_stack4_best_confirmed.pt

# Reevaluar los snapshots de un entrenamiento (aquí cada 100 episodios, 20 partidas, ε = 0,05)
python eval_snapshots.py demonattack_stable --every 100 --games 20 --epsilon 0.05

# Vídeo de progreso: una partida por snapshot (episodios 200, 400, …), máximo 1.500 pasos cada una
python make_video.py demonattack_stack4 --every 200 --max-steps 1500

# Entrenamientos largos en paralelo dentro de tmux (ver la sección siguiente)
bash run_long.sh
```

### Entrenamientos largos en paralelo (`run_long.sh`)

`run_long.sh` lanza cuatro entrenamientos largos a la vez en una sesión de tmux llamada `dqn`, una ventana
por entrenamiento. Siguen corriendo aunque se cierre la conexión remota.

| Bloque | Qué es |
|---|---|
| `ss4_long_s1`, `ss4_long_s2`, `ss4_long_s3` | `stable` + 4 estados apilados, con semillas 1, 2 y 3 |
| `ss4_long_lr5e5` | igual, pero con la tasa de aprendizaje a la mitad (5e-5) y semilla 1, para compararlo con `s1` |

Diferencias con los experimentos de la tabla: 10.000 episodios, replay memory de 500.000 transiciones
(se empieza a entrenar con 50.000), y evaluación y snapshot cada 100 episodios.

Está pensado para una máquina Ubuntu con muchos núcleos y mucha RAM (unos 1 GB por entrenamiento). Fija
`OMP_NUM_THREADS=1` y `MKL_NUM_THREADS=1` y usa `torch_threads: 1`, para que los procesos no compitan
por los núcleos. Necesita `tmux` y un entorno conda llamado `dql`.

```bash
bash run_long.sh
tmux attach -t dqn                  # ver los entrenamientos (Ctrl+b n: siguiente ventana; Ctrl+b d: salir sin pararlos)
tail -n 2 runs/ss4_long_*.log       # progreso rápido
```

### Claves de configuración opcionales

Además de los hiperparámetros de la tabla de montaje, cada bloque de `hyperparameters.yml` admite:

| Clave | Por defecto | Qué hace |
|---|---|---|
| `seed` | ninguna (aleatoria) | Fija las semillas de `random`, NumPy, PyTorch y el entorno, para poder repetir un entrenamiento y comparar semillas. |
| `snapshot_every` | 50 | Cada cuántos episodios se guarda un snapshot en `runs/<set>_snapshots/`. |
| `torch_threads` | lo que decida PyTorch | Número de hilos de PyTorch (`torch.set_num_threads`). Con 1, varios entrenamientos en paralelo no se pisan. |
| `frame_stack`, `double_dqn`, `life_loss`, `loss`, `eval_every`, `eval_episodes`, `resume_from` | — | Las opciones de los experimentos: apilado de estados, Double DQN, pérdida de vida, pérdida, evaluación periódica y reanudar desde un checkpoint. |

## Archivos

| Archivo | Qué hace |
|---|---|
| `agent.py` | Bucle de entrenamiento y de prueba: entorno (con apilado opcional de estados RAM), ε-greedy, replay memory, actualización DQN / Double DQN, red objetivo, pérdida de vida como terminal, evaluación periódica con ε = 0, snapshots cada 50 episodios, checkpoints para reanudar y gráfica de progreso. |
| `dqn.py` | La red: MLP de dos capas ocultas que pasa de la RAM normalizada a un valor Q por acción. |
| `experience_replay.py` | Replay memory: cola de tamaño fijo con muestreo aleatorio de transiciones. |
| `hyperparameters.yml` | Un bloque por experimento (`demonattack_debug`, `_long`, `_ddqn`, `_stack4`, `_stable`, `_stable_stack4`, y `_long2` / `_stable2` para continuar un entrenamiento desde su checkpoint), más los cuatro bloques `ss4_long_*` de los entrenamientos largos. |
| `eval_snapshots.py` | Reevalúa los snapshots con muchas partidas y semillas fijas. Guarda una tabla CSV y copia el mejor snapshot; ε y el número de partidas van en el nombre (p. ej. `runs/<set>_snapshot_eval_eps0.05_g20.csv` y `runs/<set>_best_confirmed_g30.pt`). |
| `make_video.py` | Genera un MP4 con una partida por snapshot, con el episodio y la puntuación encima. |
| `run_long.sh` | Lanza en tmux los cuatro entrenamientos largos `ss4_long_*` en paralelo. |
| `test_env.py` | Comprobación rápida del entorno: una partida con acciones aleatorias. |
| `models/` | El mejor agente entrenado. |

## Limitaciones

- **Una semilla por experimento.** Las diferencias pequeñas de la tabla (p. ej. `long` frente a `stack4`)
  pueden deberse al azar.
- **Evaluaciones cortas:** 10 partidas durante el entrenamiento y 30 en la evaluación final. Demon Attack tiene
  mucha varianza entre partidas.
- **Entrenamientos cortos:** entre 1,7 y 6,7 millones de pasos, frente a los 50 millones del paper de DQN.
- **No es comparable punto por punto con el paper:** aquí la entrada es la RAM y no los píxeles, y `v5` usa
  la configuración por defecto de ALE en Gymnasium, distinta de la del paper.

## Fuentes

- Guía *"DQN PyTorch Beginners Tutorial: Flappy Bird"*, que sirve de base para la estructura del código
  (agente, red, replay memory e hiperparámetros en YAML).
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
