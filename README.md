# Deep Q-Learning en Demon Attack (RAM)

Agente DQN en PyTorch que aprende a jugar a **Demon Attack** de Atari (`ALE/DemonAttack-v5`) usando como
estado la **RAM de la consola (128 bytes)** en lugar de los píxeles de la pantalla.

El código adapta la guía *"DQN PyTorch Beginners Tutorial: Flappy Bird"* a un juego de Atari y añade
varios experimentos para estabilizar el aprendizaje, una forma más fiable de elegir el mejor modelo y
herramientas para aprender de partidas humanas.

## Resumen

- **Mejor agente confirmado: 1.239 ± 143 puntos** (30 partidas, ε = 0), un ~36 % de la referencia humana
  del paper de DQN (3.401). Lo consiguen de forma repetible 4 semillas distintas (entre 788 y 1.239).
- **Hallazgo principal: la red encuentra buenas políticas, pero no las conserva.** Los picos duran menos de
  100 episodios y reaparecen de forma esporádica (episodios 700, 1.100, 2.400, 4.900…). Por eso el modelo se
  elige reevaluando snapshots, nunca el último.
- **Un humano (el autor) hace 1.961 ± 277** en las mismas condiciones. Copiar sus teclas (behavior cloning)
  no basta: el clon se queda en 731. La siguiente tanda mete esas partidas dentro del propio DQN.

## Montaje

| Elemento | Valor |
|---|---|
| Estado | RAM de 128 bytes (uint8, normalizada a [0, 1]); con `frame_stack: 4`, 4 estados apilados = 512 entradas |
| Red | MLP 128 → 256 → 256 → 6 (ReLU), un valor Q por acción (512 → 256 → 256 → 6 con 4 estados) |
| Replay memory | 100.000 transiciones, se empieza a entrenar con 10.000 (500.000 y 50.000 en los entrenamientos largos) |
| Optimización | lotes de 32, una actualización cada 4 pasos |
| Descuento | γ = 0,99 |
| Optimizador | Adam, lr = 1e-4 |
| Exploración | ε-greedy, ε de 1 a 0,05 multiplicando por 0,995 en cada episodio (llega al mínimo hacia el episodio 600) |
| Recompensa | recortada a ±1 (`sign(r)`) |
| Red objetivo | copia de la red de política cada 1.000 pasos (10.000 desde los experimentos `stable`) |
| Pérdida | MSE (Huber desde los experimentos `stable`) |

El objetivo de aprendizaje es el habitual de DQN: `y = r + γ · (1 − terminal) · max_a' Q_objetivo(s', a')`.

Hardware:

- **MacBook Air M2**, solo CPU, a unos 2.500 pasos por segundo: experimentos iniciales.
- **Ubuntu con un Xeon E5-2697A v4** (16 núcleos, 64 GB): entrenamientos largos y tandas en paralelo, hasta 15
  a la vez, a unos 170–350 pasos por segundo cada uno.

## Fase 1: experimentos en el Mac

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
episodios 500–1000, frente a ~266 de `stable`. El mejor agente de esta fase es el snapshot del episodio 400
de `stack4`: **1.218 ± 80** en 30 partidas con ε = 0
([`models/demonattack_stack4_best_confirmed.pt`](models/demonattack_stack4_best_confirmed.pt)).

> El ~266 de `stable` sale de la salida de terminal de esa evaluación, no de un CSV: el CSV se sobrescribió
> después con otra evaluación. Por eso `eval_snapshots.py` incluye ahora ε y el número de partidas en el
> nombre del archivo, y acumula resultados en lugar de sobrescribirlos.

## Fase 2: entrenamientos largos con varias semillas (Ubuntu)

Configuración `stable_stack4` (perder vida = terminal, Huber, red objetivo cada 10.000 pasos, 4 estados),
replay de 500.000 y 10.000 episodios previstos, en 6 entrenamientos paralelos (`run_long.sh`).

Mejor snapshot confirmado de cada uno (30 partidas, mismas semillas, ε = 0; media ± error estándar):

| Entrenamiento | Mejor snapshot | Media | Peor / mejor partida |
|---|---|---|---|
| `ss4_long_s3` | ep 1.100 | **1.239 ± 143** | 150 / 3.315 |
| `ss4_long_s2` | ep 4.900 | **1.199 ± 143** | 325 / 4.690 |
| `ss4_long_lr5e5` (lr 5e-5) | ep 2.400 | 1.058 ± 101 | 100 / 2.020 |
| `ss4_long_s4` | ep 700 | 1.032 ± 85 | 310 / 2.000 |
| `ss4_long_s1` | ep 1.400 | 788 ± 97 | 120 / 2.020 |
| `ss4_long_ddqn_s1` (Double DQN) | ep 700 | 737 ± 70 | 70 / 1.660 |

Conclusiones:

- **Es repetible:** las 4 semillas de la misma configuración producen modelos de 790–1.240.
- **Los picos son estrechos:** el snapshot anterior y el posterior al mejor rinden mucho menos
  (s4: 215 → **1.032** → 231; s2: 303 → **1.199** → 548). La media de entrenamiento se queda plana
  (~320–420) desde el episodio ~1.500.
- **Las evaluaciones de 10 partidas engañan** en ambos sentidos (lr5e5 ep 7.200: 1.134 en la evaluación corta,
  843 al confirmar con 30). Solo cuentan las de 30 partidas.
- **Double DQN** vuelve a quedar último: descartado, y su entrenamiento se paró en el episodio 4.426.
- A veces el agente **se esconde en una esquina**: sobrevive hasta el límite de 27.000 pasos disparando hacia
  arriba y hace unos 70 puntos.

## Fase 3: ¿qué estabiliza? (tanda 2, en curso)

10 entrenamientos de 3.000 episodios sobre la configuración anterior, cambiando una sola cosa (bloques `b2_*`,
generados con `make_batch2.py`):

| Grupo | Cambio | Semillas |
|---|---|---|
| `b2_lr5e5` | lr 5e-5 | 2, 3 (+ `ss4_long_lr5e5`) |
| `b2_lr25e6` | lr 2,5e-5 | 1, 2, 3 |
| `b2_pen` | `life_loss: terminal_penalty` (−1 al perder vida) | 1, 2, 3 |
| `b2_sync30k` | red objetivo cada 30.000 pasos | 1, 2 |

Resultado provisional: lr 2,5e-5 tiene la media de entrenamiento más alta en sus 3 semillas, pero también
colapsa y se esconde a veces. Para comparar la estabilidad se reevalúan **todos** los snapshots (cada 200
episodios), no solo el mejor.

## Fase 4: partidas humanas

`record_human.py` permite jugar con el teclado y graba lo mismo que ve el agente (RAM apilada, acción,
recompensa, vidas perdidas) y, cada 30 pasos, el estado completo del emulador.

- 12 partidas normales: **1.961 ± 277 puntos** (máximo 4.220), en las mismas condiciones que el agente
  (`v5`, sticky actions 0,25).
- 1 sesión con **vidas infinitas** (se rellena el byte 114 de la RAM, las vidas de reserva) para llegar a
  oleadas avanzadas: 18.560 puntos.
- En total, ~49.000 pasos y 1.334 estados del emulador (705 con más de 1.000 puntos).

### Behavior cloning (`train_bc.py`)

La misma red entrenada como clasificador (estado → mi acción), evaluada con 30 partidas:

| Versión | Media | Qué hace |
|---|---|---|
| Clases balanceadas | 0 | Se va a la esquina y no dispara nunca |
| Sin balancear | 184 ± 57 | Nivel aleatorio |
| Disparo siempre activo (`--fire-always`) | 731 ± 98 | Dispara quieto, casi no esquiva |
| + retraso de reacción de 3 pasos (`--lag 3`) | 463 ± 56 | Peor |

Referencia: "disparar siempre sin moverse" hace ~480. El clon copia las teclas pero no ve las consecuencias,
así que no aprende cuándo apartarse. Las partidas humanas aportan más como experiencia para el DQN.

## Fase 5: DQN con partidas humanas (tanda 3, preparada)

`make_batch3.py` genera 4 grupos de 3 semillas (3.000 episodios) sobre la mejor configuración de la tanda 2:

| Grupo | Qué añade |
|---|---|
| `b3_a3` | Solo 3 acciones (FIRE, RIGHTFIRE, LEFTFIRE): el disparo siempre está activo |
| `b3_demo` | `b3_a3` + el 25 % de cada mini-batch sale de las partidas humanas (se quedan siempre en memoria) |
| `b3_dqfd` | `b3_demo` + pérdida de margen de DQfD (Hester et al., 2018) |
| `b3_start` | `b3_demo` + el 30 % de los episodios empieza en un estado humano con ≥ 1.000 puntos |

La red parte de cero (no del clon). La comparación sin partidas humanas son las semillas 1–3 de la fase 2.

## Cómo reproducir

```bash
conda create -n dql python=3.11
conda activate dql
pip install "gymnasium[atari]" torch pyyaml matplotlib pillow "imageio[ffmpeg]" pygame
```

Todos los comandos se ejecutan desde la carpeta del proyecto. Los resultados (logs, gráficas, checkpoints,
snapshots y vídeos) se guardan en `runs/`, que no está en el repositorio.

```bash
# Entrenar un experimento (nombre = bloque de hyperparameters.yml)
python agent.py demonattack_stable_stack4 --train

# Ver jugar a un agente (ventana, ε = 0, hasta Ctrl+C)
python agent.py ss4_long_s3 --model models/ss4_long_s3_ep1100.pt

# Reevaluar snapshots (30 partidas, ε = 0); el CSV acumula lo ya evaluado
python eval_snapshots.py ss4_long_s2 --episodes 4800 4900 5000
python eval_snapshots.py b2_lr25e6_s2 --every 200

# Resumen de todos los entrenamientos en curso
python resumen.py                    # runs/ss4_long_*.log
python resumen.py "runs/b2_*.log"

# Vídeos: progreso (una partida por snapshot) o una partida completa elegida entre varias
python make_video.py demonattack_stack4 --every 200 --max-steps 1500
python best_game_video.py ss4_long_s3 models/ss4_long_s3_ep1100.pt --fps 30

# Partidas humanas y clon
python record_human.py                     # partidas normales
python record_human.py --infinite-lives    # para llegar a oleadas avanzadas
python train_bc.py --fire-always --out models/bc_fire.pt
python play_clone.py models/bc_fire.pt --sample --games 30
```

### Entrenamientos en paralelo

`run_long.sh` lanza en una sesión de tmux llamada `dqn` una ventana por entrenamiento. Siguen corriendo aunque
se cierre la conexión remota. Fija `OMP_NUM_THREADS=1` y `MKL_NUM_THREADS=1` y usa `torch_threads: 1`, para que
los procesos no compitan por los núcleos; cada uno ocupa ~1,5 GB de RAM con una replay de 500.000.

```bash
bash run_long.sh
tmux attach -t dqn                  # Ctrl+b n: siguiente ventana; Ctrl+b d: salir sin pararlos
python resumen.py                   # una fila por entrenamiento
```

Las tandas 2 y 3 se generan con `make_batch2.py` / `make_batch3.py` (ejecutar una sola vez: añaden bloques a
`hyperparameters.yml`) y se lanzan con un bucle de `tmux new-window` igual que el de `run_long.sh`.

### Claves de configuración opcionales

| Clave | Por defecto | Qué hace |
|---|---|---|
| `seed` | ninguna (aleatoria) | Fija las semillas de `random`, NumPy, PyTorch y el entorno. |
| `snapshot_every` | 50 | Cada cuántos episodios se guarda un snapshot en `runs/<set>_snapshots/`. |
| `torch_threads` | lo que decida PyTorch | Hilos de PyTorch. Con 1, varios entrenamientos en paralelo no se pisan. |
| `frame_stack`, `double_dqn`, `life_loss`, `loss`, `eval_every`, `eval_episodes`, `resume_from` | — | Apilado de estados, Double DQN, pérdida de vida, pérdida, evaluación periódica y reanudar desde un checkpoint. |
| `action_set` | todas (6) | Acciones permitidas, p. ej. `[1, 4, 5]` = FIRE, RIGHTFIRE, LEFTFIRE. |
| `demo_glob`, `demo_ratio` | —, 0,25 | Partidas humanas (`demos/*.npz`) y fracción de cada mini-batch que sale de ellas. |
| `demo_margin` | 0 | Peso de la pérdida de margen de DQfD (0 = sin imitación). |
| `demo_start_prob`, `demo_start_min_score` | 0, 0 | Probabilidad de empezar un episodio en un estado humano grabado, y puntos mínimos de ese estado. |
| `init_model` | — | `.pt` con el que inicializar la red. |

## Archivos

| Archivo | Qué hace |
|---|---|
| `agent.py` | Entrenamiento y prueba: entorno con apilado opcional, ε-greedy, replay, DQN / Double DQN, red objetivo, pérdida de vida como terminal, evaluación periódica, snapshots, checkpoints, gráfica; y las opciones de la fase 5 (3 acciones, partidas humanas, margen DQfD, arranque desde estados del emulador). |
| `dqn.py` | La red: MLP de dos capas ocultas de la RAM normalizada a un valor Q por acción. |
| `experience_replay.py` | Replay memory: cola de tamaño fijo con muestreo aleatorio. |
| `hyperparameters.yml` | Un bloque por experimento: `demonattack_*` (fase 1), `ss4_long_*` (fase 2), `b2_*` (tanda 2), y los `b3_*` cuando se genere la tanda 3. |
| `eval_snapshots.py` | Reevalúa snapshots con muchas partidas y semillas fijas; el CSV acumula resultados y copia el mejor como `runs/<set>_best_confirmed_g30.pt`. |
| `resumen.py` | Tabla con el estado de cada entrenamiento (episodio, media, mejor evaluación, velocidad). |
| `run_long.sh` | Lanza en tmux los entrenamientos largos de la fase 2. |
| `make_batch2.py`, `make_batch3.py` | Generan los bloques de las tandas 2 y 3. |
| `make_video.py` | MP4 con una partida por snapshot. |
| `best_game_video.py` | MP4 de una partida completa, elegida entre varias semillas descartando las que se esconden. |
| `record_human.py` | Jugar con el teclado y grabar demostraciones (RAM, acciones, estados del emulador; opción de vidas infinitas). |
| `train_bc.py`, `play_clone.py` | Behavior cloning sobre las partidas humanas y evaluación del clon. |
| `test_env.py` | Comprobación rápida del entorno con acciones aleatorias. |
| `demos/` | Partidas humanas grabadas (~49.000 pasos). |
| `models/` | Los mejores agentes confirmados de cada fase. |
| `results/` | CSV de las reevaluaciones de snapshots de la fase 2. |

## Limitaciones

- **Pocas semillas:** 4 en la configuración principal y 1–3 en las variantes; las diferencias de menos de
  ~150 puntos entre modelos están dentro del error.
- **Evaluación:** 30 partidas por snapshot; Demon Attack tiene mucha varianza entre partidas. La media humana
  sale de 12 partidas de una sola persona.
- **Entrenamientos cortos frente al paper:** unos 10–12 millones de pasos en los largos (hacia el
  episodio 6.000), frente a 50 millones en el paper de DQN.
- **No es comparable punto por punto con el paper:** aquí la entrada es la RAM y no los píxeles, y `v5` usa la
  configuración por defecto de ALE en Gymnasium (sticky actions), distinta de la del paper.

## Fuentes

- Guía *"DQN PyTorch Beginners Tutorial: Flappy Bird"*, base de la estructura del código.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533.
- Hester, T. et al. (2018). *Deep Q-learning from Demonstrations*. AAAI.
