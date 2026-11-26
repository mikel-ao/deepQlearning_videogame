#!/usr/bin/env bash
# Lanza los entrenamientos largos en paralelo dentro de tmux (sobreviven a cerrar AnyDesk).
# Uso:  bash run_long.sh            -> crea la sesión tmux "dqn" con una ventana por entrenamiento
#       tmux attach -t dqn          -> ver cómo van (Ctrl+b, n = siguiente ventana; Ctrl+b, d = salir sin pararlos)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUNS="ss4_long_s1 ss4_long_s2 ss4_long_s3 ss4_long_lr5e5"
tmux new-session -d -s dqn -n info "echo 'Entrenamientos: $RUNS'; bash"
for r in $RUNS; do
  tmux new-window -t dqn -n "$r" "conda run --no-capture-output -n dql python agent.py $r --train; bash"
done
echo "Lanzados en tmux: $RUNS"
echo "Ver: tmux attach -t dqn   |   Progreso rápido: tail -n 2 runs/ss4_long_*.log"
