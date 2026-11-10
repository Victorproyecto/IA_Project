#!/bin/sh
set -e

case "$1" in
  ingest)   exec python -m src.pipeline.ingest ;;
  pipeline) exec python -m src.pipeline.run ;;
  train)    exec python -m src.pipeline.train ;;
  app)      exec streamlit run src/frontend/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true ;;
  test)     exec python -m pytest -q ;;
  *)        echo "Uso: ingest | pipeline | train | app | test" >&2; exit 1 ;;
esac
