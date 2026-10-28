#!/bin/sh
set -e

case "$1" in
  pipeline) exec python -m src.pipeline.run ;;
  train)    exec python -m src.models.train ;;
  app)      exec streamlit run src/app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true ;;
  test)     exec python -m pytest -q ;;
  *)        echo "Uso: pipeline | train | app | test" >&2; exit 1 ;;
esac
