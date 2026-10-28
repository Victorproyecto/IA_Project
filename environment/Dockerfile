# Una imagen con tres puntos de entrada: pipeline, train y app (Fase 1, 5.5)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data

# rasterio necesita libexpat en la imagen slim
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

COPY environment/requirements.txt environment/requirements.txt
RUN pip install --no-cache-dir -r environment/requirements.txt

COPY pytest.ini .
COPY src/ src/
COPY tests/ tests/
COPY environment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && useradd --create-home dana && mkdir -p /app/data && chown -R dana /app
USER dana

VOLUME /app/data
EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
CMD ["app"]
