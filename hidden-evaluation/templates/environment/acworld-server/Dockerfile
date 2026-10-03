FROM python:3.11.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/private \
    ACWORLD_CASE_PATH=/case.json \
    ACWORLD_EVIDENCE_DIR=/acworld-evidence

COPY adapter/requirements.txt /adapter/requirements.txt
RUN pip install --no-cache-dir -r /adapter/requirements.txt
COPY private/ /private/
COPY adapter/ /adapter/

# bootstrap verifies the unchanged, pinned source archive before importing it.
RUN python -c "import bootstrap" \
    && mkdir -p /acworld-evidence

COPY case.json /case.json

WORKDIR /adapter
EXPOSE 8765
CMD ["python", "/adapter/server.py"]
