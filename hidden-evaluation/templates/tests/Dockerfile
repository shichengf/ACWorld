FROM python:3.11.16-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# A separate verifier image must contain its own test entrypoint.
COPY test.sh verify.py expected.json /tests/
RUN mkdir -p /logs/verifier /acworld-evidence
WORKDIR /tests
