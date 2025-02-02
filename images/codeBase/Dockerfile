# Basis-Image mit Python
FROM python:3.9-slim

# Arbeitsverzeichnis erstellen
WORKDIR /tmp/codeBase

# Notwendige Dateien ins Image kopieren
COPY activation_script_ai.py .
COPY Readme.md .
COPY activation_script_ols.py .

# Python-Abhängigkeiten direkt im Dockerfile installieren
RUN pip install --no-cache-dir tensorflow pandas statsmodels

# Standard-Befehl: Python-Skript ausführen
CMD ["sh"]