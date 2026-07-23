# Korpus Malti v4.0 Integration & Setup Guide

This document describes how to download, preprocess, verify, and containerize **Korpus Malti v4.0** for candidate evidence scoring in the Maltese Spellchecker.

---

## 1. Prerequisites & Hugging Face Authentication

Accessing `MLRS/korpus_malti` on Hugging Face requires:
1. Accepting the dataset access terms at: [https://huggingface.co/datasets/MLRS/korpus_malti](https://huggingface.co/datasets/MLRS/korpus_malti)
2. Creating a User Access Token on Hugging Face.

### Setting your HF Token

#### Windows (PowerShell)
```powershell
$env:HF_TOKEN="hf_your_access_token_here"
```

#### Linux / macOS (Bash)
```bash
export HF_TOKEN="hf_your_access_token_here"
```

#### Alternative: Hugging Face CLI
```bash
huggingface-cli login
```

---

## 2. Offline Setup & Preprocessing Command

Run the setup script to download and convert Korpus Malti into compact runtime indexes.

### Windows (PowerShell)
```powershell
py tools/setup_korpus_malti.py
```

### Linux / macOS
```bash
python3 tools/setup_korpus_malti.py
```

### Passing token via CLI option
```bash
python3 tools/setup_korpus_malti.py --token hf_your_access_token_here
```

### Force rebuild existing indexes
```bash
python3 tools/setup_korpus_malti.py --force
```

---

## 3. Verification Command

To verify that the corpus index files are ready and valid:

```powershell
py tools/setup_korpus_malti.py
```
If the indexes are already present and valid, output will confirm:
```
INFO [setup_korpus_malti] Found existing valid Korpus Malti index:
INFO [setup_korpus_malti]   Version: 1.0.0
INFO [setup_korpus_malti]   Total Tokens: ...
```

Or test using pytest:
```bash
python -m pytest Other/tests/test_corpus_scorer.py -v
```

---

## 4. Docker Build Instructions

To include the preprocessed Korpus Malti runtime index inside a Docker image:

1. Run the offline setup locally **before** building Docker:
   ```bash
   python3 tools/setup_korpus_malti.py
   ```
2. Include `Essentials/corpus` in your `Dockerfile` COPY directive:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 5000
   ENV SPELLCHECK_CORPUS_SCORING=true
   CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:flask_app"]
   ```

> **Note:** The application will **never** download or preprocess the corpus during image launch or a live Cloud Run request. If index files are missing, it logs a warning and cleanly disables corpus evidence.

---

## 5. Attribution & Licence Notice

```
Korpus Malti v4.0 is provided by MLRS (Maltese Language Resource Server), Institute of Linguistics & Language Technology, University of Malta.
```
Dataset terms and conditions apply: non-commercial / research use as specified by MLRS.
