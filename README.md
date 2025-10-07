# llm-symptom-classifier
LLM symptom classifier API built with FastAPI and Redis.

> **Note:** This project is a demonstration of the core methodology used in a professional setting. **Due to confidentiality agreements, the original code and data cannot be shared.** Therefore, this repository uses mock data and a re-implementation of the general concepts to showcase the technical approach.

---

## Table of Contents
1.  [Project Background & Motivation](#1-project-background--motivation)
2.  [Getting Started: Installation & Usage](#2-getting-started-installation--usage)
3.  [API Architecture: FastAPI & Redis](#3-api-architecture-fastapi--redis)
4.  [Model: Fine-tuning & Evaluation](#4-model-fine-tuning--evaluation)
5.  [Future Improvements & Production Considerations](#5-future-improvements--production-considerations)

---

## 1. Project Background & Motivation

**Problem** : For rare disease patients, accurately categorizing patient symptoms is a challenge as they have often mixed or vague set of symptoms. Symptoms are often represented as HPO (Human Phenotype Ontology) terms which has a hierarchical tree structure. Traditional methods like scoring symptoms by their onotolgy depth struggle to capture the contextutal nuances between symptomms.

**Solution & Impact** : This projects demonstrates an approach using a fine-tuned language model to capture semantic relationships and the subtle nuances between symptoms. This approach achieved production-level accuracy, allowing downstream application like improved cohort analysis for clinical research or collecting marketing metrics.

---

## 2. Getting Started: Installation & Usage
### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for Redis)
- `pip`

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/sukwondkim/llm-symptom-classifier
    cd llm-symptom-classifier
    ```
2.  Download the newest HPO file(hp.obo):
    ```bash
    curl -L https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo > data/hp.obo
    ```
3.  Set up the Python environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```


### Fine-tuning model
- You can fine-tune the model using your dataset and configuration.
- After defining the training parameters in the script, run:
    ```bash
    bash models/run_training.sh
    ```
The fine-tuned model and training logs will be saved in the `outputs/` directory.

### Running the API
#### Option A: Run with docker compose
- Setup the Redis and api image together with docker-compose:
    ```bash
    docker-compose up --build -d
    ```
- API service will be available at: http://localhost:8000
- Redis service will be running on port 6379.
- To stop services:
    ```bash
    docker-compose down
    ```
#### Option B: Run locally
- If you have a running Redis instance(port 6379), you can start the FastAPI app directly:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
#### API Docs
- Interactive API docs (Swagger UI): http://localhost:8000/docs

---

## 3. API Architecture: FastAPI & Redis
The LLM Symptom Classifier API provides endpoints to predict, retrieve, store, and delete symptom-category mappings. The system first checks the Redis cache — if the record exists, it returns the cached result; otherwise, it triggers inference using the fine-tuned model.
### Diagram
<img width="2148" height="928" alt="image" src="https://github.com/user-attachments/assets/82c1d121-a5f8-456e-8949-1c6bb452f63b" />

### Endpoints
1. Health check
- Health check the api:
    ```bash
    curl -X get http://localhost:8000/health
    ```
  Expected response:
    ```json
    {"status": "Healthy"}
    ```
2. Post
- Get a prediction of symptom category from hpo input(use cache data if available)
    ```bash
    curl -X POST http://localhost:8000/post \
    -H "Content-Type: application/json" \
    -d '{
      "hpo_ids": ["HP:0000641", "HP:0001250"]
    }'
    ```
  Expected response:
    ```json
    {
      "prediction": "Neurodevelopmental disorder",
      "cached": false,
      "model_version": "v1.0"
    }
    ```
    Or:
    ```json
    {
      "prediction": "Neurodevelopmental disorder",
      "cached": true,
    }
    ```
3. Get
- Retrieve cached prediction
    ```bash
    curl -X POST http://localhost:8000/get \
    -H "Content-Type: application/json" \
    -d '{
      "hpo_ids": ["HP:0000641", "HP:0001250"]
    }'
    ```
    Expected response:
    ```json
    {
      "prediction": "Neurodevelopmental disorder",
      "cached": true,
    }
    ```
4. Put
- Manually store a new mapping of symptom - category
    ```bash
    curl -X POST http://localhost:8000/put \
    -H "Content-Type: application/json" \
    -d '{
      "hpo_ids": ["HP:0000641", "HP:0001250"],
      "category": "Neurodevelopmental disorder",
    }'
    ```
    Expected response:
    ```json
    {
      "status": "Created"
    }
    ```
    Or:
    ```json
    {
      "status": "Updated"
    }
    ```
5. Delete
- Delete a cached record
    ```bash
    curl -X POST http://localhost:8000/delete \
    -H "Content-Type: application/json" \
    -d '{
      "hpo_ids": ["HP:0000641", "HP:0001250"],
    }'
    ```
    Expected response:
    ```json
    {
      "status": "Deleted"
    }
    ```
---

## 4. Model: Fine-tuning & Evaluation
### Data
A synthetic data `data/train.tsv` mimics the structure of real-world clinical data (HPO terms mapped to symptom categories).

### Methodology
To enable the model to understand the contextual relationships between co-occurring HPO terms:
- HPO ID was converted into matching name(or optionally synonyms, definitions).
- A pre-trained biomedical language model (e.g., bioGPT) was fine-tuned for this multi-class classification task using the HuggingFace Transformers library.
- Base model parameters were frozen and LoRA was applied to efficiently fine-tune.

### Performance
When trained with real patient data, the llm-symptom-classifier significantly outperformed the baseline tree-based model, especially in handling complex, multi-symptom cases.
| Metric | (baseline)Tree-based model | llm-symptom-classifier |
|:-------|:-----------:|:---------------:|
| Top-1 Recall | 0.58 | 0.90 |
| Top-3 Recall | 0.70 | 0.98 |

---

## 5. Future Improvements & Production Considerations
- Authentication & Security:
  For product-level deployment, authentication(e.g., API keys or OAuth2) and input validation can be integrated to ensure secure and authorized access.
- Monitoring & Observability:
  While AWS ECS was used as an orchestration service, other open-source tools like Prometheus & Grafana can be added for service-level monitoring, performance tracking, and alerting.
- Model Generalization:
  Current model focuses on HPO inputs, but can support free-text symptom descriptions for broader applicability in clinical settings.
- Continuous Learning Pipeline:
  Automated retraining and evaluation on new datasets can be incorporated as MLOps pipeline.

