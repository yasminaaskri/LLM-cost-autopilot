[![CI](https://github.com/yasminaaskri/LLM-cost-autopilot/actions/workflows/ci.yml/badge.svg)](https://github.com/yasminaaskri/LLM-cost-autopilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

> **Intelligent routing layer that sends each prompt to the cheapest capable LLM, with LLM-as-judge verification and automatic escalation.**
>
> On 96 logged requests, routing cost **40% less** than sending everything to Llama 3.3 70B (see [Results](#results) for how this was measured and its limits).

---

## 📋 Table of Contents

- [Why This Project?](#why-this-project)
- [Key Features](#key-features)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Dashboard](#dashboard)
- [The Flywheel (Self-Improvement Loop)](#the-flywheel-self-improvement-loop)
- [Project Structure](#project-structure)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)

---

## Why This Project?

### The Problem

Many applications send every request to their most expensive model, even simple ones such as extraction, formatting or yes/no questions. Those prompts can often be served by a much cheaper model.

### The Solution

**LLM Cost Autopilot** intelligently routes each prompt to the cheapest capable model based on complexity:

| Task Complexity | Example | Model | Cost per 1k Tokens |
|-----------------|---------|-------|-------------------|
| **Simple** | Extraction, formatting, yes/no | Llama 3.1 8B | $0.000035 |
| **Moderate** | Summarization, classification | Gemini 2.5 Flash | $0.00030 |
| **Complex** | Reasoning, coding, analysis | Llama 3.3 70B | $0.00044 |

The system also:
- ✅ **Verifies quality** asynchronously using LLM-as-judge
- ✅ **Automatically escalates** to better models when quality fails
- ✅ **Learns from mistakes** via weekly retraining flywheel
- ✅ **Provides real-time cost savings** dashboard

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Intelligent Routing** | Classifies prompts into 3 complexity tiers using a Random Forest classifier trained on 189 hand-labeled prompts |
| 🤖 **LLM-as-Judge** | Asynchronously verifies quality using the best model as judge (Llama 3.3 70B) |
| ⚡ **Auto-Escalation** | Automatically re-runs with expensive model when cheap model quality < 3.0/5 |
| 🔄 **Self-Improving** | Weekly retraining on routing failures (the flywheel) |
| 📊 **Real-Time Dashboard** | Live cost savings, routing distribution, and quality metrics |
| ⚙️ **Hot-Reload Routing** | Change tier→model mapping without restarting the server |
| 🐳 **Docker Ready** | One-command deployment with Docker Compose |
| ✅ **CI/CD Pipeline** | Pytest unit tests in GitHub Actions; 20-prompt golden-set regression test (80% gate) |
| 🔍 **Full Audit Trail** | Every request logged with cost baseline for savings calculation |

---

## Results

### 📊 Measured results

| Metric | Value | How it was measured |
|--------|-------|---------------------|
| **Cost reduction** | **40.4%** (37.3% including escalations) | 96 logged requests (49 distinct prompts), June 2026, vs. sending every request to Llama 3.3 70B |
| **Classifier accuracy** | 89.5% | Held-out 20% split of the 189 labeled prompts |
| **Golden prompts** | 19/20 | `tests/golden_prompts.json`, run locally |

### 💲 Model tiers

| Tier | Model | Cost per 1k Input | Cost per 1k Output | Used For |
|------|-------|-------------------|--------------------|----------|
| Complex | Llama 3.3 70B | $0.00044 | $0.00067 | Reasoning, coding, analysis |
| Moderate | Gemini 2.5 Flash | $0.00030 | $0.00025 | Summarization, classification |
| Simple | Llama 3.1 8B | $0.000035 | $0.000055 | Extraction, formatting, yes/no |

### ⚠️ Limitations of the current evaluation

- **The baseline is a price calculation, not a separate run.** Each request's real token counts are priced at Llama 3.3 70B rates. 70B was not called to measure its own output length.
- **The 96 requests come from several manual runs**, not one fixed benchmark. Savings varied from 5% to 92% per day depending on the prompt mix.
- **Quality was only spot-checked.** 10 of the 96 requests were scored by the judge, so this README makes **no quality-parity claim**. Some Gemini 2.5 Flash answers were truncated and scored poorly.
- **The judge is the same model as the baseline** (Llama 3.3 70B), so self-preference bias is possible.

### 🔜 Next evaluation steps

1. Run `scripts/load_test.py` on its fixed, seeded 200-prompt set and commit the results.
2. Score every request with the verifier and store both the routed and the reference score.
3. Use a judge from a different model family than the baseline.
4. Fix the Gemini output truncation before re-measuring.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **API Framework** | FastAPI + Uvicorn |
| **Validation** | Pydantic |
| **Machine Learning** | Scikit-learn (Random Forest) |
| **Token Counting** | Tiktoken |
| **Database** | SQLite (WAL mode) |
| **Dashboard** | Streamlit + Plotly |
| **Deployment** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **LLM Providers** | Groq (Llama 3.3 70B, Llama 3.1 8B), Google Gemini (2.5 Flash) |

---

## Architecture

<img width="2720" height="3600" alt="llm_cost_autopilot_architecture" src="https://github.com/user-attachments/assets/a14b304f-1d1e-4bf1-8e24-ad5645b135b1" />



---

## Quick Start

### Prerequisites

- Python 3.11+
- API Keys:
  - [Groq API Key](https://console.groq.com/keys) (free tier)
  - [Google Gemini API Key](https://aistudio.google.com/apikey) (free tier)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yasminaaskri/LLM-cost-autopilot.git
cd LLM-cost-autopilot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 5. Train the classifier
python -m src.classifier.train

# 6. Run the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 7. Run the dashboard (in another terminal)
streamlit run src/dashboard/app.py

# 8. Test the API
curl http://localhost:8000/health

# 9. Open Swagger UI
# Go to: http://localhost:8000/docs

# 10. Open Dashboard
# Go to: http://localhost:8501

```
---

## Dashboard

The Streamlit dashboard provides **real-time visibility into cost savings, routing decisions, quality metrics, and system health.**

Open the dashboard at:

```
http://localhost:8501
```

### Dashboard Sections

| Section | Description |
|----------|-------------|
| 📊 **Overview** | Displays total requests, actual cost, baseline cost, savings percentage, average latency, and average quality score |
| 💰 **Daily Cost** | Bar chart comparing actual cost vs. baseline cost over time |
| 🧠 **Routing Distribution** | Pie chart showing how requests are distributed across Llama 3.1 8B, Gemini 2.5 Flash, and Llama 3.3 70B |
| ⭐ **Quality Distribution** | Histogram of LLM-as-Judge quality scores |
| ⚡ **Escalation Rate** | Trend of automatic escalations over time |
| 📋 **Request Audit** | Recent requests with selected model, latency, token usage, quality score, and escalation status |
| ⚙️ **Live Routing Config** | Modify tier → model mapping without restarting FastAPI |
| 🔄 **Retrain Button** | One-click retraining of the routing classifier |

---
<img width="1166" height="542" alt="Capture d&#39;écran 2026-06-22 130051" src="https://github.com/user-attachments/assets/9a347df3-05c5-41d1-874c-e51ad5196bb4" />

<img width="1919" height="915" alt="Capture d&#39;écran 2026-06-23 102801" src="https://github.com/user-attachments/assets/a600d6e0-2f7a-4c8c-acda-4e5777c11845" />

<img width="1914" height="910" alt="Capture d&#39;écran 2026-06-23 102815" src="https://github.com/user-attachments/assets/a3cbdca9-8311-414a-bbb8-1407d3538941" />

<img width="1919" height="916" alt="Capture d&#39;écran 2026-06-23 102829" src="https://github.com/user-attachments/assets/f470b388-f04d-49b4-b360-b23ca0f4ce95" />


## The Flywheel (Self-Improvement Loop)

```
Routing Failure
       │
       ▼
Logged to Database
       │
       ▼
Weekly Retraining
       │
       ▼
Improved Classifier
       │
       ▼
Better Routing Decisions
       │
       ▼
Fewer Failures
       ▲
       └───────────────────────────────┘
```

### How the Flywheel Works

The routing system continuously improves through a self-reinforcing feedback loop.

| Step | Description |
|------|-------------|
| **1. Routing Failure** | When a cheap model produces a response with quality score < **3.0 / 5**, the request is marked as a routing failure. |
| **2. Log Failure** | The request is stored in the `routing_failures` table together with the prompt, predicted tier, correct tier, and quality gap. |
| **3. Accumulate Data** | Failures are collected until the weekly retraining job (or manual retraining). |
| **4. Weekly Retraining** | Every Sunday at **2:00 AM**, the classifier is retrained using the newly collected failures. |
| **5. Accuracy Guard** | The new classifier is accepted only if its accuracy is at least **old accuracy − 2%**, preventing performance degradation. |
| **6. Deploy Updated Model** | The improved classifier replaces the previous model. |
| **7. Better Routing** | Future prompts are routed more accurately, reducing escalations and improving cost savings. |

---

## Project Structure

```
LLM-cost-autopilot/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
│
├── config/
│   ├── registry.yaml              # Model registry and pricing
│   ├── routing.yaml               # Tier → Model mapping
│   └── tiers.yaml                 # Complexity tier definitions
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── classifier/
│   │   ├── features.py
│   │   ├── predict.py
│   │   └── train.py
│   │
│   ├── providers/
│   │   ├── dispatcher.py
│   │   ├── groq_provider.py
│   │   └── google_provider.py
│   │
│   ├── router/
│   │   └── router.py
│   │
│   ├── verifier/
│   │   ├── escalation.py
│   │   ├── judge.py
│   │   └── verifier.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── config.py
│   ├── database.py
│   └── models.py
│
├── scripts/
│   ├── baseline_run.py
│   ├── load_test.py
│   ├── retrain.py
│   └── worker.py
│
├── tests/
│   ├── golden_prompts.json
│   ├── test_day1.py
│   ├── test_day5.py
│   └── test_verifier_escalation.py
│
├── data/
│   ├── labeled_prompts.csv
│   └── autopilot.db
│
├── models/
│   └── classifier.pkl
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .env.example
```

---

## CI/CD

GitHub Actions automatically performs:

- ✅ Dependency installation
- ✅ Unit tests
- ✅ Golden prompt regression tests
- ✅ Routing verification
- ✅ Quality verification tests
- ✅ Build validation

Every pull request is automatically validated before merging.

---

## Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes.

```bash
git commit -m "Add amazing feature"
```

4. Push the branch.

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**.

See the **LICENSE** file for more information.

---

## Acknowledgments

- Groq for providing fast inference APIs
- Google Gemini API
- FastAPI
- Scikit-learn
- Streamlit
- Plotly
- Docker

---

## Author

**Yasmina Askri**

GitHub: https://github.com/yasminaaskri

---

# ⭐ Star This Project

If you found this project useful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future development.

---

Built with ❤️ by **Yasmina Askri**
