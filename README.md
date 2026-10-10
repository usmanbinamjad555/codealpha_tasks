# CodeAlpha Internship Tasks

Projects completed during the CodeAlpha internship (March–April 2026): two machine learning pipelines in Python and two Next.js web apps. Each project lives in its own folder with its own README and can be run independently.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-189FDD)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?logo=tailwindcss&logoColor=white)

## Projects

| Project | Folder | What it does | Key result | Stack |
| --- | --- | --- | --- | --- |
| Disease Prediction | [`ML_task1/CodeAlpha_Disease_Prediction`](ML_task1/CodeAlpha_Disease_Prediction) | Predicts benign vs malignant breast tumours from the UCI Breast Cancer Wisconsin (Diagnostic) dataset; compares Random Forest and XGBoost | XGBoost: 97.37% accuracy, 0.9954 ROC-AUC on a 114-sample test set (Random Forest: 95.61%) | Python, scikit-learn, XGBoost, pandas, matplotlib, seaborn |
| Credit Scoring Model | [`ML_task2`](ML_task2) | Predicts loan approval from 15 financial features (expanded to 24) on a seeded synthetic dataset; compares Logistic Regression and Random Forest | From `reports/model_comparison.csv`: Logistic Regression ROC-AUC 0.8742, Random Forest ROC-AUC 0.8639 (accuracy 0.779 and 0.805) | Python, scikit-learn, pandas, NumPy |
| Lingua (translator) | [`task1-app`](task1-app) | Web translator with 40+ languages, auto-detect, debounced auto-translate, swap and copy; a Next.js API route proxies the MyMemory Translation API | No API key required | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| NexaFlow FAQ Bot | [`task2-app/faqbot`](task2-app/faqbot) | FAQ chatbot that matches questions with a from-scratch NLP pipeline (tokenize, stopwords, stemming, TF-IDF, cosine similarity) and shows a confidence score | No external ML libraries or API keys | Next.js 14, React 18, TypeScript, Tailwind CSS 3 |

## Repository structure

```
.
├── ML_task1/CodeAlpha_Disease_Prediction/   # Breast cancer prediction (RF vs XGBoost)
├── ML_task2/                                # Credit scoring (LR vs RF, synthetic data)
├── task1-app/                               # Lingua translator (Next.js)
└── task2-app/faqbot/                        # NexaFlow FAQ chatbot (Next.js)
```

## Getting started

### Prerequisites

- Python 3.10+ and pip (ML projects)
- Node.js and npm (web apps; Next.js 16 in `task1-app` needs Node.js 20.9 or newer)

No environment variables or API keys are needed for any of the four projects.

### Disease Prediction

```bash
cd ML_task1/CodeAlpha_Disease_Prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                 # add --no-tune to skip hyperparameter search
python predict.py               # demo predictions with the saved best model
```

### Credit Scoring Model

```bash
cd ML_task2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                  # options: --samples 10000, --no-save
```

### Lingua translator

```bash
cd task1-app
npm install
npm run dev                     # http://localhost:3000
```

### NexaFlow FAQ Bot

```bash
cd task2-app/faqbot
npm install
npm run dev                     # http://localhost:3000
```

## Known limitations

- The credit scoring model is trained on synthetic data, so its metrics do not reflect real lending performance.
- The disease prediction model is for educational use only and must not inform clinical decisions.
- Lingua depends on MyMemory's free tier, which has daily usage limits.
- The FAQ bot only answers questions covered by its built-in FAQ dataset.

See each project's own README for details on the approach, structure and results.

---

Author: Muhammad Usman — [LinkedIn](https://www.linkedin.com/in/muhammad-usman-the-dev) · [Portfolio](https://muhammad-usman-engineer.vercel.app)
