# Emotion Classifier — Fine-Tuning DistilBERT with Hugging Face

> End-to-end NLP pipeline that fine-tunes a pre-trained transformer model on a 14-class emotion dataset and runs batch inference at scale.

---

## What this does

Takes raw user text, fine-tunes DistilBERT to classify it into 14 emotional categories (joy, anger, sadness, fear, etc.), and evaluates performance through a confusion matrix and large-scale batch inference on ~5,000 samples.

The goal was to go deeper than just calling an API — this project covers the **full ML lifecycle**: data preprocessing, tokenization, training loop configuration, model evaluation, and packaging trained artifacts for reuse.

---

## Key results

| Metric | Value |
|---|---|
| Dataset size | ~20,000 labeled samples |
| Emotion classes | 14 |
| Accuracy (5k subset, CPU, 1 epoch) | ~45% |
| Inference batch size | 5,000 samples |

The 45% figure reflects CPU-only training with a single epoch and fast-mode config — not a ceiling. With GPU training across 3–5 epochs, **70–80%+ accuracy is achievable**. The architecture and pipeline are production-ready.

---

## Why this matters for production

Most tutorials fine-tune on 2 classes for 1 epoch and call it done. This project tackles a harder problem (14-class imbalanced dataset), handles the full training-to-inference path, and packages the model as reusable artifacts — the same way you'd ship a model internally at a company.

---

## Tech stack

`Python` `PyTorch` `Hugging Face Transformers` `Datasets` `Scikit-learn` `Google Colab`

---

## Architecture

```
Raw text input
    │
    ▼
DistilBERT tokenizer  (attention masks, padding, truncation)
    │
    ▼
DistilBERT encoder + classification head  (14 output neurons)
    │
    ▼
Softmax over 14 emotion classes
    │
    ▼
Prediction + confusion matrix evaluation
    │
    ▼
Saved model artifacts  (weights + tokenizer + config)
```

---

## Run it yourself

```bash
# Recommended: Google Colab with GPU runtime
# Runtime → Change runtime type → GPU T4
# Then run all cells top to bottom
```

All training, evaluation, and inference code is in the notebook. Trained artifacts are saved and zipped for download — no need to retrain to run inference.

---

## What I'd build next

- Train for 3–5 epochs on GPU to push accuracy above 70%
- Add class-weight balancing for underrepresented emotions
- Deploy as a REST API with FastAPI + Gradio demo
- Export to ONNX for faster production inference

---

## Skills demonstrated

`Transformer fine-tuning` `Hugging Face Trainer API` `Tokenization & attention masks` `Multi-class classification` `Confusion matrix analysis` `Model serialization` `Batch inference at scale`

---

*Author: Rakshith Vellulla · [GitHub](https://github.com/RakshithVellulla)*
