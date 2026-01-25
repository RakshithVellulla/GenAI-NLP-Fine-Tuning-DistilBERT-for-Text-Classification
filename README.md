# GenAI NLP DistilBERT
📌 Project Overview
This project demonstrates an end-to-end Natural Language Processing (NLP) pipeline using Hugging Face Transformers to build a text classification system.
We fine-tuned a pre-trained DistilBERT model on an emotions dataset and evaluated its performance using:
Confusion Matrix
Accuracy
Large-scale batch inference
The entire workflow was executed in Google Colab.

🎯 Objectives
Use a pre-trained Transformer model from Hugging Face
Tokenize and preprocess raw text
Encode categorical labels
Fine-tune the model on a custom dataset
Evaluate predictions on small and large samples
Save and zip trained model artifacts
Prepare the model for deployment/inference

🧠 Concepts Covered
✔ Machine Learning (ML)
Train / test split
Optimization using gradient descent
Evaluation metrics
Overfitting vs generalization

✔ Deep Learning (DL)
Transformer architectures
DistilBERT encoder
Classification head
Softmax outputs

✔ Natural Language Processing (NLP)
Tokenization
Attention masks
Sequence classification
Emotion classification
Inference pipelines

📂 Dataset
Emotion dataset containing:
Text → raw user sentences
Emotion → category label
Labels were encoded numerically before training.

🛠️ Tools & Libraries
Python
PyTorch
Hugging Face Transformers
Datasets
Scikit-learn
NumPy / Pandas
Google Colab

⚙️ Workflow

1️⃣ Load dataset
2️⃣ Clean and inspect text
3️⃣ Encode labels
4️⃣ Tokenize using DistilBERT tokenizer
5️⃣ Convert to Hugging Face Dataset format
6️⃣ Load pre-trained model with correct number of labels
7️⃣ Configure TrainingArguments
8️⃣ Train using Trainer
9️⃣ Save model + tokenizer
🔟 Zip trained artifacts
1️⃣1️⃣ Run inference on new data
1️⃣2️⃣ Generate confusion matrix & accuracy
1️⃣3️⃣ Test on large dataset subset

📊 Model Evaluation
On a large subset (~5,000 rows):
Accuracy ≈ 45%
14-class emotion classification
Given:
• CPU-only training
• single epoch
• fast-mode configuration
this demonstrates successful fine-tuning and learning.

📁 Saved Artifacts
After training, the following were saved:
distilbert_finetuned/
distilbert_finetuned_trainer/
distilbert_finetuned_10k.zip
These include:
✔ model weights
✔ tokenizer
✔ configuration files
They can be reloaded without retraining.

▶️ How to Run in Google Colab
Open notebook in Colab
Change runtime to GPU (if available)
Run cells from top to bottom
Wait for training to complete
Execute saving & evaluation cells
Download ZIP file from /content

🧪 Inference Example
A function was implemented to:
tokenize new text
move tensors to device
predict class
return label ID
Used for:
• batch predictions
• evaluation
• production-style testing

💡 Learning Outcomes
Through this assignment I learned:
How Hugging Face models are used in real ML workflows
Fine-tuning vs using frozen models
Tokenization & attention masks
Trainer API
Model evaluation strategies
Packaging trained models
Confusion matrix interpretation
Scaling inference

🚀 Future Improvements
With GPU and larger training:
Train for more epochs
Tune learning rate
Handle class imbalance
Increase max sequence length
Deploy as API using FastAPI/Gradio
Add monitoring & retraining loop

👤 Author
Rakshith Vellulla
