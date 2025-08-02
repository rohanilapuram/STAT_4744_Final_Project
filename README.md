# Amazon Alexa Review Sentiment Classification

This project applies deep learning techniques to perform sentiment classification on Amazon Alexa customer reviews. Using a Recurrent Neural Network (RNN) with an attention mechanism, the model classifies reviews as either **positive** or **negative** based on their textual content.

## 🧠 Motivation

Understanding customer sentiment at scale is critical for businesses to improve product quality, customer service, and engagement. This project explores how attention-augmented RNNs can be leveraged to extract insights from real-world customer feedback data.

## 📊 Dataset

- **Source**: Kaggle
- **Size**: 3,150 reviews
- **Target variable**: `feedback` (1 for positive, 0 for negative)
- **Feature used**: `verified_reviews`

## 🔧 Preprocessing Pipeline

- Text normalization (lowercasing, punctuation removal, etc.)
- Stopword removal and lemmatization (using `nltk`)
- Tokenization using Keras’ `Tokenizer` with a vocab size of 1000
- Padding sequences to fixed length of 50 tokens

## 🧱 Model Architecture

Implemented in **PyTorch**, the model pipeline includes:
1. **Embedding Layer** – transforms tokens into dense vectors
2. **Unidirectional RNN** – encodes sequential context
3. **Attention Layer** – weighs important time steps
4. **Fully Connected Layer** – outputs logits for binary classification

> **Loss Function**: Binary Cross-Entropy  
> **Optimizer**: Adam (learning rate = 0.001)  
> **Training Epochs**: 5  
> **Batch Size**: 32

## 📈 Results

- **Validation Accuracy**: **92.54%**
- Strong performance on positive reviews
- Struggled with the minority (negative) class due to class imbalance
- Visualized training/validation loss, accuracy, confusion matrix, and precision-recall scores

## 🧐 Error Analysis

- Negative reviews were often short, ambiguous, or contained rare/out-of-vocabulary tokens
- Suggestions for improvement include:
  - Using weighted loss functions
  - Incorporating more negative samples
  - Leveraging subword tokenization (e.g., Byte Pair Encoding)
  - Using pre-trained models like BERT or BiLSTMs

## 🧪 Future Work

- Integrate BERT or Transformer-based classifiers
- Experiment with bidirectional RNNs or LSTMs
- Apply SMOTE or other balancing techniques
- Deploy model as a web API for real-time sentiment classification

## 📁 Files

- `STAT_4744_Final_Project.ipynb`: Full implementation in Jupyter Notebook
- `STAT_4744_Final_Project.pdf`: Final report with analysis and figures
- `output.csv`: Contains the dataset with feedback given by the model



---

**Author**: Rohan Ilapuram  
**Email**: ilapuramrohan@gmail.com  
**Affiliation**: Virginia Tech, Department of Statistics  
