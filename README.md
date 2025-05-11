# MNIST Digit Classifier Using Keras Sequential and Dense Layers (Hugging Face Dataset)

This project demonstrates a simple handwritten digit classifier built using:
- The [MNIST dataset](https://huggingface.co/datasets/ylecun/mnist) (loaded via Hugging Face in Parquet format)
- A **Dense-only** neural network using Keras' **Sequential API**
- No CNNs, Dropout, or advanced techniques — just clean and beginner-friendly code

---

## 📂 Project Structure

- `mnist.py` – full training, evaluation, and prediction code
- `MNIST.h5` – saved trained model
- `Module_results.png` – 10 random test predictions with actual vs predicted labels

---

## 📊 Model Architecture

The model is built using only Dense layers:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
