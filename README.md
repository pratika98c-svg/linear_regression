# Linear Regression From Scratch (Using NumPy)

## Project Overview

This project implements **Linear Regression from scratch using Gradient Descent** without using machine learning libraries like `sklearn`.

The goal of this assignment is to understand the **core mathematics behind machine learning algorithms** and how models learn from data.

The model is trained on a simple dataset representing **house size vs house price**.

---

## Dataset

The dataset used in this project:

| House Size (sq ft) | Price |
| ------------------ | ----- |
| 500                | 100   |
| 800                | 150   |
| 1200               | 200   |
| 1500               | 250   |
| 1800               | 300   |

The model learns the relationship between **house size and price**.

---

## Model

We implement the linear regression equation:

Price = m × Size + b

Where:

* **m** = slope (weight)
* **b** = intercept (bias)

The parameters are optimized using **Gradient Descent**.

---

## Loss Function

We use **Mean Squared Error (MSE)** as the loss function.

MSE = (1/n) Σ (y − ŷ)²

Where:

* **y** = actual values
* **ŷ** = predicted values

---

## Gradient Descent Update

The gradients used to update parameters are:

dm = (-2/n) Σ x(y − ŷ)
db = (-2/n) Σ (y − ŷ)

Parameter update rule:

m = m − learning_rate × dm
b = b − learning_rate × db

---

## Libraries Used

* NumPy
* Matplotlib

---

## Output

The model prints the **final learned equation** and plots the **Loss vs Epoch graph**.

Example:

Price = m × Size + b

The loss decreases with each epoch as the model learns the best fit line.

---

## How to Run

Clone the repository:

```
git clone https://github.com/yourusername/linear_regression.git
```

Install dependencies:

```
pip install numpy matplotlib
```

Run the script:

```
python linear_regression.py
```

---

## Learning Outcome

Through this project, I learned:

* How linear regression works internally
* How gradient descent updates parameters
* How loss functions guide model training
* How to implement machine learning algorithms using only NumPy

---

## Author

Pratika
