# customer-purchase-intent
A machine learning project that predicts online customer purchase intent using a 1-Nearest Neighbor (kNN) classifier.
This is the complete content for the `README.md` file for your **Online Shopping Purchase Prediction** project, based on the project specification you provided.

-----

### Online Shopping Purchase Prediction

[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
[](https://choosealicense.com/licenses/mit/)

## 📝 Project Overview

[cite\_start]This project implements a machine learning solution to address the challenge of **predicting customer purchase intent** in an online shopping environment[cite: 3]. [cite\_start]By accurately predicting whether a user intends to make a purchase, an e-commerce platform can dynamically adjust content—such as showing a discount offer to a user who is not planning to purchase—to maximize conversion[cite: 12].

[cite\_start]This solution utilizes a **1-Nearest Neighbor (kNN) classifier** built using the `scikit-learn` library[cite: 74, 75].

## ✨ Key Features

  * [cite\_start]**Purchase Intent Prediction:** Classifies user sessions as either resulting in a purchase (`1`) or not (`0`) based on 17 features[cite: 46, 72].
  * [cite\_start]**k-Nearest Neighbor Classifier:** Implements a k-Nearest Neighbors classifier with $k=1$[cite: 74].
  * [cite\_start]**Data Preprocessing Pipeline:** Custom logic to load data from `shopping.csv` [cite: 35] [cite\_start]and convert mixed data types (float, integer, categorical, and Boolean) into a purely numeric format required for the kNN model[cite: 65, 66].
  * [cite\_start]**Advanced Evaluation Metrics:** Model performance is rigorously measured using **Sensitivity** (True Positive Rate) and **Specificity** (True Negative Rate) to ensure a balanced classifier that performs reasonably well on both metrics[cite: 24, 29, 78, 79].

## 📊 Data

[cite\_start]The model is trained and evaluated on the **Online Shoppers Purchasing Intention Dataset**, which consists of approximately **12,000 user sessions**[cite: 17, 37].

[cite\_start]The 17 features (evidence) used for prediction include[cite: 62]:

| Feature Category | Examples | Description |
| :--- | :--- | :--- |
| **User Activity** | [cite\_start]`Administrative`, `Informational`, `ProductRelated` (page counts and durations) [cite: 38] | Measures user engagement with different page types. |
| **Web Analytics** | [cite\_start]`Bounce Rates`, `ExitRates`, `PageValues` [cite: 39] | Metrics derived from Google Analytics. |
| **Temporal/Contextual** | [cite\_start]`Month`, `Weekend`, `SpecialDay` [cite: 40, 41, 44] | Information about the time and date of the session. |
| **User Attributes** | [cite\_start]`OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` [cite: 42, 43] | Technical and demographic information about the user. |

### Data Transformation Rules

[cite\_start]The `load_data` function must convert all data to numeric types [cite: 65, 66] [cite\_start]following these specific rules[cite: 67, 68]:

  * [cite\_start]**`Month`**: Mapped to an integer from 0 (January) to 11 (December)[cite: 69].
  * [cite\_start]**`VisitorType`**: `Returning_Visitor` is mapped to `1`, and all others are mapped to `0`[cite: 43, 70].
  * [cite\_start]**`Weekend`**: `TRUE` is mapped to `1`, and `FALSE` is mapped to `0`[cite: 44, 71].
  * [cite\_start]**`Revenue` (Label)**: `TRUE` is mapped to the integer `1`, and `FALSE` is mapped to `0`[cite: 46, 72].
  * [cite\_start]**All other duration/rate columns**: Stored as `float`[cite: 68].
  * [cite\_start]**All other count/type columns**: Stored as `int`[cite: 67].

## 💻 Tech Stack

  * **Language:** Python 3
  * [cite\_start]**Machine Learning:** `scikit-learn` (`KNeighborsClassifier`) [cite: 32, 75]
  * [cite\_start]**Data Handling:** Python's built-in `csv` module (or other standard libraries like `numpy` or `pandas`) [cite: 83, 87]

## ⚙️ Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/Online-Shopping-Prediction.git
    cd Online-Shopping-Prediction
    ```

2.  **Install dependencies:**
    The core dependency is `scikit-learn`.

    ```bash
    pip install scikit-learn
    ```

    *Note: Ensure the `shopping.csv` data file is present in the project directory.*

## 🚀 Running the Project

[cite\_start]To run the classifier on the provided data, execute the `shopping.py` script and pass the data filename as an argument[cite: 4]:

```bash
python shopping.py shopping.csv
```

### Example Output

[cite\_start]The script prints the evaluation metrics upon completion[cite: 5, 6, 7, 8].

```
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%  <-- Sensitivity
True Negative Rate: 90.55%  <-- Specificity
```

## 📐 Project Implementation

The core machine learning logic is implemented by completing the following functions in `shopping.py`:

1.  [cite\_start]**`load_data(filename)`**: Returns a tuple of `(evidence, labels)` by reading the CSV and applying the necessary data transformations[cite: 56].
2.  [cite\_start]**`train_model(evidence, labels)`**: Trains and returns a `KNeighborsClassifier` instance with **k=1**[cite: 74].
3.  [cite\_start]**`evaluate(labels, predictions)`**: Calculates and returns the model's **sensitivity** (true positive rate) and **specificity** (true negative rate) as two floating-point values between 0 and 1[cite: 77].

## 📜 Acknowledgements

Data set provided by Sakar, C.O., Polat, S.O., Katircioglu, M. et al. [cite\_start]Neural Comput & Applic (2018)[cite: 90].
