# customer-purchase-intent
A machine learning project that predicts online customer purchase intent using a 1-Nearest Neighbor (kNN) classifier.
This is the complete content for the `README.md` file for your **Online Shopping Purchase Prediction** project, based on the project specification you provided.

-----

## 📝 Project Overview

This project implements a machine learning solution to address the challenge of **predicting customer purchase intent** in an online shopping environment. By accurately predicting whether a user intends to make a purchase, an e-commerce platform can dynamically adjust content—such as showing a discount offer to a user who is not planning to purchase—to maximize conversion.

This solution utilizes a **1-Nearest Neighbor (kNN) classifier** built using the `scikit-learn` library.

## 📊 Data

The model is trained and evaluated on the **Online Shoppers Purchasing Intention Dataset**, which consists of approximately **12,000 user sessions**.

The 17 features (evidence) used for prediction include:

| Feature Category | Examples | Description |
| :--- | :--- | :--- |
| **User Activity** | `Administrative`, `Informational`, `ProductRelated` (page counts and durations) | Measures user engagement with different page types. |
| **Web Analytics** | `Bounce Rates`, `ExitRates`, `PageValues` | Metrics derived from Google Analytics. |
| **Temporal/Contextual** | `Month`, `Weekend`, `SpecialDay` | Information about the time and date of the session. |
| **User Attributes** | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | Technical and demographic information about the user. |

### Data Transformation Rules

The `load_data` function must convert all data to numeric types following these specific rules:

* **`Month`**: Mapped to an integer from 0 (January) to 11 (December).
* **`VisitorType`**: `Returning_Visitor` is mapped to `1`, and all others are mapped to `0`.
* **`Weekend`**: `TRUE` is mapped to `1`, and `FALSE` is mapped to `0`.
* **`Revenue` (Label)**: `TRUE` is mapped to the integer `1`, and `FALSE` is mapped to `0`.
* **Duration/Rate Columns**: Stored as `float`.
* **Count/Type Columns**: Stored as `int`.
