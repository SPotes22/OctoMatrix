# OctoMatrix — Pipeline Description

## 1. Purpose

This document describes the **end-to-end Machine Learning pipeline** used in OctoMatrix. The pipeline is designed for **research and laboratory experimentation**, focusing on detecting malicious payloads aligned with OWASP Top 10 patterns.

The pipeline prioritizes:

* Reproducibility
* Explainability
* Controlled experimentation

It does **not** aim to be a production-grade IDS or WAF.

---

## 2. High-Level Flow

The pipeline follows a linear and deterministic flow:

1. Data Collection
2. Feature Extraction
3. Model Training
4. Evaluation
5. Model Export (.pkl)
6. Quick Inference Validation

Each stage is executed sequentially within a **single script**, ensuring traceability and minimal operational complexity.

---

## 3. Data Collection

### 3.1 Data Sources

The dataset is generated programmatically and simulates multiple traffic categories:

* **Malicious Payloads**

  * SQL Injection
  * Cross-Site Scripting (XSS)
  * Path Traversal
  * Command Injection
  * XML External Entity (XXE)

* **Heuristic Patterns (CSIC-like)**

  * Parameter Pollution
  * Buffer Overflow
  * Integer Overflow
  * Format String Attacks

* **Normal Traffic**

  * REST API requests
  * Web navigation paths
  * Static file access
  * Benign user input

This approach avoids dependence on external datasets while maintaining realistic payload diversity.

### 3.2 Labeling Strategy

* `1` → Malicious payload
* `0` → Normal traffic

Labels are deterministic and generated alongside the samples.

---

## 4. Feature Extraction

The pipeline combines **statistical NLP features** with **security-oriented heuristics**.

### 4.1 TF-IDF Vectorization

* N-grams: 1 to 3
* Max features: 1500

TF-IDF captures lexical and contextual patterns commonly observed in attack payloads.

### 4.2 Advanced Security Features

Additional handcrafted features include:

* Payload length
* Special character density
* SQL keyword frequency
* XSS pattern detection
* Path traversal indicators
* Shannon entropy
* URL encoding frequency
* Whitespace ratio

These features improve robustness against simple obfuscation techniques.

---

## 5. Feature Fusion

TF-IDF vectors and handcrafted features are concatenated into a single feature space:

```
X = [TF-IDF | Advanced Features]
```

This hybrid representation balances semantic understanding and syntactic anomaly detection.

---

## 6. Model Training

### 6.1 Algorithm

* **Random Forest Classifier**

  * Number of estimators: 100
  * Parallel execution enabled

Random Forest was selected due to:

* Resistance to overfitting
* Interpretability
* Strong baseline performance for tabular + sparse data

### 6.2 Train/Test Split

* Test size: 20%
* Stratified split
* Fixed random seed for reproducibility

---

## 7. Evaluation

The model is evaluated using:

* Accuracy score
* Precision / Recall / F1-score
* Confusion matrix (via classification report)

Observed accuracy exceeds **82%** under default conditions.

---

## 8. Model Export

After successful training, the pipeline exports:

* `security_model.pkl`

  * Trained classifier
  * TF-IDF vectorizer
  * Dataset metadata
  * Export timestamp and versioning info

* `training_dataset.csv`

  * Full dataset used during training

The exported `.pkl` is intended for **controlled inference experiments only**.

---

## 9. Quick Inference Test

A lightweight validation step runs predefined payloads to verify:

* End-to-end integrity
* Feature compatibility
* Prediction sanity
  nThis step is **not** a benchmark, only a functional check.

---

## 10. Assumptions and Limitations

* Synthetic data may not capture all real-world adversarial behavior
* Obfuscation techniques beyond lexical patterns are limited
* Model confidence degrades if the pipeline or dataset is modified

These constraints are intentional and documented.

---

## 11. Ethical Use Notice

OctoMatrix is intended for:

* Academic research
* Security education
* Defensive experimentation

It must **not** be used for offensive, surveillance, or military purposes.

---

## 12. Conclusion

The OctoMatrix pipeline demonstrates that a **single, well-documented script** can serve as a valid research artifact when transparency, reproducibility, and scope boundaries are clearly defined.

