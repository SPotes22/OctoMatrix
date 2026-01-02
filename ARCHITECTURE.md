# Architecture.md

## 1. Overview

This document defines the **reference architecture** for the integration of **OctoMatrix** as a security middleware inside **Parchate Pereira**.

The system is designed under the principle:

> **Safe by Design is not a feature, it is the foundation.**

OctoMatrix operates as a **defensive intelligence layer** that sanitizes, classifies, and predicts malicious intent in user inputs before they propagate into business logic, AI services, or persistence layers.

---

## 2. System Roles

### Parchate Pereira (Application Layer)

* Public-facing tourism platform.
* Handles user interaction, sessions, business rules, and premium services.
* Exposes HTTP endpoints via Flask.

### OctoMatrix (Security Middleware)

* Independent security subsystem.
* Acts as a **request gatekeeper**.
* Returns contextual security predictions.
* Loads a trained `.pkl` ML model with **82%+ accuracy**.

---

## 3. High-Level Architecture

```
Client
  |
  v
[ HTTP Request ]
  |
  v
┌─────────────────────────────┐
│   OctoMatrix Middleware     │
│                             │
│  1. Input Sanitization      │
│  2. Regex Deterministic     │
│  3. ML Prediction (.pkl)    │
│  4. Risk Classification     │
└─────────────────────────────┘
  |
  v
┌─────────────────────────────┐
│   Application Logic         │
│   (Parchate Pereira)        │
│                             │
│ - Events                    │
│ - HelpBoy                   │
│ - Gemini Premium            │
└─────────────────────────────┘
  |
  v
[ Response ]
```

---

## 4. Request Lifecycle

### Step 1. User Input

* Input enters via REST endpoints (`/octo/predict`, `/premium/recommend`, `/helpBoy`).
* Maximum input length enforced (≤ 500 chars).

### Step 2. Sanitization Layer

Handled in `OctoMatrixBrain._sanitize_input()`:

* Removes high-risk tokens (`<`, `>`, `script`, `javascript`, `eval`).
* Enforces hard length truncation.

Purpose:

* Neutralize trivial XSS and script-based payloads.
* Prevent model poisoning and unsafe downstream execution.

### Step 3. Deterministic Regex Layer

Embedded in the ML feature extraction pipeline:

* SQL keyword detection
* XSS signature detection
* Path traversal markers
* Command injection patterns

Purpose:

* Fast rejection and signal amplification for known OWASP Top 10 patterns.

### Step 4. ML Prediction Layer

Powered by a **RandomForestClassifier** exported as `.pkl`.

Features include:

* TF-IDF (1–3 grams)
* Payload entropy
* Symbol density
* Encoded character ratio
* Keyword frequency

Output:

* Binary classification: `Normal` vs `Attack`
* Confidence score

### Step 5. Decision Layer

The application decides how to act based on OctoMatrix output:

* Allow request
* Soft-handle (log + degrade)
* Hard-block (future extension)

---

## 5. Lazy Loading Strategy

OctoMatrix uses **lazy model loading**:

```python
if not self._loaded:
    self.load_model()
```

Benefits:

* Fast application startup
* Low memory footprint
* Suitable for constrained environments (Render, Chromebooks)

---

## 6. Vertical Scalability Model

The architecture scales **vertically**, not horizontally by default:

* Stateless prediction calls
* No session coupling
* No external state required for inference

This enables:

* Single-node deployment
* Predictable costs
* Planet-scale replication by duplication, not orchestration

---

## 7. Security Scope

### Covered Threats

* XSS
* SQL Injection
* Command Injection
* Path Traversal
* Payload fuzzing / typo attacks

### Explicitly Out of Scope

* Authentication bypass
* Business logic abuse
* Zero-day browser exploits
* Client-side sandbox escapes

These boundaries are intentional and documented.

---

## 8. Integration Contract

OctoMatrix guarantees:

* No mutation of application state
* No outbound network calls during inference
* Deterministic behavior for identical inputs

Application guarantees:

* Inputs are routed through OctoMatrix before AI or DB usage
* Predictions are respected, not ignored

---

## 9. Reference Implementation

* **brain.py**: Lazy-loaded inference engine
* **app.py**: Secure Flask integration
* **octomatrix_poc_moe_owasp.py**: Training + export pipeline

Parchate Pereira serves as the **cleanest live reference** of the architecture.

---

## 10. Design Philosophy

* Simple beats clever
* Deterministic beats opaque
* Ethics beats optimization

If it runs safely on a Chromebook, it scales to a planet.

---

## 11. Status

* Architecture: Stable
* Security Model: Trained (82%+ accuracy)
* Documentation: In progress → 99%

