# 🛡️ Conformal Risk Control: Theory, Proofs & Visual Benchmark Suite

An end-to-end implementation, visual benchmark, and empirical study suite for the foundational paper:
> **"Conformal Risk Control"** by *Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (UC Berkeley, MIT, Stanford, Google Research)*.

---

## 📌 Table of Contents
1. [Section 1 & 2: Theoretical Formulation & Calibration](#1-section-1--2-theoretical-formulation--calibration)
2. [Section 3.1: Gut Polyp Tumor Segmentation (FNR Control)](#2-section-31-gut-polyp-tumor-segmentation-fnr-control)
3. [Section 3.2: Multi-Label Object Detection (MS COCO)](#3-section-32-multi-label-object-detection-ms-coco)
4. [Section 3.3 & 3.4: Hierarchical Classification & NLP QA](#4-section-33--34-hierarchical-classification--nlp-qa)
5. [Section 4.1: Weighted Risk Control Under Distribution Shift](#5-section-41-weighted-risk-control-under-distribution-shift)
6. [Quick Start](#-quick-start)

---

## 1. Section 1 & 2: Theoretical Formulation & Calibration

Standard conformal prediction guarantees coverage over a binary miscoverage indicator:

$$\mathbb{P}(Y_{n+1} \notin \mathcal{C}(X_{n+1})) \le \alpha$$

**Conformal Risk Control (CRC)** generalizes this guarantee to any arbitrary **bounded, monotone loss function** $\ell(\mathcal{C}_\lambda(X), Y) \in (-\infty, B]$:

$$\mathbb{E}\left[\ell\left(\mathcal{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1}\right)\right] \le \alpha$$

### 📐 Finite-Sample Calibration Threshold (Equation 4)
Given $n$ exchangeable calibration points, the optimal threshold $\hat{\lambda}$ is chosen as:

$$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \widehat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha \right\}$$

![Theoretical Calibration](outputs/fig1_theory_calibration.png)

---

## 2. Section 3.1: Gut Polyp Tumor Segmentation (FNR Control)

In clinical imaging, standard miscoverage fails because missing $2\text{ mm}^3$ of tumor is not catastrophic, but missing the entire tumor is. CRC defines loss as **False Negative Rate (FNR)**:

$$L_i^{\text{FNR}}(\lambda) = 1 - \frac{|Y_i \cap \mathcal{C}_\lambda(X_i)|}{|Y_i|}$$

![Tumor Segmentation](outputs/fig2_tumor_segmentation.png)

* **Target Risk:** $\alpha = 0.10$
* **Empirical Risk (1,000 Trials):** **%9.87** ($\le 10\%$, Validated)

---

## 3. Section 3.2: Multi-Label Object Detection (MS COCO)

For multi-label object detection across $K=80$ classes, the prediction set includes all labels with scores $\ge 1 - \hat{\lambda}$. CRC controls the average fraction of missed bounding classes.

![MS COCO Detection](outputs/fig3_multilabel_detection.png)

* **Target Risk:** $\alpha = 0.10$
* **Empirical Risk (1,000 Trials):** **%9.96** ($\le 10\%$, Validated)

---

## 4. Section 3.3 & 3.4: Hierarchical Classification & NLP QA

![Hierarchical and NLP](outputs/fig4_hierarchical_and_nlp.png)

### 🌳 3.3 Hierarchical ImageNet WordNet Classification
* **Loss Metric:** Taxonomy tree distance $d_H(y, s) / D$.
* **Target Risk:** $\alpha = 0.05$ $\rightarrow$ **Empirical Risk: 0.0499**

### 💬 3.4 Open-Domain Question Answering (Google Natural Questions)
* **Loss Metric:** $1 - \max(\text{Token F1-Score})$.
* **Target Risk:** $\alpha = 0.30$ $\rightarrow$ **Empirical Risk: 0.2996**

---

## 5. Section 4.1: Weighted Risk Control Under Distribution Shift

When the test distribution deviates from the training distribution ($P_{\text{test}} \neq P_{\text{train}}$), standard methods fail. Using importance weights $w(x) = \frac{\text{d}P_{\text{test}}(x)}{\text{d}P_{\text{train}}(x)}$, CRC maintains finite-sample validity:

$$\hat{\lambda}(x) = \inf \left\{ \lambda : \frac{\sum_{i=1}^n w(X_i)L_i(\lambda) + w(x)B}{\sum_{i=1}^n w(X_i) + w(x)} \le \alpha \right\}$$

![Distribution Shift](outputs/fig5_distribution_shift.png)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt