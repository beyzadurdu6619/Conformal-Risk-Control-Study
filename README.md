# 🛡️ Conformal Risk Control: Theory, Benchmarks & PDF Guide

An academic study suite and comprehensive Turkish study guide for the paper:
> **"Conformal Risk Control"** by *Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (UC Berkeley, MIT, Stanford, Google Research)*.

---

## 📊 Core Concepts
* **Standard Conformal Prediction:** Binary 0/1 miscoverage loss.
* **Conformal Risk Control (CRC):** Arbitrary bounded monotone continuous loss with exact risk control guarantees: $\mathbb{E}[\ell(C_{\hat{\lambda}}(X_{n+1}), Y_{n+1})] \le \alpha$.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Kütüphaneleri Yükle
pip install -r requirements.txt

# 2. Simülasyonları Çalıştır
python src/run_experiments.py
python src/visualize_comparison.py

# 3. PDF Dokümanını Üret
python src/generate_pdf.py
```
