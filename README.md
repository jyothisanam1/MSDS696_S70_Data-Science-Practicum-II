# Consumer Complaint Triage: Predicting Relief Outcomes at Scale

**Author:** Jyothi Sanam  
**Course:** MSDS 696 — Data Science Practicum II  

## 📌 Summary
This project builds an automated machine learning triage engine for retail banking operations. By predicting the likelihood that an incoming consumer complaint will require financial or operational relief, the model allows banks to abandon rigid "first-in, first-out" (FIFO) queues and prioritize high-risk compliance cases. 

**The Bottom Line:** Flag a complaint for priority review when the model has >0.40 confidence, catches 85% of relief cases, and cuts the standard queue by 33%.

## 📊 The Data & Methodology
The model was trained on a refined, text-rich subset of the official **[Consumer Financial Protection Bureau (CFPB) Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/search/?chartType=line&company_response=Closed%20with%20explanation&company_response=Closed%20with%20non-monetary%20relief&company_response=Closed%20with%20monetary%20relief&dateInterval=Month&date_received_max=2026-06-30&date_received_min=2023-07-01&has_narrative=true&lens=Product&product=Credit%20card&product=Checking%20or%20savings%20account&searchField=all&subLens=sub_product&tab=Trends)**. 

### Dataset Source & Parameters
* **Source:** CFPB Public Consumer Complaint Database, filtering specifically for retail banking transactions.
* **Temporal Window:** July 1, 2023 – June 30, 2026.
* **Scope:** Downsampled from over 12.5M total CFPB records to a targeted pool of ~213k retail banking complaints containing consumer narrative text.
* **Target Categories:** Core retail products including `Checking or savings account` and `Credit card`.
* **Target Variable:** Binary classification predicting if a complaint will be `Closed with explanation` (0) or `Closed with relief` (1) (combining monetary and non-monetary remediation).

### Feature Engineering
The project utilizes horizontal matrix fusion, combining a 15,000-feature sparse TF-IDF text matrix with 5 dense, domain-specific behavioral indicators discovered during Exploratory Data Analysis (EDA):
1. Word Count
2. Character Length
3. Privacy Mask Density (e.g., `XXXX` redactions)
4. P2P Payment App Indicator (e.g., Zelle, Venmo)
5. Identity Theft Indicator

## 🧠 Model Architecture & Results
Baseline models (Logistic Regression, standard LightGBM) missed ~80% of relief cases due to a lack of structural context. While a deep learning Large Language Model (**DistilBERT**) solved the recall issue (72.36%), it acted as a "black box" that violated banking explainability requirements and incurred heavy cloud compute costs.

The **Proposed Hybrid LightGBM Model** solves this by injecting the 5 behavioral EDA features into the baseline. 
* **Recall:** 67.87% (Near-parity with the LLM)
* **ROC AUC:** 0.7467 (Mathematically superior queue-sorting power compared to DistilBERT)
* **Explainability:** 100% transparent feature importance for compliance auditors.
* **Compute ROI:** Trains in seconds on standard CPUs rather than requiring expensive T4 GPUs.

## 📂 Repository Structure
This repository contains the main Python codebase as well as a chronological log of weekly artifacts (Status Reports, LLM Collaboration Logs, and Capstone Defense Practice Talks).

    ├── ConsumerComplaintTriage_Project_Notebook.ipynb  # Core Data Pipeline & Modeling Code
    ├── LICENSE
    ├── week1/
    │   ├── Jyothi_MSDS696_Wk1_Project_Proposal.docx
    │   ├── Jyothi_Project_LLM_Brainstorming_Log_Wk1.docx
    │   └── Jyothi_Wk1_Practice_Talk.mp4
    ├── week2/
    │   ├── Jyothi_MSDS696_Wk2_Status_Report.docx
    │   └── Jyothi_Wk2_Practice_Talk.mp4
    ├── week3/
    │   ├── Jyothi_MSDS696_Wk3_LLM_Describe_And_Wave.docx
    │   ├── Jyothi_MSDS696_Wk3_Status_Report.docx
    │   └── Jyothi_Wk3_Practice_Talk.mp4
    ├── week4/
    │   ├── Jyothi_MSDS696_Wk4_Method_Defense.docx
    │   ├── Jyothi_MSDS696_Wk4_Status_Report.docx
    │   └── Jyothi_Wk4_Practice_Talk.mp4
    ├── week5/
    │   ├── Jyothi_MSDS696_Wk5_Chart_Redesign_Activity.docx
    │   ├── Jyothi_MSDS696_Wk5_Status_Report.docx
    │   └── Jyothi_Wk5_Practice_Talk.mp4
    ├── week6/
    │   ├── Jyothi_MSDS696_Wk6_Status_Report.docx
    │   └── Jyothi_Wk6_Practice_Talk.mp4
    └── week7/
        ├── Jyothi_MSDS696_Wk7_Status_Report.docx
        └── Jyothi_Wk7_Full_Dry_Run.mp4

## 🚀 Limitations & Strategic Next Steps
To safely transition this model into a live banking environment, a **90-Day Shadow Launch** is recommended alongside a roadmap to address current model constraints:

**Limits:**
* **Static Regex:** Vulnerable to changing fraud indicators.
* **Length Bias:** Brief texts may be deprioritized.
* **Sarcasm Blindness:** Misses subtle human nuance.

**Strategic Next Steps:**
* **Dynamic Dictionary:** Update new fraud terms.
* **Agent-Facing SHAP:** Highlights keywords for reviewers.
* **Closed-Loop Human Feedback:** Overrides capture false positives.