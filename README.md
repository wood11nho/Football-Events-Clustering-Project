# Football Events Clustering Project

## 📌 Overview

This project explores **unsupervised learning techniques** to uncover patterns in football data. The primary focus is on analyzing play styles of football teams and shot placement trends using **clustering algorithms**.

## 📊 Objectives

- Cluster football teams based on their **play styles**.
- Analyze and identify patterns in **shot placement**.
- Compare **clustering methods** (BIRCH & OPTICS) with **supervised baselines** and **random chance**.

## 📂 Dataset

The dataset comes from Kaggle: [Football Events Dataset](https://www.kaggle.com/datasets/secareanualin/football-events), containing **900,000+ events** from **9,074 matches** across Europe's top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).

### **Main Files:**

- `events.csv`: Detailed event-level data (shots, fouls, passes, etc.).
- `ginf.csv`: Metadata including match stats and betting odds.
- `dictionary.txt`: Mappings for categorical variables.

## 🛠 Methods & Techniques

### **1️⃣ BIRCH Clustering - Football Team Play Styles**

- **Feature Engineering:** Extracted meaningful team-level statistics.
- **Clustering Process:**
  - One-hot encoding categorical variables.
  - Aggregating event-level data at the team level.
  - Normalizing features and removing low-variance ones.
  - **BIRCH** applied with hyperparameter tuning (optimal threshold: `0.1`).
- **Results:**
  - Found **5 main clusters**, partially aligned with league divisions.
  - Accuracy vs. ground truth league labels: **43%**.
  - Some teams (e.g., Leicester City, Burnley) exhibited unique styles outside their league norm.

### **2️⃣ OPTICS Clustering - Shot Placement Analysis**

- **Feature Engineering:** Contextualized shot events using:
  - Event location, shot type, assist method, and textual commentary (TF-IDF).
  - Derived features like time since last event.
- **Clustering Process:**
  - OPTICS hyperparameter tuning (best: `min_samples=5`, `xi=0.05`, `metric=cosine`).
  - Used **reachability plots** and PCA visualization.
- **Results:**
  - Identified **204 clusters**, mapped to **13 shot placement categories**.
  - Clustering accuracy for shot placement: **62.16%**.
  - Some misclassification for central goal areas, but strong patterns in top/bottom corner shots.

### **3️⃣ DBSCAN Clustering - Additional Shot Placement Analysis**

- **Alternative Approach:** Tested **DBSCAN** to compare performance.
- **Results:**
  - Found **161 clusters**, with an accuracy of **54.61%**.
  - Slightly worse separation than OPTICS, especially in central goal regions.

## 📈 Evaluation & Benchmarking

| Method       | Accuracy | Silhouette Score | Key Takeaways |
|-------------|----------|----------------|--------------|
| **BIRCH (Play Styles)** | 43% | 0.18 | Clusters captured league-wide trends but some overlap in styles. |
| **OPTICS (Shot Placement)** | 62.16% | 0.35 | Well-defined clusters, good alignment with shot placement labels. |
| **DBSCAN (Shot Placement)** | 54.61% | 0.26 | Less separation than OPTICS, some issues in central goal shots. |
| **Supervised (Random Forest)** | 91.3% | - | Clearly outperforms unsupervised models, showing the importance of labels. |
| **Random Chance** | 21% | - | Baseline performance, proves clustering methods add value. |

## 📌 Key Insights

- **Football team styles** don't always align with leagues, as some teams adopt cross-league tactics.
- **Shot placement clustering** works well but struggles with ambiguous middle-goal shots.
- **Supervised learning (Random Forest) still dominates**, proving that labeled data is crucial for precise predictions.

## 📜 Conclusion

This project highlights the power of **unsupervised learning** in football analytics. While clustering provides useful insights, **supervised models significantly outperform** in accuracy. Future work could involve **hybrid models**, integrating clustering with semi-supervised learning for better performance.

## 📜 Acknowledgments

- Dataset: [Kaggle - Football Events](https://www.kaggle.com/datasets/secareanualin/football-events)
- Faculty of Mathematics and Computer Science, University of Bucharest

## 🏆 Author

**Elias-Valeriu Stoica** - [GitHub](https://github.com/wood11nho)