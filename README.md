
# Multivariate Analysis of Banana Flour Physicochemical Properties Using Principal Component Analysis (PCA)

## Project Overview
This project investigates the physicochemical properties of banana flour using **Principal Component Analysis (PCA)** as a dimensionality reduction technique. It aims to classify flour derived from four distinct banana groups: *green pulp, green peel, ripe pulp*, and *ripe peel*, based on chemical and physical components.

Using Python-based data science tools, the study reduces feature complexity, visualizes hidden patterns in high-dimensional space, and improves model interpretability for classification tasks.

---

## 📊 Objectives
- Reduce dimensionality of the banana flour dataset using PCA
- Visualize data clusters via PCA biplots
- Project new (synthetic) samples onto PCA space for prediction
- Train a Support Vector Classifier (SVC) using both raw and PCA-transformed data
- Compare classification accuracy with and without PCA

---

## 🛠️ Tools & Libraries
- Python 3.10+
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 🧬 Dataset
The dataset contains:
- **13 numerical features** describing chemical and physical properties (e.g., moisture, ash, protein, fat content, etc.)
- **1 categorical variable** indicating the flour group label

---

## 🔁 Methodology
1. **Data Preprocessing**
   - Standardization using `StandardScaler`
   - Encoding of the group variable

2. **Exploratory Analysis**
   - Variance plots to assess feature scaling needs
   - Heatmap correlation matrix

3. **PCA Execution**
   - Performed PCA with scikit-learn
   - Explained variance plotted to choose optimal components
   - Biplot created to visualize group clustering

4. **New Sample Projection**
   - Synthetic data point projected into PCA space to simulate classification

5. **Classification Modeling**
   - Support Vector Classifier (SVC) trained on both raw and PCA-transformed data
   - Oversampling used to balance the classes

---

## 📌 Key Findings
- PCA effectively revealed structure in the flour dataset.
- Projected samples could be visually and statistically grouped.
- Even with slight accuracy reduction, PCA-enhanced interpretability justifies its use in exploratory data analysis and modeling.
- PCA provided interpretability and dimensionality reduction with minimal model performance trade-off.

---
