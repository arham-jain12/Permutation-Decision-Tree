# Permutation Decision Trees and Forests

This repository contains the implementation and performance evaluation of various tree-based machine learning models, including Decision Trees (DT), Random Forests (RF), and Permutation Decision Forests (PDF). Additionally, we used a novel impurity measure, Effort-To-Compress (ETC), and empirically evaluate its performance compared to traditional impurity measures like Shannon Entropy and Gini Index.

---

## 📂 **Project Structure**

```
├── load.py                                            # Script for loading all datasets
├── classical_decision_tree_using_entropy.ipynb        # DT using Shannon Entropy
├── classical_decision_tree_using_gini.ipynb           # DT using Gini Index
├── random_forest_entropy.ipynb                        # RF using Shannon Entropy
├── random_forest_gini.ipynb                           # RF using Gini Index
├── permutated_decision_tree_and_forest.ipynb          # PDF and PDT Implemenataion using PDT
├── barplots.ipynb                                     # Script for generating barplots for results
└── README.md                                          # Project documentation
```

---

## 📊 **Datasets Used**

The following datasets were analyzed:

- **Appendicitis**
- **Breast Cancer Wisconsin**
- **Diabetes Pima Indian**
- **Ionosphere**
- **Iris**
- **Sonar**
- **Wine**

Each dataset was preprocessed and split into training and testing sets using a random seed (42) and time series split (validation size = 15).

---

## ⚙️ **Hyperparameter Tuning**

The following parameters were optimized:

### Decision Trees (DT)
- **Depth**: 2 to 20  
- **Impurity Measures**: Shannon Entropy and Gini Index

### Random Forests (RF)
- **n_estimators**: [10, 50, 100, 150, 1000]  
- **Depth**: 2 to 20  
- **Impurity Measures**: Shannon Entropy and Gini Index  

### Permutation Decision Tree (PDT) and Permutation Decision Forest (PDF)

1. **Permutation of Data**:  
   - **21 unique seed values** for datasets: Ionosphere, Iris, Wine, Diabetes, Appendicitis  
   - **100 and 200 unique seed values** for datasets: Breast Cancer Wisconsin and Sonar  
   
2. **Depth Tuning**:  
   - PDT depths range from **2 to 20**.  
   - For each permutation, the ideal **depth** that produces the highest **macro F1-score** is determined using **time series split** (five splits, validation size = 15).  

3. **Testing Phase**:  
   - PDT models are retrained on the shuffled data with optimal depth and seed.  
   - For **PDF**, the test data is passed through multiple PDTs, and a **voting scheme** is employed to predict the final label.
   
---

## 🧪 **Results**

The models were evaluated based on the following metrics:

- **Accuracy**
- **Macro F1-Score**
- **Precision**
- **Recall**

### Key Findings:
1. **Permutation Decision Tree (PDT)** outperformed classical Decision Trees on average.  
2. **Permutation Decision Forest (PDF)** achieved performance close to **Random Forests** with significantly fewer estimators:  
   - **PDF**: 21–200 estimators  
   - **RF**: 50–1000 estimators  
3. ETC impurity measure behaved comparably to Shannon Entropy in empirical evaluations.

---

## 📈 **Results Visualization**

Results are visualized using **barplots** to compare **Macro F1-Scores** across all models and datasets. Execute `barplots.ipynb` to generate the plots.

---

## 🚀 **How to Run the Code**

### Prerequisites
Ensure you have the required libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/arham-jain12/Permutation-Decision-Tree.git
   cd Permutation-Decision-Tree
   ```

2. Load the datasets by running `load.py`:
   ```bash
   python load.py
   ```

3. Run the specific Jupyter Notebooks for model implementation:
   - **Decision Trees (DT):**
     - `classical_decision_tree_using_entropy.ipynb`
     - `classical_decision_tree_using_gini.ipynb`
   - **Random Forests (RF):**
     - `random_forest_entropy.ipynb`
     - `random_forest_gini.ipynb`
   - **Permutation Decision Forests:**
     - `permutated_decision_tree_and_forest.ipynb`

4. Generate the barplots using `barplots.ipynb`.

---

## 🔬 **Future Work**

1. Investigate the relationship between **ETC** and **Kolmogorov complexity**.  
2. Test PDT and PDF on **non-iid datasets** to evaluate their robustness further.

---

## 👤 **Contributions**

Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to **open an issue** or submit a **pull request**.

---
