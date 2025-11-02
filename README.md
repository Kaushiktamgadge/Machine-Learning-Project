# **Crime Early Warning System — Prediction of High-Risk Districts in India**

---

## **Project Overview**

Crime Early Warning System is a predictive analytics system, which is developed based on machine learning and which can predict high-risk districts of crimes against Scheduled Tribes (STs) in India.

In this system, crime statistics in the historical district are examined using the National Crime Records Bureau (NCRB) data so as to predict the areas that have a high likelihood of experiencing high crime rates in the next year.

This is aimed at delivering an early warning and decision-support system to law enforcement, which will support data-driven prevention measures, effective resources distribution, and community safety.

---

## **Problem Statement**

Scheduled Tribes crimes are a continuing social issue in India, which is usually focused in some districts, because of the entrenched socio-economic determinants.  
The conventional crime tracking systems are reactive in nature — they document crime but seldom anticipate or avert it.

The following were the main issues addressed in this project:

- Insufficient foresight on existing crime data systems.  
- Failure to detect high-risk districts in time.  
- Wasting resources in planning law enforcement.  

The system estimates the level of risk in the future by utilizing the trends in criminal activities in the past turning reactive policing to proactive intervention.

---

## **Project Importance**

| **Impact Area** | **Benefit** |
|------------------|-------------|
| **Public Safety** | Anticipate and thwart crimes. |
| **Data-Driven Policing** | Give precedence to high-risk districts. |
| **Resource Optimization** | Improved resource use of law enforcement. |
| **Policy Planning** | Determine crime trends in the long term to intervene. |
| **Transparency & Accountability** | Government decisions are made based on data. |

---

## **Dataset Information**

### **Dataset Source**
It was found that the proportion of crimes recorded by the National Crime Records Bureau (NCRB), Government of India, is low.

- **File Used:** districtwise-crime-against-sts-2017-onwards.csv  
- **Data Coverage:** 2017–2023  
- **Granularity:** Annual data of the District.

### **Dataset Characteristics**

| **Attribute** | **Description** |
|----------------|-----------------|
| **state_name** | The Name of the state |
| **district_name** | The name of the districts. |
| **year** | The Year of record |
| **total_crimes** | Total recorded crimes against STs |
| **label_high_risk** | Target variable (1 = high risk, 0 = low risk) |

---

## **Data Preprocessing Steps**

- **Data Cleaning:** Deleted the missing or duplicate district records.  
- **Feature Engineering:** The lag features (lag1, lag2) were developed in order to reflect the temporal trends.  
- **Encoding:** Applied label encoding to state and district names.  
- **Normalization:** Normalized numeric data with the StandardScaler.  
- **Feature Selection:** 40 most relevant features were selected by means of feature importance ranking.  
- **Splitting:** 80 percent training, 20 percent testing where stratified sampling was used to maintain classes.  

---

## **Methodology**

### **Approach Overview**

The given project is structured around a machine learning pipeline, which resembles the standard procedure of working on an analytical and predictive modeling. The idea was to construct a dependable system which would predict high-risk districts and at the same time be easy to interpret and deploy.

**Pipeline Overview:**  
`Raw Data → Data Cleaning → Feature Engineering → Model Training → Evaluation → Streamlit Deployment`

---

### **Why This Approach?**

**Temporal Trends:**  
The tendency to commit crimes in the past is usually a good predictor of crimes in future. The model effectively captures these temporal dependencies by the introduction of lag features (e.g. the count of crimes in the last one or two years).

**Ensemble Learning:**  
Since as an ensemble of decision trees, the Random Forests can reduce overfitting and non-linear association between the characteristics and are therefore best suited to real world and noisy data like crime records.

**Explainability:**  
The feature importance metrics in the model offer transparency which is key to accountability in governance and law enforcement since it enables the stakeholders to know the variables used to make the prediction.

---

### **Models Evaluated**

In order to attain a trade off between accuracy and interpretability, multiple machine learning algorithms were experimented and contrasted.

| **Model** | **Type** | **Key Characteristics** |
|------------|-----------|--------------------------|
| **Logistic Regression** | Linear | Simple, quick and interpretable baseline model. |
| **Random Forest** | Ensemble | High accuracy and resistance to noise; can be interpreted by importance of features. |
| **XGBoost** | Boosting | XGBoost Boosting Structured data gradient-based boosting algorithm. |
| **SVM (Support Vector Machine)** | Kernel-based | Models more complicated non-linear relationships at the expense of increased computing power. |

---

## **Final Model**

The Random Forest Classifier turned out to be the best performer after experiments, because it is able to provide consistent results in terms of accuracy, recall, and ROC-AUC, and can also cope with imbalanced data.

**Model Configuration:**
n_estimators = 300
class_weight = 'balanced'
random_state = 42

**Training Strategy:**
- The model was tested on unseen data on a 5-fold cross-validation.  
- To maximize both ROC-AUC and Recall as the primary goal was to identifiably detect high-risk districts and not achieve the highest accuracy of the system, the hyperparameters were optimized.

---

## **Experiments and Results**

### **Model Performance**

| **Metric** | **Value** |
|-------------|-----------|
| **Accuracy** | 0.86 |
| **Precision** | 0.83 |
| **Recall** | 0.80 |
| **F1-Score** | 0.81 |
| **ROC-AUC** | 0.89 |

These metrics suggest that this model is capable of distinguishing between high-risk and low-risk districts, which offers a good balance between false positives and false negatives.

---

### **Model Comparison**

| **Model** | **Accuracy** | **ROC-AUC** | **Remarks** |
|------------|--------------|--------------|--------------|
| **Logistic Regression** | 0.78 | 0.80 | Simple base but not very in-depth. |
| **Random Forest** | 0.86 | 0.89 | Good trade-off between interpretability and accuracy. |
| **XGBoost** | 0.84 | 0.88 | Similar results with an increased training time. |
| **SVM** | 0.82 | 0.84 | Good but expensive to compute. |

Random Forest was selected to deploy as it continuously showed higher performance compared to other models and it can be explained, which is one of the conditions of the implementation to the public sector.

---

### **Feature Importance**

| **Feature** | **Relative Importance (%)** |
|--------------|:---------------------------:|
| **Crime count (lag1)** | 41 |
| **Crime count (lag2)** | 25 |
| **Total crimes reported** | 18 |
| **Encoded state** | 9 |
| **Year** | 7 |

**Interpretation:**  
According to the model, the patterns of crime are quite self-predictable, high-crime districts in the last two years are likely to remain at risk. This is true to the reality on the ground whereby, crime trends tend to be perpetuated in systemic social and economic realities.

---

## **Visualization Insights**

**Crime Trend Graphs:**  
Cyclical or upward trends are evident in many districts over a sequence of years which justifies the lag features.

**Feature Importance Plot:**  
The bar chart that is used to show the features gives more weight to, showing which features affect most of the model, making them more explainable.

**ROC Curve:**  
The model is shown to be very strong with a high ability to separate classes, as the ROC curve is found to have an AUC of about 0.89.

**Future Scope (Geographical Heatmaps):**  
Results can be displayed on an interactive map with minor extensions to find the high-risk areas around India to support the field deployment and communicate with the stakeholders.

---

## **Implementation and Setup**

### **System Requirements**
- Python 3.9 or higher  
- pip package manager  
- Model training: Jupyter Notebook.  
- Streamlit (to work with a dashboard interface)  

### **Required Libraries**
Install the dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib


---

### **Project Structure**

Crime-Early-Warning/
│
├── app.py # Streamlit frontend application
├── model.ipynb # Model training and analysis notebook
├── rf_high_risk_model.joblib # Trained Random Forest model
├── scaler.joblib # Data scaler
├── le_state.joblib # State label encoder
├── le_district.joblib # District label encoder
├── model_features.joblib # Saved list of features used in training
├── model_target.joblib # Target variable reference
├── requirements.txt # Dependency file
└── README.md # Project documentation

---

### **How to Run**

**Step 1 — Train the Model:**  
Open the Jupyter Notebook:
Notebook model.ipynb


**Step 2 — Launch the App:**  
Open and launch the Streamlit interface:
streamlit run app.py

**OR**
python -m streamlit run app.py


**Step 3 — Use the App:**  
- Single predictions can be made for a state, district, and a year.  
- Batch predictions are made by uploading a CSV file.  
- Real-time risk probability and label and feature importance charts.  

---

## **Experiments Summary**

- The temporal correlation is a dominant factor in forecasting future risk of crime.  
- Random Forest was found to be the best with imbalanced data of mixed types.  
- Class imbalance issue was also addressed well by `class_weight='balanced'`.  
- The unseen 2023 data was used to test the model, thereby making it robust and forward-generalized.  

---

## **Conclusions**

This project illustrates that past data can be used to forecast the risk of a crime with the help of the existing machine learning methods.

**Major Implication:** Crime in some districts is periodically persistent - regions that were high-risk in the past are likely to be high-risk areas.

**Impact:** The tool has great predictive value that can be used to guide resource mobilization and community safety initiatives.

**Accessibility:** With the help of the Streamlit app, the non-technical users will be able to visualize predictions, download the reports, and interpret model insights.

Such a structure preconditions data-driven policing and evidence-based policymaking, which is a transition towards proactive public safety management instead of reactive management.

---

## **Future Work**

- Live monitoring should be integrated with real-time NCRB or state police data.  
- Add geospatial visualization, e.g. Folium or Plotly.  
- Test LSTM or Prophet models of the sequence-based forecasting of the time dimension.  
- Use on Hugging Face or Streamlit Cloud to make it accessible to more people.  
- Add explainability modules (SHAP, LIME) to make the policymakers understand why a region is forecasted to be high-risk.  

---

## **References**

- National Crime Records Bureau (NCRB). District Crime Against Scheduled Tribes (2017).  
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.  
- Pedregosa et al. (2011). Scikit-learn: Python Machine Learning. JMLR, 12, 2825–2830.  
- Streamlit Documentation — [https://docs.streamlit.io](https://docs.streamlit.io)  
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.  
- Indian Government, Home Affairs. Crime in India Reports (2017–2023).  

---

## **Author & Acknowledgment**

**Developed by:** Kaushik Tamgadge  
**Institution/Team:** Symbiosis Institute of Technology, Nagpur  
**Purpose:** Academic and research-based project on predictive policing with open crime data.  

**Acknowledgments:**
- The datasets which are publicly accessible are from NCRB India.  
- The open-source tools developer community is Scikit-learn and Streamlit.  
- Faculty mentors and continuous feedback and guidance reviewers.  
