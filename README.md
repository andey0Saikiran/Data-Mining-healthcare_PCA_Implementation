# **Chronic Disease Data Mining Analysis (USA Region)**

## **Project Overview**
This project aims to analyze chronic disease data using data mining techniques to uncover meaningful insights. The focus is on understanding geographic, demographic, and health-related factors influencing disease prevalence. We apply **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means clustering** for pattern identification. 

By leveraging data-driven insights, we aim to **develop proactive and personalized healthcare solutions** to mitigate the impact of chronic diseases.

## **Data Sources**
- **Primary Dataset:** [U.S. Chronic Disease Indicators (CDI) Dataset](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi)
- **Population Dataset:** External dataset providing population statistics for different locations.

### **Dataset Characteristics**
| Attribute            | Value       |
|----------------------|------------|
| **Number of Variables** | 34      |
| **Total Rows**        | 1,004,594 |
| **Missing Data**      | 38.1%     |
| **File Size**         | 1.8GB (CSV Format) |
| **Numerical Features**| 6        |
| **Categorical Features** | 12     |
| **Text Features**     | 5        |
| **Unsupported Columns** | 11     |

## **Data Preprocessing and Cleaning**
1. **Dropping Unnecessary Columns:**  
   - Removed 19 columns due to inconsistencies and redundancy.

2. **Handling Missing Data:**  
   - **Mean Imputation**: Missing values in confidence intervals were filled using the mean of respective groups.

3. **Merging Population Data:**  
   - Joined the CDI dataset with population data to **calculate incidence rates** (per 100,000 individuals).

4. **Sampling Data:**  
   - To optimize computational efficiency, **5% of the dataset** was sampled.

5. **Data Standardization:**  
   - String columns were stripped of whitespaces.
   - Non-numeric values were converted to appropriate formats.

## **Key Analysis Techniques**

### **1. Exploratory Data Analysis (EDA)**
- **Visualizations:**
  - **Bar Chart:** Displays the top 5 cities with the highest average incidence rates.
  - **Nested Pie Chart:** Showcases disease distribution across different stratifications.

### **2. Principal Component Analysis (PCA)**
- **Purpose:**  
  - Reduces high-dimensional data to **fewer principal components** while retaining maximum variance.
- **Implementation Steps:**
  1. Data normalization using **StandardScaler**.
  2. PCA applied to identify the **optimal number of components** based on the **cumulative explained variance**.
  3. **Dimensionality Reduction:** The first two principal components are visualized in a scatter plot.

### **3. K-Means Clustering**
- **Objective:**  
  - Identify groups of locations with similar disease incidence patterns.
- **Approach:**
  1. **Elbow Method**: Determines the optimal number of clusters.
  2. **K-Means Algorithm**: Applied to PCA-reduced data to cluster locations.
  3. **Cluster Visualization**: Scatter plot showing clustered locations.

## **Findings and Insights**
1. **Population vs. Disease Incidence:**
   - States with **higher population density (e.g., California)** tend to have **lower cancer incidence rates**.
   - Smaller population areas (e.g., **Guam**) exhibit **higher disease prevalence**.

2. **Cancer Prevalence and Ethnic Influence:**
   - PCA results suggest that **Asian American and Pacific Islander communities** show moderate correlation with cancer prevalence.

3. **Geographic Trends:**
   - Certain chronic diseases are **region-specific**, highlighting the need for **localized public health strategies**.

## **Technologies Used**
- **Programming Language:** Python
- **Libraries:**  
  - `pandas` (Data Manipulation)  
  - `numpy` (Numerical Computation)  
  - `matplotlib`, `seaborn` (Data Visualization)  
  - `sklearn` (Machine Learning)  
  - `dask` (Parallel Processing)

## **How to Run the Code**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repository/chronic-disease-analysis.git
   cd chronic-disease-analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis script:**
   ```bash
   python analysis.py
   ```

## **Conclusion**
This project showcases how **data-driven insights can support public health interventions** by identifying high-risk areas and populations. The combination of **PCA and K-Means clustering** enables a **structured approach to understanding disease prevalence** across different locations.

---
