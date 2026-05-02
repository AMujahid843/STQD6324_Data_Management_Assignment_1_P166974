# STQD6324 DATA MANAGEMENT ASSIGNMENT 1 (P166974)
## Overview of the Project
This project demonstrates a complete machine learning pipeline using PySpark to classify iris flower species based on their physical measurements. It implements a full end-to-end workflow using PySpark MLlib, focusing on classification of iris species based on their morphological features.

The workflow follows a structured data science approach, which includes data preprocessing, feature engineering, model development, optimization, and performance analysis. It also involves evaluating multiple classification algorithms to compare their effectiveness.

The main objective of this project is to build a scalable machine learning solution using Apache Spark and identify the best-performing model for the Iris dataset classification task. Through systematic evaluation and hyperparameter tuning, the project aims to determine the most effective model for accurate species prediction.
## Dataset Description
The project uses the classic Iris dataset, which is a well-known benchmark dataset widely applied in classification tasks. It contains measurements of iris flowers from three different species: **Setosa**, **Versicolor**, and **Virginica**.

The dataset consists of a total of **150 samples**, where each sample represents an individual iris flower. It includes four numerical features that describe the physical characteristics of the flowers, namely **sepal length (cm)**, **sepal width (cm)**, **petal length (cm)** and **petal width (cm)**.

The target variable in this dataset is the species, which serves as the classification label indicating which of the three iris species each observation belongs to. These three target classes are Setosa, Versicolor, and Virginica.

The main objective of this project is to build a machine learning model that can accurately predict the species of an iris flower based on the four input features. Due to its simplicity, well-structured format, and balanced class distribution, this dataset is widely used in machine learning as a standard benchmark for evaluating classification algorithms.

## Methodology

The first step in the methodology is **data preprocessing**, where the dataset is prepared for use with Spark MLlib. This includes converting categorical labels into a numeric format using StringIndexer, as well as combining all feature columns into a single feature vector using VectorAssembler. This ensures that the data is properly structured and compatible with Spark’s machine learning pipeline.

Next, the dataset is **split into training and testing sets**, with 80% of the data used for training and 20% reserved for testing. A random seed is applied to ensure that the results are reproducible. This split allows for unbiased evaluation of the model on unseen data.

In the **model development stage**, three different classification algorithms are implemented: **Decision Tree**, **Random Forest**, and **Logistic Regression**. The Decision Tree model splits data based on feature thresholds, while Random Forest improves performance by combining multiple decision trees to enhance generalisation. Logistic Regression is used as a linear model adapted for multiclass classification. These models represent different learning approaches, allowing for a meaningful performance comparison.

**Hyperparameter tuning** is performed next to optimise model performance. Grid Search is used to test multiple combinations of parameters, and 5-fold cross-validation is applied to ensure model robustness. The parameters tuned include "maxDepth" and "minInstancesPerNode" for Decision Tree, "numTrees" and "maxDepth" for Random Forest, and "regParam" and "elasticNetParam" for Logistic Regression. This step helps to reduce overfitting and improves the generalisation ability of the models.

Finally, the **performance of the classification models was evaluated using standard evaluation metrics** to ensure a comprehensive assessment of their predictive ability. These metrics provide different perspectives on how well each model performs in correctly classifying Iris species based on the input features.

**Accuracy** was used as the primary evaluation metric and measures the overall proportion of correctly predicted instances out of all predictions made. It provides a general indication of model performance; however, it may not always fully reflect performance in cases where class distribution is imbalanced.

In addition to accuracy, the **F1-score** was also used as an important evaluation metric. The F1-score is the harmonic mean of precision and recall, and it provides a balanced measure of model performance. It is particularly useful when there is a need to balance different types of classification errors, offering a more comprehensive view of performance than accuracy alone.

Overall, while accuracy provides a general measure of correctness, the F1-score offers a more reliable and balanced evaluation for classification tasks. This makes it a more informative metric when comparing the performance of different machine learning models. For this study, both accuracy and F1-score are used to evaluate and compare model performance.


📈 Results
Model	Accuracy	F1 Score
Decision Tree	0.93	0.92
Random Forest	0.97	0.96
Logistic Regression	0.95	0.94
________________________________________
🔍 Discussion
🔸 Performance Analysis
•	Random Forest achieved the highest performance across all metrics
•	Logistic Regression performed competitively but was limited by its linear assumptions
•	Decision Tree showed lower performance due to overfitting tendencies
________________________________________
🔸 Model Strengths & Limitations
🌳 Decision Tree
Strengths:
•	Interpretable and easy to visualise
•	Fast training time
Limitations:
•	Prone to overfitting
•	Sensitive to small data variations
________________________________________
🌲 Random Forest
Strengths:
•	High predictive accuracy
•	Reduces overfitting through ensemble learning
•	Captures complex, non-linear relationships
Limitations:
•	Less interpretable
•	Higher computational cost
________________________________________
📈 Logistic Regression
Strengths:
•	Simple and efficient
•	Works well with linearly separable data
Limitations:
•	Assumes linear relationships
•	Less effective for complex patterns
________________________________________
🏆 Best Model Justification
The Random Forest model is selected as the best-performing model due to its superior accuracy and F1-score.
Its ensemble approach reduces variance and enhances generalisation, making it more robust compared to individual models. Additionally, it effectively captures non-linear relationships present in the dataset.
________________________________________
▶️ Reproducibility Guide
🔹 1. Clone Repository
git clone <your-repo-link>
cd iris-spark-classification
🔹 2. Create Environment (Anaconda Recommended)
conda create -n pyspark_env python=3.10
conda activate pyspark_env
🔹 3. Install Dependencies
conda install -c conda-forge pyspark
conda install notebook
pip install findspark
🔹 4. Launch Notebook
jupyter notebook
🔹 5. Run Analysis
Open:
notebook/iris_classification.ipynb
Then run all cells.
________________________________________
📂 Project Structure
iris-spark-classification/
│
├── notebook/
│   └── iris_classification.ipynb
├── data/
│   └── iris.csv
├── README.md
└── requirements.txt
________________________________________
📌 Key Takeaways
•	Proper preprocessing is essential for Spark ML pipelines
•	Hyperparameter tuning significantly improves performance
•	Ensemble models (Random Forest) outperform single models
•	Evaluation metrics beyond accuracy provide deeper insights
________________________________________
📚 References
•	Apache Spark MLlib Documentation
•	UCI Machine Learning Repository – Iris Dataset
________________________________________
👨‍💻 Author
Student Name: [Your Name]
Course: [Your Course Name]
Institution: [Your University]
________________________________________
