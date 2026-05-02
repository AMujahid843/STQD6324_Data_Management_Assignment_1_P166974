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

## Results & Summary of Findings
| Model                | Accuracy | F1-score |
|---------------------|----------|----------|
| Decision Tree       | **1.0000** | **1.0000** |
| Logistic Regression | **1.0000** | **1.0000** |
| Random Forest       | 0.9583   | 0.9578   |

The results indicate that both the Decision Tree and Logistic Regression models achieved perfect performance, with an accuracy and F1-score of 1.000. In contrast, the Random Forest model achieved slightly lower performance, with an accuracy of 0.9583 and an F1-score of 0.9578.

This outcome can be attributed to several factors related to the characteristics of the Iris dataset. Firstly, the dataset is relatively simple and well-structured, with clear separation between classes—particularly for the Setosa species, which is linearly separable from the others. As a result, even relatively simple models such as Decision Trees and Logistic Regression are capable of perfectly classifying the data.

Secondly, the dataset is small, consisting of only 150 observations. With an 80/20 train-test split, the test set contains a limited number of samples. This increases the likelihood of a favourable split, where the selected test instances are easier to classify, allowing some models to achieve perfect scores.

The slightly lower performance of the Random Forest model can be explained by its ensemble nature. Random Forest introduces randomness through bootstrap sampling and feature selection, which helps improve generalisation but can reduce performance slightly on small datasets. Unlike single models, it does not attempt to perfectly fit the training data, making its predictions more conservative and often more realistic.

In addition, feature importance analysis revealed that petal length and petal width are the most significant predictors in determining the iris species. These features contributed most strongly to the model’s decision-making process and played a key role in achieving high classification accuracy.

Therefore, although Decision Tree and Logistic Regression achieved perfect evaluation metrics in this experiment, this does not necessarily indicate superior generalisation ability. The Random Forest model may still be considered more robust in practice due to its ability to handle variability and reduce overfitting.

## Model Strengths & Limitations
The Decision Tree model is highly interpretable and easy to visualise, making it useful for understanding decision-making processes. It also has fast training time. However, it is prone to overfitting and can be sensitive to small variations in the dataset, which reduces its generalisation ability.

The Random Forest model offers high predictive accuracy and reduces overfitting through ensemble learning. It is also capable of capturing complex, non-linear relationships in the data. However, its main limitations are lower interpretability and higher computational cost compared to simpler models.

Logistic Regression is simple and efficient, and it performs well when data is linearly separable. Despite this, it assumes linear relationships between features and the target variable, making it less effective for capturing complex patterns in the dataset.

## Best Model Justification
Although the Decision Tree and Logistic Regression models achieved perfect accuracy and F1-score in this study, the Random Forest model is selected as the most reliable model for practical applications. This is because its ensemble learning approach reduces variance and improves generalisation, making it less prone to overfitting compared to individual models.

The slightly lower performance of Random Forest in this experiment can be attributed to the small size and simplicity of the Iris dataset, as well as the limited test set. Despite not achieving perfect scores, Random Forest provides more stable and realistic predictions due to its use of multiple decision trees and random sampling.

In contrast, the perfect performance of Decision Tree and Logistic Regression may be influenced by the highly separable nature of the dataset and a favourable train-test split, which may not reflect real-world scenarios. Therefore, Random Forest is considered the most robust and reliable model overall, particularly for handling more complex and unseen data.

## Instructions to Reproduce the Analysis
### 1. Clone the Repository
git clone https://github.com/AMujahid843/STQD6324_Data_Management_Assignment_1_P166974.git  
cd STQD6324_Data_Management_Assignment_1_P166974  

### 2. Create and Activate Environment
conda create -n pyspark-env python=3.10  
conda activate pyspark-env  

### 3. Install Required Packages
conda install -c conda-forge pyspark  
pip install pandas matplotlib seaborn jupyter  

### 4. Run the Notebook
jupyter notebook  

Open the `.ipynb` file and run all cells sequentially.

### 5. Dataset
The Iris dataset is automatically downloaded within the notebook using a public URL. No manual download is required.

### 6. Expected Output
After execution, the notebook will produce:
- Model evaluation results (Accuracy and F1-score)
- Comparison of multiple classification models
- Identification of the best-performing model

