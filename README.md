# \# Customer Churn Prediction \& Retention System

# 

# An end-to-end machine learning project that predicts whether a customer is likely to churn based on customer demographics, account information, service usage, and support interactions.

## Live Demo

🚀 **Try the deployed application:**  
https://customer-churn-prediction-asraiyani.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/ASRAIYANI/customer-churn-prediction

# 

# The project covers the complete machine learning workflow from data exploration and preprocessing to model comparison, evaluation, and deployment through an interactive Streamlit application.

# 

# \## Project Overview

# 

# Customer churn prediction helps businesses identify customers who may stop using their services.

# 

# This project builds a binary classification model where:

# 

# \- `0` = Customer does not churn

# \- `1` = Customer churns

# 

# The final model is integrated into a Streamlit web application where users can enter customer information and receive a churn prediction.

# 

# \## Dataset

# 

# The project uses a synthetic customer churn dataset containing 5,000 customer records.

# 

# Features include:

# 

# \- Age

# \- Tenure in months

# \- Monthly charges

# \- Contract type

# \- Internet service

# \- Tech support

# \- Online security

# \- Payment method

# \- Senior citizen status

# \- Number of support calls

# 

# `customer\_id` is excluded from model training.

# 

# The dataset contains missing values, which are handled as part of the preprocessing pipeline.

# 

# > \*\*Important:\*\* The dataset is synthetic and intended for educational and portfolio purposes. Results should not be interpreted as evidence of real-world business performance.

# 

# \## Machine Learning Workflow

# 

# The project includes:

# 

# 1\. Exploratory data analysis

# 2\. Missing-value handling

# 3\. Numerical feature preprocessing

# 4\. Categorical feature encoding

# 5\. Train/validation/test splitting

# 6\. Logistic Regression training

# 7\. Random Forest training

# 8\. Model comparison using validation performance

# 9\. Final evaluation on the held-out test set

# 10\. Model serialization with Joblib

# 11\. Interactive prediction using Streamlit

# 

# The data is split into:

# 

# \- 60% training

# \- 20% validation

# \- 20% testing

# 

# The random state is set to `42` for reproducibility.

# 

# \## Model Selection

# 

# Two classification algorithms were compared:

# 

# \- Logistic Regression

# \- Random Forest

# 

# The model was selected using validation F1 score.

# 

# \*\*Selected model: Random Forest\*\*

# 

# \## Final Test Results

# 

# The selected Random Forest model achieved:

# 

# | Metric | Score |

# | --- | ---: |

# | Accuracy | 0.7980 |

# | Precision | 0.9213 |

# | Recall | 0.8383 |

# | F1 Score | 0.8779 |

# | ROC-AUC | 0.7511 |

# 

# These metrics are calculated on the held-out test set.

# 

# Because the synthetic dataset has a high proportion of churn cases, the headline metrics should be interpreted alongside class-level performance rather than treated as representative of a typical real-world churn system.

# 

# \## Streamlit Application

# 

# The project includes an interactive Streamlit application that accepts customer information and generates a churn prediction using the trained Random Forest pipeline.

# 

# Run the application with:

# 

# ```bash

# streamlit run app/streamlit\_app.py

# ```

# 

# \## Project Structure

# 

# ```text

# customer\_churn\_prediction\_project/

# ├── app/

# │   ├── app.py

# │   └── streamlit\_app.py

# ├── data/

# │   └── customer\_churn.csv

# ├── models/

# │   ├── churn\_model.joblib

# │   └── model\_metadata.json

# ├── reports/

# │   ├── model\_comparison.csv

# │   ├── model\_metadata.json

# │   ├── test\_metrics.json

# │   └── validation\_model\_comparison.csv

# ├── src/

# │   ├── explore\_data.py

# │   └── train.py

# ├── .gitignore

# ├── README.md

# ├── requirements.txt

# └── TRAINING\_OUTPUT.txt

# ```

# 

# \## Installation

# 

# Clone the repository and move into the project directory:

# 

# ```bash

# git clone <repository-url>

# cd customer\_churn\_prediction\_project

# ```

# 

# Create a virtual environment:

# 

# ```bash

# python -m venv .venv

# ```

# 

# Activate it on Windows:

# 

# ```bash

# .venv\\Scripts\\activate

# ```

# 

# On macOS/Linux:

# 

# ```bash

# source .venv/bin/activate

# ```

# 

# Install the dependencies:

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \## Train the Model

# 

# Run:

# 

# ```bash

# python src/train.py

# ```

# 

# The training script preprocesses the data, compares candidate models, selects the model using validation F1 score, evaluates the selected model on the test set, and saves the trained pipeline and evaluation results.

# 

# \## Run the Web Application

# 

# Run:

# 

# ```bash

# streamlit run app/streamlit\_app.py

# ```

# 

# Open the local Streamlit URL displayed in the terminal.

# 

# \## Technologies Used

# 

# \- Python

# \- Pandas

# \- NumPy

# \- Scikit-learn

# \- Joblib

# \- Streamlit

# \- Matplotlib

# \- Seaborn

# \- Git

# \- GitHub

# 

# \## Limitations

# 

# This is an educational portfolio project built with synthetic data.

# 

# The dataset has an unusually high churn rate, so the model's results should not be interpreted as representative of a production customer-retention system.

# 

# In a real-world deployment, additional work would be required, including representative production data, data-drift monitoring, threshold optimization, fairness analysis, explainability, model monitoring, and periodic retraining.

# 

# \## Future Improvements

# 

# Potential improvements include:

# 

# \- Hyperparameter tuning

# \- Additional classification algorithms

# \- Feature engineering

# \- Decision-threshold optimization

# \- SHAP-based model explainability

# \- Improved class-imbalance analysis

# \- Model monitoring

# \- Integration with a customer retention workflow

