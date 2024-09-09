# A Multi-Task Learning News Analysis Model using Sentence Transformer

## Overview

This project focuses on building a multi-task learning model to analyze women's clothing reviews using the DistilBERT transformer. The goal is to predict both the product rating as a classification task and the user's age as a regression task based on textual reviews. The model is fine-tuned with a classification head for predicting ratings and a regression head for predicting age. We leverage the Hugging Face library, PyTorch, and Python for model development and data processing, and Flask for deployment.

## Introduction

The [Women's Clothing Review Analysis](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) project aims to develop a sophisticated model capable of extracting meaningful insights from clothing reviews. By employing a multi-task learning approach, we address two related tasks simultaneously:
1. **Classification:** Predicting the product rating from the review text.
2. **Regression:** Predicting the user's age from the review text.


## Technical Tools Used

- **Data Processing:** Pandas for data cleaning and manipulation.
- **Modeling Framework:** Hugging Face Transformers and PyTorch for developing and training the multi-task learning model.
- **Model Architecture:** DistilBERT transformer with additional classification and regression heads.
- **Deployment:** Flask for creating a web application to deploy the model and provide an interface for predictions.

## Multi-Task Learning (MTL)

Multi-task learning (MTL) is a machine learning approach where a single model is trained on multiple tasks simultaneously. This method leverages shared representations and features to improve performance across tasks, particularly when they are related. 

### Key Concepts

- **Shared Representation:** DistilBERT generates embeddings from the text that serve as input for both regression and classification tasks.
- **Regression Head:** A linear layer predicting continuous values, such as user age.
- **Classification Head:** A linear layer predicting discrete labels, such as product ratings.

### Benefits

- **Improved Generalization:** Sharing representations between tasks helps the model generalize better and avoid overfitting.
- **Efficient Use of Data:** Related tasks can use shared representations to leverage data more effectively, especially when labeled data is scarce.
- **Better Performance:** Multi-task learning often enhances performance on individual tasks compared to training separate models for each.




### Installation

Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```


### Running the Application

To run the application, execute the `main.py` file using Python. This command can be executed in your terminal or command prompt:

```bash
python main.py
```

Running this command will start the application and execute the main script, allowing you to interact with the functionality provided by the program.


Alternatively, if you prefer to run the application using the Flask web framework, you can use the following command:
```bash
python app.py
```
