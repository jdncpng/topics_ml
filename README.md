# Prediction of Overall Rating, Sentiment Analysis, and Topic Modeling of Airline Reviews

This project explores customer sentiment in airline reviews to extract actionable insights using machine learning and NLP techniques. It was developed for the Topics in Machine Learning course.

## Project Overview
Customer reviews play a significant role in influencing airline selection. This project analyses a dataset of airline reviews to:

- Predict overall customer ratings
- Perform sentiment analysis
- Extract major topics from review texts

By understanding key satisfaction drivers, airlines can improve their services and enhance customer experience.

## Dataset
**Source:** Kaggle – Airline Reviews Dataset
**Size:** 2000 reviews

**Features include:**
- Overall rating (1–10)
- Individual service ratings (e.g. seat comfort, food, staff)
- Full textual reviews
- Categorical features like travel class and traveller type

**Preprocessing steps:**
- Column renaming, type conversion, NaN imputation
- Feature merging and column dropping
- Date conversion and ordering by importance

## Methods
📌 Sentiment Analysis
Tool: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Captures negation, polarity, and intensity

📌 Prediction Models
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost

**Evaluation metrics:**
- Mean Squared Error (MSE)
- Adjusted R² Score

📌 Topic Modeling
- LDA (Latent Dirichlet Allocation) – to extract interpretable topics
- RAG (Retrieval-Augmented Generation) – for enriched context and accuracy

📈 Key Findings
Lasso regression yielded the lowest MSE, though R² metrics require refinement.
Strong positive correlations found between overall rating and features like:
- In-flight entertainment
- Seat comfort
- Ground service
- Food & beverages

Sentiment analysis showed a bimodal distribution, indicating highly polarised customer opinions.

**Topics Identified:**
- Booking and boarding experience
- Baggage and delays
- Cabin crew service
- In-flight entertainment

🧰 Technologies Used
Languages: Python
Libraries:
pandas, numpy, scikit-learn, matplotlib, seaborn, nltk,
transformers, huggingface_hub, torch, pytorch_lightning,
sentence_transformers, langchain, openai

📌 Conclusion
This project highlights the importance of combining machine learning with NLP to gain deeper insights into customer satisfaction. Although predictive models need refinement, the topic and sentiment analyses provide a solid foundation for targeted improvements in the airline industry.
