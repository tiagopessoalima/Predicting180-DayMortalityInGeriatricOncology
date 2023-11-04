# Predicting 180-Day Mortality In Geriatric Oncology

Background: The healthcare landscape faces a significant challenge in providing care for geriatric oncology patients due to the increasing prevalence of cancer among the aging population. Comprehensive Geriatric Assessment (CGA) has emerged as a holistic approach to address the unique needs of geriatric patients, encompassing medical, functional, psychological, and social dimensions. Integrating Baseline Laboratory Tests (BLT) and clinical data into CGA can enhance prognosis, treatment decisions, and overall understanding of patients' health. Machine Learning (ML) brings data-driven capabilities to this context, improving risk stratification and treatment planning for geriatric oncology patients. Objective: The primary goal is to integrate BLT and clinical data into CGA to accurately predict 180-day mortality in geriatric oncology patients, aiming to optimize care and decision-making for this population. Methods: The research involves data preparation, feature selection, model development, and evaluation. Data includes 2,748 elderly patients, and a temporal window analysis confines outcomes within a 180-day window. Features encompass CGA parameters validated in previous research, BLT parameters, and cancer site data. Statistical tests are applied for feature selection. Models span various ML approaches with hyperparameter optimization and focus on data normalization, imputation, and balancing. Evaluation metrics include accuracy, precision, recall, geometric mean, F1-score, and ROC AUC. Results: Feature selection reveals the significance of CGA parameters, hematological parameters (hemoglobin, leukocyte count, and platelet count), and cancer site in predicting mortality. SVM stands out as the model with a sensitivity of 76.57%, the highest gmean, and the highest roc_auc. It achieves an overall accuracy of 81.27%, with high precision and recall for the "survived" class and moderate precision and recall for the "death" class. The ROC curve demonstrates the model's strong discriminatory power. Conclusion: Integrating BLT and clinical data into CGA and applying ML models, particularly SVM, can effectively predict 180-day mortality in geriatric oncology patients. This approach holds promise for improving care and clinical decision-making in the field of geriatric oncology. AI explainable analyses provide insights into feature importance, enhancing the interpretability and clinical relevance of the models.

To make predictions in our model, access: https://huggingface.co/spaces/tiagopessoalim/Predicting180-DayMortalityInGeriatricOncology
