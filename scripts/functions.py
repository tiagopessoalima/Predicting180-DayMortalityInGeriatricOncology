from collections import Counter
from imblearn import FunctionSampler
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import numpy as np

# Function to convert dates from 'ddmmmyyyy' to 'YYYY-MM-DD'
def convert_date(date_str):
    try:
        month_dict = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        day = date_str[:2]
        month = month_dict.get(date_str[2:5].lower())
        year = date_str[5:]
        formatted_date = f'{year}-{month}-{day}'
        return formatted_date
    except Exception:
        return pd.NaT
  
# Define a function to calculate the geometric mean
def geometric_mean_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    specificity = precision_score(y_true, y_pred)
    return np.sqrt(recall * specificity)
    
def roughly_balanced_bagging(X, y, replace=False):
    """Implementation of Roughly Balanced Bagging for binary problem."""
    # find the minority and majority classes
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # compute the number of sample to draw from the majority class using
    # a negative binomial distribution
    n_minority_class = class_counts[minority_class]
    n_majority_resampled = np.random.negative_binomial(n=n_minority_class, p=0.5)

    # draw randomly with or without replacement
    majority_indices = np.random.choice(
        np.flatnonzero(y == majority_class),
        size=n_majority_resampled,
        replace=replace,
    )
    minority_indices = np.random.choice(
        np.flatnonzero(y == minority_class),
        size=n_minority_class,
        replace=replace,
    )
    indices = np.hstack([majority_indices, minority_indices])

    return X[indices], y[indices]
