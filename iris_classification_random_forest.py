# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas and numpy
import pandas as pd
import numpy as np

# Set random seed
np.random.seed(0)

# Load the iris dataset
iris = load_iris()

# Create a dataframe with the feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species names to the dataframe
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# View the top 5 rows
df.head()

# Create a new column for train/test split
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Split the data
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

# Show the number of observations in each set
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))

# Define the feature columns
features = df.columns[:4]

# Convert species names to numerical values for training
y = pd.factorize(train['species'])[0]

# Initialize the classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the classifier
clf.fit(train[features], y)

# Predict species for the test data
preds = clf.predict(test[features])

# Display the first 10 predicted probabilities
print(preds[:10])
# Create confusion matrix
conf_matrix = pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

# Display the confusion matrix
print(conf_matrix)
# Display feature importance scores
print(list(zip(train[features], clf.feature_importances_)))
