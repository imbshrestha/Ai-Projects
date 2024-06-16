import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('/Users/bimalshrestha/Desktop/opencv/iris.csv')

# Split the data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN Classifier
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Input the values for sepal length, sepal width, petal length, and petal width
sepal_length = float(input("Enter sepal length in cm: "))
sepal_width = float(input("Enter sepal width in cm: "))
petal_length = float(input("Enter petal length in cm: "))
petal_width = float(input("Enter petal width in cm: "))

# Predict the type of iris
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
input_data_scaled = scaler.transform(input_data)
predicted_class = knn_classifier.predict(input_data_scaled)[0]

# Print the predicted class
print(f"The predicted class of iris is: {predicted_class}")
