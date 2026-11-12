
from sklearn.model_selection import train_test_split
import joblib
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
#print(iris)

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

print(X.shape)
print(y.shape)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual values:", y_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted flower:", iris.target_names[prediction[0]])


# MODEL COMPARISON

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Model 2: Logistic Regression
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200)
)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 3: Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Compare accuracy
print("\n--- MODEL COMPARISON ---")
print("Decision Tree:", accuracy_score(y_test, dt_pred))
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))


# CROSS VALIDATION

dt_scores = cross_val_score(
    dt_model, X, y, cv=5
)

print("\nCross Validation Scores:", dt_scores)

print("Average CV Accuracy:", dt_scores.mean())

print("CV Accuracy Percentage:", dt_scores.mean() * 100, "%")


# CHECKING OVERFITTING

train_prediction = dt_model.predict(X_train)
test_prediction = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_prediction)
test_accuracy = accuracy_score(y_test, test_prediction)

print("\n--- OVERFITTING CHECK ---")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# DECISION TREE DEPTH COMPARISON

for depth in [1, 2, 3, 5, 10, None]:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5)

    print(
        "Depth:", depth,
        "| Average Accuracy:", scores.mean()
    )


    # FINAL TUNED MODEL

best_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

best_model.fit(X_train, y_train)

final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("\n--- FINAL TUNED MODEL ---")
print("Best Depth: 3")
print("Test Accuracy:", final_accuracy)
print("Test Accuracy Percentage:", final_accuracy * 100, "%")


# SAVING THE MODEL

joblib.dump(best_model, "iris_model.pkl")

print("Model saved successfully!")

# LOADING THE SAVED MODEL

loaded_model = joblib.load("iris_model.pkl")

print("Model loaded successfully!")


# Make a prediction using the loaded model

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = loaded_model.predict(new_flower)

print("Predicted flower:", iris.target_names[prediction[0]])