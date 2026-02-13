import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

cancers = ["breast", "lung", "oral", "thyroid"]
accuracies = []

for c in cancers:
    print("Training:", c)

    data = pd.read_csv(f"datasets/{c}.csv")
    data = data.apply(lambda col: col.astype('category').cat.codes)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

accuracies.append(acc)

joblib.dump(model, f"{c}.joblib")

# plot accuracy graph
plt.bar(cancers, accuracies)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.show()
