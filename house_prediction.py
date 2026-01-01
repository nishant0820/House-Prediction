import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Housing.csv")
binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]
for col in binary_cols:
    data[col] = data[col].map({"yes": 1, "no": 0})
data = pd.get_dummies(data, drop_first=True)
X = data.drop("price", axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print("Predicted House Prices:")
print(predictions[:5])
print("\nModel Accuracy:")
print(accuracy)