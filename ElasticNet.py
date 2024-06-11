import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import joblib

df = pd.read_csv(
    "C:/DESKTOP CLONE/Major/Major Work Final/Code/Data/Gain_table_radiation_pattern_by_HFSS.csv"
)

X = df[["Lp", "Wp", "Phi", "Theta"]]
y = df["dB(GainTotal)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ElasticNet(random_state=42)

X_test = np.array(X_test)

model.fit(X_train, y_train)

print(X_test)

y_pred = model.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

plt.scatter(y_test, y_pred)
plt.show()

print(f"Mean Squared Error: {mse:.2f}")

user_input = input("Enter feature values Lp, Wp, Theta separated by commas: ").split(
    ","
)
ans = []

with open("output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Angle", "Predicted Output"])

    for i in range(361):
        copy_input = []
        x = str(i - 180)
        copy_input = user_input[:]
        copy_input.insert(2, x)

        user_features = np.array([float(i) for i in copy_input]).reshape(1, -1)
        predicted_output = model.predict(user_features)

        ans.append(str(predicted_output))

        writer.writerow([x, predicted_output[0]])

print(ans)

joblib.dump(
    model,
    "C:/DESKTOP CLONE/Major/Major Work Final/Code/Models/ElasticNet.pkl",
)
