import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.flatten()

X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv").values.flatten()


model = LinearRegression()

start_time= time.time()
model.fit(X_train, y_train)
end_time = time.time()

fitting_time = end_time - start_time
print(f"Convergence Time: {fitting_time:.2f} seconds")

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)



# Validation
mae_val = mean_absolute_error(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_val = r2_score(y_val, y_val_pred)

# Training
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

print(f"\nValidation MAE: {mae_val:.4f}")
print(f"Validation RMSE: {rmse_val:.4f}")
print(f"Validation R²: {r2_val:.4f}")

print(f"\nTraining MAE: {mae_train:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Training R²: {r2_train:.4f}")




