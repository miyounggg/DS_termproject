import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('C:\\Users\\choim\\DS_py\\16\\data.csv')
data['smoking_status'] = data['smoking_status'].replace('Unknown', 'no info')

data['age'] = np.ceil(data['age'])

nom_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
data = pd.get_dummies(data, columns=nom_cols)

data_with_bmi = data.dropna(subset=['bmi'])
data_missing_bmi = data[data['bmi'].isna()].copy()

X = data_with_bmi.drop(columns=['bmi', 'id'])
y = data_with_bmi['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) on Test Data:", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

X_missing = data_missing_bmi.drop(columns=['bmi', 'id'])
predicted_bmi = model.predict(X_missing)
predicted_bmi_rounded = predicted_bmi.round(2)
data_missing_bmi['bmi'] = predicted_bmi_rounded.astype(float)

filled_data = pd.concat([data_with_bmi, data_missing_bmi])

scaler = MinMaxScaler()
filled_data_scaled = filled_data.copy()
filled_data_scaled[['bmi']] = scaler.fit_transform(filled_data[['bmi']].values.reshape(-1, 1))

filled_data_scaled.to_csv('filled_bmi.csv', index=False)