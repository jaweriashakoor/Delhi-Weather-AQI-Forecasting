import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('delhi-weather-aqi-2025.csv')

df['date_ist'] = pd.to_datetime(df['date_ist'], dayfirst=True)

df = df.sort_values('date_ist')

df['tomorrow_temp'] = df['temp_c'].shift(-1)

df = df.dropna()

features = ['temp_c', 'humidity', 'windspeed_kph', 'aqi_index', 'pressure_mb']
X = df[features]
y = df['tomorrow_temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print(f"Model Training Complete.")
print(f"Mean Absolute Error: {error:.2f} degrees Celsius")

latest_data_row = X.tail(1)
future_forecast = model.predict(latest_data_row)

print("-" * 30)
print(f"PREDICTION FOR THE NEXT STEP:")
print(f"The predicted temperature will be: {future_forecast[0]:.2f}°C")
print("-" * 30)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Weather (Delhi 2025)')
plt.show()