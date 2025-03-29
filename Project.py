import os
import numpy as np
import pandas as pd

# Ensure data directory exists
data_dir = "resource_data"
os.makedirs(data_dir, exist_ok=True)

# Sample dataset generation if not exists
data_file = os.path.join(data_dir, "resource_usage.csv")

if not os.path.exists(data_file):
    np.random.seed(42)
    days = np.arange(1, 101)
    cpu_usage = np.random.randint(30, 80, size=100) + days * 0.2  # Simulated CPU usage pattern
    memory_usage = np.random.randint(1000, 5000, size=100) + days * 5  # Simulated Memory usage

    df = pd.DataFrame({"Day": days, "CPU_Usage": cpu_usage, "Memory_Usage": memory_usage})
    df.to_csv(data_file, index=False)

# Load dataset
df = pd.read_csv(data_file)

# Prepare training data
X = df["Day"].values
y_cpu = df["CPU_Usage"].values
y_memory = df["Memory_Usage"].values

# Least Squares Regression Function
def least_squares_fit(X, y):
    """Computes slope (m) and intercept (b) for y = mX + b using Least Squares Method"""
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    m = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    b = y_mean - m * X_mean
    return m, b

# Train prediction models
m_cpu, b_cpu = least_squares_fit(X, y_cpu)
m_memory, b_memory = least_squares_fit(X, y_memory)

# Predict future usage for the next 10 days
future_days = np.arange(101, 111)
cpu_pred = m_cpu * future_days + b_cpu
memory_pred = m_memory * future_days + b_memory

# Evaluate models using Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

cpu_mae = mean_absolute_error(y_cpu, m_cpu * X + b_cpu)
memory_mae = mean_absolute_error(y_memory, m_memory * X + b_memory)

print("Model Evaluation:")
print(f"CPU Usage Prediction MAE: {cpu_mae:.2f}")
print(f"Memory Usage Prediction MAE: {memory_mae:.2f}")

# Display predictions
future_df = pd.DataFrame({
    "Day": future_days,
    "Predicted_CPU_Usage": cpu_pred,
    "Predicted_Memory_Usage": memory_pred
})
print("\nPredicted Resource Usage for Next 10 Days:")
print(future_df)

# Resource Allocation Strategy
def allocate_resources(cpu, memory):
    """Allocate resources based on predicted demand."""
    if cpu > 75 or memory > 4500:
        return "High Allocation"
    elif cpu > 50 or memory > 3000:
        return "Medium Allocation"
    else:
        return "Low Allocation"

future_df["Resource_Allocation"] = future_df.apply(lambda row: allocate_resources(row["Predicted_CPU_Usage"], row["Predicted_Memory_Usage"]), axis=1)

print("\nResource Allocation Strategy:")
print(future_df)
