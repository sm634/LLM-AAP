import pandas as pd

# Example multi-index series
index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)])
data = pd.Series([10, 20, 30, 40], index=index)

# Original multi-index series
print("Original Series:")
print(data)

# Pivot the multi-index series
pivot_data = data.unstack()
print("\nPivoted DataFrame:")
print(pivot_data)

breakpoint()