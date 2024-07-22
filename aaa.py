import matplotlib.pyplot as plt

# Data for plotting
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a figure and axis
plt.figure(figsize=(8, 6))

# Plot x vs y
plt.plot(x, y, marker='o', linestyle='-', color='b')

# Title and labels
plt.title('Simple Plot Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Show plot
plt.show()