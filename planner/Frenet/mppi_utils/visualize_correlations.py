import numpy as np
import matplotlib.pyplot as plt

# Example data generation
mean = [0, 0]
a = 0.3
b = -0.14
c = -0.14
d = 0.3

cov = [[a,b], [c,d]]  # Covariance matrix with negative correlation
samples = np.random.multivariate_normal(mean, cov, size=1000)

A1_samples = samples[:, 0]
A2_samples = samples[:, 1]

# Plotting the scatter plot
plt.scatter(A1_samples, A2_samples, alpha=0.5)
plt.title('Scatter Plot of A1 vs A2')
plt.xlabel('Noise in acceleration')
plt.ylabel('Noise in Steering Angle')


# Make axes equal
plt.axis('equal')


plt.grid(True)
plt.show()