import matplotlib.pyplot as plt

from gaussian import set_seed
from gaussian import generate_uniform
from gaussian import generate_gaussian

seed_value = int(input("Enter seed value: "))
set_seed(seed_value)

a = float(input("Enter lower limit (a): "))
b = float(input("Enter upper limit (b): "))
count = int(input("Enter number of random variables: "))

uniform_data = generate_uniform(a, b, count)

print("\nQ2(a): Uniform Random Variables")
print("First 10 values:")
print(uniform_data[:10])

plt.hist(uniform_data, bins=20, edgecolor="black")
plt.title("Histogram of Uniform Random Variables")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


mu = float(input("\nEnter mean (μ): "))
sigma = float(input("Enter standard deviation (σ): "))

gaussian_data = generate_gaussian(mu, sigma, count)

print("\nQ2(b): Gaussian Random Variables")
print("First 10 values:")
print(gaussian_data[:10])

plt.hist(gaussian_data, bins=30, edgecolor="black")
plt.title("Histogram of Gaussian Random Variables")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
