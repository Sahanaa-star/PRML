# 1(a) Generate two Gaussian datasets D1 and D2
import numpy as np
import matplotlib.pyplot as plt

from gaussian import set_seed
from gaussian import generate_gaussian

seed_value = int(input("Enter seed value: "))
set_seed(seed_value)

mu1 = float(input("Enter mean for X1: "))
mu2 = float(input("Enter mean for X2: "))

count = int(input("Enter number of random variables: "))

sigma = 1

X1 = generate_gaussian(mu1, sigma, count)
X2 = generate_gaussian(mu2, sigma, count)

print("\nX1 first 10 values:")
print(X1[:10])

print("\nX2 first 10 values:")
print(X2[:10])

# 1(b) Form the 2-D data vector X and mean vector
X = np.column_stack((X1, X2))

mu = np.array([mu1, mu2])

print("\nMean vector:")
print(mu)

print("\nFirst 10 two dimensional values:")
print(X[:10])

# 1(c) Estimate covariance matrix, eigenvalues and eigenvectors
covariance = np.cov(X, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(covariance)

print("\nCovariance matrix:")
print(covariance)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# 1(d) Plot the constant density curves
x1 = np.linspace(-2, 10, 100)
x2 = np.linspace(-2, 10, 100)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

diff = points - mu

covariance_inv = np.linalg.inv(covariance)

value = np.sum((diff @ covariance_inv) * diff, axis=1)

value = value.reshape(X1_grid.shape)

plt.contour(X1_grid, X2_grid, value, levels=[1, 2, 3, 4, 5])

plt.scatter(X[:, 0], X[:, 1])

plt.scatter(mu1, mu2, marker="x", s=100)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Constant Density Curves")

plt.grid(True)

plt.xlim(-2, 10)
plt.ylim(-2, 10)

plt.xticks(np.arange(-2, 11, 2))
plt.yticks(np.arange(-2, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 1(e) Plot eigenvectors and mark major and minor axes
plt.figure()

plt.scatter(X[:, 0], X[:, 1])

plt.scatter(mu1, mu2, marker="x", s=100)

for i in range(2):
    vector = eigenvectors[:, i]
    length = np.sqrt(eigenvalues[i])

    color = "black" if eigenvalues[i] == np.max(eigenvalues) else "deeppink"
    label = "Major axis" if eigenvalues[i] == np.max(eigenvalues) else "Minor axis"

    plt.plot(
        [mu1 - vector[0] * length, mu1 + vector[0] * length],
        [mu2 - vector[1] * length, mu2 + vector[1] * length],
        color=color,
        linewidth=3,
        label=label
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Eigenvectors and Major-Minor Axes")
plt.legend()
plt.grid(True)
plt.xlim(-2, 10)
plt.ylim(-2, 10)

plt.xticks(np.arange(-2, 11, 2))
plt.yticks(np.arange(-2, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 1(f) Compare the major and minor axes with the eigenvalues
if eigenvalues[0] > eigenvalues[1]:
    major = eigenvalues[0]
    minor = eigenvalues[1]
else:
    major = eigenvalues[1]
    minor = eigenvalues[0]

print("\nMajor axis eigenvalue:")
print(major)

print("\nMinor axis eigenvalue:")
print(minor)

# 2(a) Generate data using diagonal covariance matrix

sigma11 = 2
sigma22 = 0.5

C = np.array([
    [sigma11**2, 0],
    [0, sigma22**2]
])

C_sqrt = np.array([
    [sigma11, 0],
    [0, sigma22]
])

Y = mu + (X - mu) @ C_sqrt

print("\nDiagonal covariance matrix:")
print(C)

print("\nFirst 10 values of Y:")
print(Y[:10])

# 2(b) Estimate covariance matrix, eigenvalues and eigenvectors

covariance_Y = np.cov(Y, rowvar=False)

eigenvalues_Y, eigenvectors_Y = np.linalg.eig(covariance_Y)

print("\nCovariance matrix of Y:")
print(covariance_Y)

print("\nEigenvalues of Y:")
print(eigenvalues_Y)

print("\nEigenvectors of Y:")
print(eigenvectors_Y)

# 2(c) Plot the constant density curves

x1 = np.linspace(-6, 10, 100)
x2 = np.linspace(-6, 10, 100)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

diff = points - mu

covariance_inv = np.linalg.inv(covariance_Y)

value = np.sum((diff @ covariance_inv) * diff, axis=1)

value = value.reshape(X1_grid.shape)

plt.contour(X1_grid, X2_grid, value, levels=[1, 2, 3, 4, 5])

plt.scatter(Y[:, 0], Y[:, 1])

plt.scatter(mu1, mu2, marker="x", s=100)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Constant Density Curves - Diagonal Covariance")

plt.grid(True)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xticks(np.arange(-6, 11, 2))
plt.yticks(np.arange(-6, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 2(d) Plot eigenvectors and mark major and minor axes

plt.figure()

plt.scatter(Y[:, 0], Y[:, 1])

plt.scatter(mu1, mu2, marker="x", s=100)

for i in range(2):
    v = eigenvectors_Y[:, i]

    color = "black" if eigenvalues_Y[i] == np.max(eigenvalues_Y) else "deeppink"
    label = "Major axis" if eigenvalues_Y[i] == np.max(eigenvalues_Y) else "Minor axis"

    plt.plot(
        [mu1 - v[0] * 2, mu1 + v[0] * 2],
        [mu2 - v[1] * 2, mu2 + v[1] * 2],
        color=color,
        linewidth=3,
        label=label
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Eigenvectors and Major-Minor Axes - Diagonal Covariance")

plt.grid(True)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xticks(np.arange(-6, 11, 2))
plt.yticks(np.arange(-6, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 2(e) Compare the major and minor axes with the eigenvalues

if eigenvalues_Y[0] > eigenvalues_Y[1]:
    major = eigenvalues_Y[0]
    minor = eigenvalues_Y[1]
else:
    major = eigenvalues_Y[1]
    minor = eigenvalues_Y[0]

print("\nMajor axis eigenvalue:")
print(major)

print("\nMinor axis eigenvalue:")
print(minor)

# 3(a) Generate data using full covariance matrix

sigma11 = 2
sigma22 = 1
sigma12 = 1

C = np.array([
    [sigma11**2, sigma12],
    [sigma12, sigma22**2]
])

C_sqrt = np.linalg.cholesky(C)

Y3 = mu + (X - mu) @ C_sqrt.T

print("\nFull covariance matrix:")
print(C)

print("\nFirst 10 values of Y:")
print(Y3[:10])

# 3(b) Plot the constant density curves

x1 = np.linspace(-6, 10, 100)
x2 = np.linspace(-6, 10, 100)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

diff = points - mu

covariance_inv = np.linalg.inv(C)

value = np.sum((diff @ covariance_inv) * diff, axis=1)

value = value.reshape(X1_grid.shape)

plt.contour(X1_grid, X2_grid, value, levels=[1, 2, 3, 4, 5])

plt.scatter(Y3[:, 0], Y3[:, 1])

plt.scatter(mu1, mu2, marker="x", s=100)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Constant Density Curves - Full Covariance")

plt.grid(True)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xticks(np.arange(-6, 11, 2))
plt.yticks(np.arange(-6, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 3(c) Estimate eigenvalues and eigenvectors

eigenvalues_Y3, eigenvectors_Y3 = np.linalg.eig(np.cov(Y3, rowvar=False))

print("\nCovariance matrix of Y:")
print(np.cov(Y3, rowvar=False))

print("\nEigenvalues:")
print(eigenvalues_Y3)

print("\nEigenvectors:")
print(eigenvectors_Y3)

# 3(d) Plot eigenvectors and mark major and minor axes

plt.figure()

plt.scatter(
    Y3[:, 0],
    Y3[:, 1],
    alpha=0.6,
    label="Gaussian data"
)

plt.scatter(
    mu1,
    mu2,
    marker="x",
    s=100,
    color="red",
    label="Mean"
)

major_index = np.argmax(eigenvalues_Y3)
minor_index = np.argmin(eigenvalues_Y3)

for i in range(2):
    vector = eigenvectors_Y3[:, i]
    length = np.sqrt(eigenvalues_Y3[i])

    if i == major_index:
        axis_label = "Major axis"
        axis_color = "black"
    else:
        axis_label = "Minor axis"
        axis_color = "pink"

    plt.plot(
        [mu1 - vector[0] * length,
         mu1 + vector[0] * length],
        [mu2 - vector[1] * length,
         mu2 + vector[1] * length],
        color=axis_color,
        linewidth=3,
        label=axis_label
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Eigenvectors and Major-Minor Axes - Full Covariance")

plt.legend()
plt.grid(True)
plt.xlim(-6, 10)
plt.ylim(-6, 10)

plt.xticks(np.arange(-6, 11, 2))
plt.yticks(np.arange(-6, 11, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 4(a) Generate two Gaussian classes

mu_class1 = np.array([2, 5])
mu_class2 = np.array([7, 5])

C1 = np.array([
    [4, 1],
    [1, 1]
])

C2 = np.array([
    [1, 0],
    [0, 4]
])

L1 = np.linalg.cholesky(C1)
L2 = np.linalg.cholesky(C2)

Y1 = mu_class1 + (X - mu) @ L1.T
Y2 = mu_class2 + (X - mu) @ L2.T

print("\nClass 1 mean:")
print(mu_class1)

print("\nClass 2 mean:")
print(mu_class2)

print("\nFirst 10 values of Class 1:")
print(Y1[:10])

print("\nFirst 10 values of Class 2:")
print(Y2[:10])

# 4(b) Plot the two Gaussian classes

plt.figure()

plt.scatter(Y1[:, 0], Y1[:, 1], label="Class 1")
plt.scatter(Y2[:, 0], Y2[:, 1], label="Class 2")

plt.scatter(mu_class1[0], mu_class1[1], marker="x", s=100)
plt.scatter(mu_class2[0], mu_class2[1], marker="x", s=100)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Two Gaussian Classes")

plt.legend()
plt.grid(True)
plt.xlim(-6, 12)
plt.ylim(-6, 12)

plt.xticks(np.arange(-6, 13, 2))
plt.yticks(np.arange(-6, 13, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# 4(c) Plot the decision boundary

x1 = np.linspace(-6, 12, 200)
x2 = np.linspace(-6, 12, 200)

X1_grid, X2_grid = np.meshgrid(x1, x2)

points = np.column_stack((X1_grid.ravel(), X2_grid.ravel()))

inv_C1 = np.linalg.inv(C1)
inv_C2 = np.linalg.inv(C2)

d1 = np.sum(((points - mu_class1) @ inv_C1) * (points - mu_class1), axis=1)
d2 = np.sum(((points - mu_class2) @ inv_C2) * (points - mu_class2), axis=1)

boundary = (d1 - d2).reshape(X1_grid.shape)

plt.figure()

plt.scatter(Y1[:, 0], Y1[:, 1], label="Class 1")
plt.scatter(Y2[:, 0], Y2[:, 1], label="Class 2")

plt.contour(X1_grid, X2_grid, boundary, levels=[0])

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundary")

plt.legend()
plt.grid(True)
plt.xlim(-6, 12)
plt.ylim(-6, 12)

plt.xticks(np.arange(-6, 13, 2))
plt.yticks(np.arange(-6, 13, 2))

plt.gca().set_aspect("equal", adjustable="box")
plt.show()
