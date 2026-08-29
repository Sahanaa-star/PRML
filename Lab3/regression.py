import numpy as np
import matplotlib.pyplot as plt


x = []
y = []

with open("noisy_8.txt", "r") as file:
    for line in file:
        values = line.split()

        if len(values) >= 2:
            x.append(float(values[0]))
            y.append(float(values[1]))

x = np.array(x)
y = np.array(y)


train_x = []
train_y = []

test_x = []
test_y = []

val_x = []
val_y = []


for i in range(len(x)):

    position = i % 10

    if position < 6:
        train_x.append(x[i])
        train_y.append(y[i])

    elif position < 8:
        test_x.append(x[i])
        test_y.append(y[i])

    else:
        val_x.append(x[i])
        val_y.append(y[i])


train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = np.array(test_x)
test_y = np.array(test_y)

val_x = np.array(val_x)
val_y = np.array(val_y)


def mean_std(values):

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = variance ** 0.5

    return mean, std


x_mean, x_std = mean_std(train_x)
y_mean, y_std = mean_std(train_y)


def normalize(values, mean, std):

    normalized = []

    for value in values:
        normalized.append(
            (value - mean) / std
        )

    return np.array(normalized)


train_x_norm = normalize(
    train_x,
    x_mean,
    x_std
)

test_x_norm = normalize(
    test_x,
    x_mean,
    x_std
)

val_x_norm = normalize(
    val_x,
    x_mean,
    x_std
)

train_y_norm = normalize(
    train_y,
    y_mean,
    y_std
)

test_y_norm = normalize(
    test_y,
    y_mean,
    y_std
)

val_y_norm = normalize(
    val_y,
    y_mean,
    y_std
)


def create_design_matrix(x_values, degree):

    matrix = []

    for value in x_values:

        row = []

        for power in range(degree + 1):
            row.append(value ** power)

        matrix.append(row)

    return np.array(matrix)


def train_polynomial(x_values, y_values, degree):

    X = create_design_matrix(
        x_values,
        degree
    )

    T = y_values

    W = (
        np.linalg.inv(X.T @ X)
        @ X.T
        @ T
    )

    return W


def calculate_error(X, T, W):

    difference = T - X @ W

    error = (
        difference.T
        @ difference
    )

    return float(error)


best_degree = None
best_weights = None
best_test_mse = None

degrees = []

train_mse_values = []
test_mse_values = []
val_mse_values = []


for degree in range(1, 11):

    weights = train_polynomial(
        train_x_norm,
        train_y_norm,
        degree
    )

    X_train = create_design_matrix(
        train_x_norm,
        degree
    )

    X_test = create_design_matrix(
        test_x_norm,
        degree
    )

    X_val = create_design_matrix(
        val_x_norm,
        degree
    )

    train_sse = calculate_error(
        X_train,
        train_y_norm,
        weights
    )

    test_sse = calculate_error(
        X_test,
        test_y_norm,
        weights
    )

    val_sse = calculate_error(
        X_val,
        val_y_norm,
        weights
    )

    train_mse = (
        train_sse
        / len(train_y_norm)
    )

    test_mse = (
        test_sse
        / len(test_y_norm)
    )

    val_mse = (
        val_sse
        / len(val_y_norm)
    )

    degrees.append(degree)

    train_mse_values.append(
        train_mse
    )

    test_mse_values.append(
        test_mse
    )

    val_mse_values.append(
        val_mse
    )

    print(
        "Degree:",
        degree,
        " Train MSE:",
        train_mse,
        " Test MSE:",
        test_mse,
        " Validation MSE:",
        val_mse
    )

    if (
        best_test_mse is None
        or test_mse < best_test_mse
    ):

        best_test_mse = test_mse
        best_degree = degree
        best_weights = weights.copy()


print()

print(
    "Best Degree =",
    best_degree
)

print(
    "Best Test MSE =",
    best_test_mse
)

print()

print("Best Weights:")


for i in range(len(best_weights)):

    print(
        "w",
        i,
        "=",
        best_weights[i]
    )


X_val_best = create_design_matrix(
    val_x_norm,
    best_degree
)

val_sse = calculate_error(
    X_val_best,
    val_y_norm,
    best_weights
)

val_mse = (
    val_sse
    / len(val_y_norm)
)


print()

print(
    "Validation SSE =",
    val_sse
)

print(
    "Validation MSE =",
    val_mse
)


predicted_val_norm = (
    X_val_best
    @ best_weights
)


predicted_val = []


for value in predicted_val_norm:

    predicted_val.append(
        value * y_std
        + y_mean
    )


predicted_val = np.array(
    predicted_val
)


residuals = (
    val_y
    - predicted_val
)


all_x_norm = normalize(
    x,
    x_mean,
    x_std
)


X_all = create_design_matrix(
    all_x_norm,
    best_degree
)


fitted_y_norm = (
    X_all
    @ best_weights
)


fitted_y = []


for value in fitted_y_norm:

    fitted_y.append(
        value * y_std
        + y_mean
    )


fitted_y = np.array(
    fitted_y
)


plt.figure()

plt.scatter(
    x,
    y,
    s=3
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Original Noisy Data"
)

plt.grid()
plt.show()


plt.figure()

plt.plot(
    degrees,
    train_mse_values,
    marker="o",
    label="Training MSE"
)

plt.plot(
    degrees,
    val_mse_values,
    marker="o",
    label="Validation MSE"
)

plt.xlabel(
    "Polynomial Degree"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "Bias-Variance Trade-off"
)

plt.legend()
plt.grid()
plt.show()


plt.figure()

plt.scatter(
    predicted_val,
    residuals,
    s=8
)

plt.axhline(
    y=0
)

plt.xlabel(
    "Predicted Values"
)

plt.ylabel(
    "Residuals"
)

plt.title(
    "Residuals vs Predicted Values"
)

plt.grid()
plt.show()


plt.figure()

plt.scatter(
    x,
    y,
    s=3,
    alpha=0.15,
    label="Original Data"
)

plt.plot(
    x,
    fitted_y,
    linewidth=3,
    label="Fitted Curve"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Polynomial Regression - Degree "
    + str(best_degree)
)

plt.legend()
plt.grid()
plt.show()
