#1(a): MATRIX MULTIPLICATION
from matrixfunc import matrix_mul

rows_A = int(input("Enter rows of Matrix A: "))
cols_A = int(input("Enter columns of Matrix A: "))

A = []

print("Enter Matrix A:")

for i in range(rows_A):

    row = []

    for j in range(cols_A):
        element = int(input())
        row.append(element)

    A.append(row)

rows_B = int(input("Enter rows of Matrix B: "))
cols_B = int(input("Enter columns of Matrix B: "))

B = []

print("Enter Matrix B:")

for i in range(rows_B):

    row = []

    for j in range(cols_B):
        element = int(input())
        row.append(element)

    B.append(row)

if cols_A != rows_B:
    print("Matrix multiplication not possible")

else:
    result = matrix_mul(A, B)

    print("Result:")

    for row in result:
        print(row)
