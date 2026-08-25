from matrixfunc import is_upper_triangular
n = int(input("Enter order of matrix: "))
print("Enter Matrix:")
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

if is_upper_triangular(A):
    print("Upper Triangular Matrix")
else:
    print("Not an Upper Triangular Matrix")
