from matrixfunc import is_lower_triangular
n = int(input("Enter order of matrix: "))
print("Enter Matrix:")
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

if is_lower_triangular(A):
    print("Lower Triangular Matrix")
else:
    print("Not a Lower Triangular Matrix")


