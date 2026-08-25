from matrixfunc  import is_symmetric
n = int(input("Enter order of matrix: "))
print("Enter Matrix:")
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)
if is_symmetric(A):
    print("Symmetric Matrix")
else:
    print("Not Symmetric")
