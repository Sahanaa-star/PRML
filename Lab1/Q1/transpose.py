from matrixfunc import transpose

rows = int(input("Enter rows: "))
cols = int(input("Enter columns: "))

print("Enter Matrix:")

A = []

for i in range(rows):
    row = list(map(int, input().split()))
    A.append(row)

result = transpose(A)

print("Transpose:")

for row in result:
    print(row)


