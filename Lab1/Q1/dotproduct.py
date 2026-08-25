from matrixfunc import dot_product

n = int(input("Enter size of vectors: "))

print("Enter Vector A:")
A = list(map(int, input().split()))

print("Enter Vector B:")
B = list(map(int, input().split()))

result = dot_product(A, B)

print(result)
