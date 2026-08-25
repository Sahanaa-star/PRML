def matrix_mul(A,B):
	row_a=len(A)
	col_a=len(A[0])
	row_b=len(B)
	col_b=len(B[0])
	if col_a!= row_b:
		return "Matrix multiplication is not possible"
	result=[]
	for m in range(row_a):
		row=[]
		for k in range (col_b):
			total=0
			for n in range(col_a):
				total+=A[m][n] * B[n][k]

			row.append(total)
		result.append(row)
	return result


def dot_product(A,B):
	if len(A)!=len(B):
		return "Dot product not possible"
	total=0
	for i in range(len(A)):
		total+=A[i] * B[i]

	return total


def transpose(A):
	rows=len(A)
	cols=len(A[0])
	result=[]
	for j in range(cols):
		row=[]
		for i in range(rows):
			row.append(A[i][j])
		result.append(row)
	return result


def is_symmetric(A):
    rows = len(A)
    cols = len(A[0])
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(cols):
            if A[i][j] != A[j][i]:
                return False
    return True


def is_upper_triangular(A):
    n = len(A)
    for i in range(n):
        for j in range(i):
            if A[i][j] != 0:
                return False
    return True


def is_lower_triangular(A):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != 0:
                return False
    return True

