import numpy as np

seed = 0 

def set_seed(value):
	global seed
	seed = value

def my_random():
	global seed

	a = 1103515245
	c = 12345
	m = 2**31

	seed = (a * seed + c) % m
	return seed / m

def my_sqrt(x):
	if x == 0:
		return 0
	guess = x/2

	for i in range(50):
		guess = (guess + x / guess) / 2

	return guess

def my_ln(x):

	e = 2.718281828459045
	exponent = 0

	while x > 1.5:
		x = x / e
		exponent += 1

	while x < 0.5:
		x = x * e
		exponent -= 1

	y = (x-1) / (x+1)
	y2 = y * y

	total = 0 
	term = y

	for i in range(1, 50, 2):
		total += term / i
		term = term * y2

	return 2 * total + exponent

def generate_uniform(a, b, count):
	values = []

	for i in range(count):
		r = my_random()
		x = a + (b - a) * r
		values.append(x)
	return np.array(values)

def generate_gaussian(mu,sigma,count):
	values = []

	while len(values) < count:
		u1 = -1 + 2 * my_random()
		u2 = -1 + 2 * my_random()
		s = u1*u1 + u2*u2

		if s > 0 and s < 1:
			k = my_sqrt((-2 * my_ln(s))/s)
			x = u1*k
			y = u2*k

			values.append(mu + sigma*x)

			if len(values) < count:
				values.append(mu + sigma*y)
	return np.array(values)
