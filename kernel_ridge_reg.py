import numpy as np
import matplotlib.pyplot as plt
import math

raw_data = []

n = 0
with open("ridgetrain.txt") as f:
	for line in f:
		t = []
		raw_data.append(line.split())
		n = n + 1

x = np.ndarray(shape = (n,1))
y = np.ndarray(shape = (n,1))

for i in range(0,n):
	try:
		x[i] = float(raw_data[i][0]) #input training data
		y[i] = float(raw_data[i][1]) # output traning data
	except ValueError, e:
		print "error", e, "on line", i
m = 0
test_data = []
with open("ridgetest.txt") as f:
	for line in f:
		test_data.append(line.split())
		m = m + 1
x_test = np.ndarray(shape = (m,1))
y_test = np.ndarray(shape = (m,1))
for i in range(0,n):
	x_test[i] = float(test_data[i][0]) #input test data
	y_test[i] = float(test_data[i][1]) #output test data

alpha = np.ndarray(shape = (n,1))
hy_param = [0.1,1,10,100]

fig, ax = plt.subplots()
colors = ['red','blue','green','yellow']
labels = ['0.1','1','10','100']
rmse = []
for l in range(0,4):
	alpha = np.dot(np.linalg.inv(np.dot(x,x.T) + hy_param[l]*np.eye(n)),y)
	y_pred = np.ndarray(shape = (m,1))
	for i in range(0,m):
		y_t = 0
		for j in range(0,n):
			y_t = y_t + alpha[j]*np.exp(-0.1*(x[j] - x_test[i])**2)
		y_pred[i] = y_t
	rmse = math.sqrt(np.mean((y_test - y_pred)**2))
	print rmse
	ax.scatter(x_test,y_pred,color =colors[l],marker = '*',label = 'lambda = ' + labels[l])
#	plt.scatter(x_test,y_pred,color ='blue',marker = '*',label = 'predicted')
ax.scatter(x_test,y_test,color = 'black',marker = '+',label = 'actual')
print n
print m
ax.legend()
plt.show()

