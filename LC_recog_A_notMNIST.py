# Author: Vidit Jain
# Dataset used: notMNIST

from PIL import Image
import numpy as np
import os

def loadData():
	X = []
	Y = []
	for filename in os.listdir("A/"):
		img = Image.open("A/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(1)

	X_train = X[:-200]
	Y_train = Y[:-200]
	X_test = X[-200:]
	Y_test = Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("B/"):
		img = Image.open("B/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("C/"):
		img = Image.open("C/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("D/"):
		img = Image.open("D/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("E/"):
		img = Image.open("E/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("F/"):
		img = Image.open("F/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("G/"):
		img = Image.open("G/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("H/"):
		img = Image.open("H/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("I/"):
		img = Image.open("I/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	for filename in os.listdir("J/"):
		img = Image.open("J/" + filename).convert("L")
		imgArray = np.array(img)
		X.append(imgArray)
		Y.append(0)

	X_train = X_train + X[:-200]
	Y_train = Y_train + Y[:-200]
	X_test = X_test + X[-200:]
	Y_test = Y_test + Y[-200:]
	X = []
	Y = []

	X_train = np.array(X_train)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	Y_train = np.array(Y_train)

	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)
	Y_train = Y_train.reshape(Y_train.shape[0],1)
	Y_test = Y_test.reshape(Y_test.shape[0],1)

	return X_train, X_test, Y_train, Y_test

def sigmoid(X):
	return 1/(1 + np.exp(-X))

def init_with_zeroes(dim):

	w = np.zeros(dim).reshape(dim,1)
	b = 0

	return w,b

def prop(w,b,X,Y):

	m = X.shape[0]
	A = sigmoid(np.dot(X,w) + b).T
	assert(A.shape == Y.shape)
	cost = (-1/m)*(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))

	dw = (1/m) * (np.dot(X.T, (A-Y).T))
	db = (1/m) * np.sum(A-Y)
	assert(dw.shape == w.shape)
	grad = {"dw" : dw, "db" : db}


	return grad, cost

def opt(w,b,X,Y,num, learnrate):

	for i in range(num):
		print("Loop Iteration: " + str(i))
		grad,cost = prop(w,b,X,Y)
		dw = grad["dw"]
		db = grad["db"]
		w = w - learnrate*dw
		b = b - learnrate*db


	return w,b

def predict(w,b,X):

	A = sigmoid(np.dot(X,w) + b)

	Y = (A > 0.5) + 0

	return Y

def model(X,Y):
	
	m = X.shape[1]
	w, b = init_with_zeroes(m)

	w,b = opt(w,b,X,Y,1000,0.005)

	Y_pred = predict(w,b,X)

	return Y_pred,w,b

X_train, X_test, Y_train, Y_test = loadData()
Y_pred,w,b = model(X_train, Y_train.T)
y_train_comp = (Y_pred == Y_train) + 0
print("Training Accuracy: " + str((np.sum(y_train_comp)/Y_train.shape[0])*100))

Y_pred_test = predict(w,b,X_test)
np.savetxt("Y_pred_test.txt", Y_pred_test, fmt='%d')
y_test_comp = (Y_pred_test == Y_test) + 0
print("Test Accuracy: " + str((np.sum(y_test_comp)/Y_test.shape[0])*100))
