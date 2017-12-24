import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import History
from Anomaly_detector import Anomaly

def scale_dataset(train, test):
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	tr_scaled = scaler.transform(train)
	te_scaled = scaler.transform(test)
	return scaler, tr_scaled, te_scaled

def main():
	x = []
	with open('Logs.txt', 'r') as f:
		for line in f:
			i = line[0:-1]
			x.append(int(i))
	print('Fetched data ....')

	'''
	y = x[0:len(x)-1]
	x = np.concatenate((np.array([0]).reshape((1, 1, 1)), np.array([np.array(x[i]) - np.array(x[i-1]) for i in range(1, len(x)-1)]).reshape((1, 1, 1))))
	x, scaler = scale_dataset(x)
	self.scaler = scaler
	y = scaler.transform(y)
	x = np.reshape(x, (len(x), 1, 1))
	y = np.reshape(y, (len(y), 1, 1))
	'''

	print('Building the anomaly_detector ....')
	inp_shape = (1, 1, 1)
	twitter_anomaly = Anomaly(inp_shape, 0, 0)
	twitter_anomaly.anomaly_detector_train(x, 1, 20)
	twitter_anomaly.save_anomaly_model()

if __name__ == '__main__':
	main()
