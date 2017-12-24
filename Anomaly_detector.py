import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import History

def scale_dataset(data):
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(data)
	data_scaled = scaler.transform(data)
	return scaler, data_scaled

class Anomaly:

	def __init__(self, inp_shape, x_curr, threshold):
		self.inp_shape = inp_shape
		model = Sequential()
		model.add(LSTM(128, return_sequences=True, stateful=True, batch_input_shape=self.inp_shape))
		model.add(Dropout(0.4))
		model.add(LSTM(64, return_sequences=True, stateful=True))
		model.add(Dropout(0.3))
		model.add(LSTM(32, return_sequences=True, stateful=True))
		model.add(Dropout(0.2))
		model.add(LSTM(16, return_sequences=False, stateful=True))
		model.add(Dropout(0.1))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		self.model = model
		self.model.save_weights('online_anomaly_detector_model')
		self.x_curr = x_curr
		self.loss = []
		self.threshold = threshold
		self.forecast = []
		self.real = []

	def online_anomaly_detector(self, x):
		x_tr = np.reshape(self.x_curr, self.inp_shape)
		y_tr = np.reshape(x, (1, 1, 1))
		pred = self.model.predict(x_tr)
		l = np.sqrt(np.mean(np.square(pred[0, 0, 0] - y_tr[0, 0, 0])))
		self.loss.append(l)
		self.model.load_weights('online_anomaly_detector_model')
		self.model.fit(x_tr, y_tr, batch_size=1)
		self.forecast.append((pred[0, 0, 0]))
		self.real.append((y_tr[0, 0, 0]))
		self.model.save_weights('online_anomaly_detector_model')
		self.x_curr.append(x)
		self.x_curr = self.x_curr[1:]
		if (pred[0, 0, 0] - pred[0, 0, 0]*(1.0 - self.threshold)) <= y_tr[0, 0, 0] <= (pred[0, 0, 0] + pred[0, 0, 0]*self.threshold):
			return b'Normal behaviour !!!!'
		else:
			return b'Anomalous behaviour observed last 2secs !!!!'

	def anomaly_detector_test(self, x):
		self.real.append(x[0, 0, 0])
		pred = self.model.predict(self.x_curr)
		self.forecast.append(pred[0, 0, 0])
		pred = pred + x
		self.x_curr = x
		rmse = np.sqrt(np.mean(np.square(pred[0, 0, 0] - x[0, 0, 0])))
		print('rmse: %f  threshold: %f', rmse, self.threshold)
		if rmse > self.threshold:
			return b'Anomalous behaviour observed last 2secs !!!!'
		else:
			return b'Normal behaviour ....'

	def plot_performance(self):
		plt.plot(self.real, 'g')
		plt.plot(self.forecast, 'r')
		plt.savefig('Forecast_performance.png')
		plt.close()

	def plot_loss(self):
		plt.plot(self.loss, 'r')
		plt.savefig('Performance_loss.png')
		plt.close()

	def save_anomaly_model(self):
		self.model.save_weights('online_anomaly_detector_model', overwrite=True)
		np.savez_compressed('Parameters', threshold=self.threshold, x_curr=self.x_curr)
