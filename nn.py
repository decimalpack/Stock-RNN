# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Sequential

WINDOW_SIZE=10
FORECAST_SIZE=10
def prepare_data(stock_name,end_date):
	df = web.DataReader(stock_name, data_source="yahoo", start="2020-01-01", end=end_date)
	data = df[["Close"]]
	dataset = data.values
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(dataset)
	X_train = []
	y_train = []
	for i in range(WINDOW_SIZE, len(scaled_data)):
		X_train.append(scaled_data[i-WINDOW_SIZE:i, 0])
		y_train.append(scaled_data[i, 0])
	X_train, y_train = np.array(X_train), np.array(y_train)
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
	return df,X_train,y_train,scaler

	
def train(x,y):
	model = Sequential([
		Input(shape=(x.shape[1],1)),
		GRU(150,return_sequences=True),
		GRU(150,return_sequences=False),
		Dense(25),
		Dense(1),
	])
	model.compile(optimizer="adam", loss="mean_squared_error")
	model.fit(x, y, batch_size=16, epochs=2)
	return model



def predictions(model, scaler, window, df):
	for _ in range(FORECAST_SIZE):
		pred = model.predict(window)
		window = np.append(window,pred[0,0]).reshape(1,-1,1)
	dates = pd.date_range(df.Close.index[-1],periods=FORECAST_SIZE+1)[1:]
	window = scaler.inverse_transform(window.reshape(1,-1))
	window = np.squeeze(window)[WINDOW_SIZE:]
	historic = df.Close.tail(2*WINDOW_SIZE+FORECAST_SIZE)
	ff = pd.Series(window,index=dates)
	forecast = historic.append(ff)

	plt.figure(figsize=(16,8))
	plt.title("Model")
	plt.ylabel("Closing Price INR", fontsize=18)
	plt.plot(forecast,color="red")
	plt.plot(historic,color="blue")
	plt.savefig("static/plot.png")
	return ff

def main(stock_name,end_date):
	df,x,y,scaler=prepare_data(stock_name,end_date)
	model= train(x,y)
	window = x[-1].reshape(1,-1,1)
	w = predictions(model,scaler,window,df)
	return w.to_json(date_format="iso")

if __name__ == '__main__':
	stock_name = "ICICIBANK.BO"
	end_date = "2020-11-25"
	main(stock_name,end_date)
