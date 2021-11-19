
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import FinanceDataReader as fdr
import os

samsung_train_data = pd.DataFrame(fdr.DataReader('005930', '2000-02-01', '2020-12-31'))
t=pd.DataFrame(fdr.DataReader('005930', '2021-01-01', '2021-01-31'))

samsung_train_data = samsung_train_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
t = t[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


print(samsung_train_data)
for i in range(len(samsung_train_data.index)) :
    for j in range(len(samsung_train_data.iloc[i])) :
        samsung_train_data.iloc[i, j]=int(samsung_train_data.iloc[i, j])




plt.figure(figsize=(16, 9))
sns.lineplot(y=samsung_train_data['Close'], x=samsung_train_data.index)
plt.xlabel('time')
plt.ylabel('price')


scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled_train = scaler.fit_transform(samsung_train_data[scale_cols])
t_test = scaler.fit_transform(t[scale_cols])
print(scaled_train)
df = pd.DataFrame(scaled_train, columns=scale_cols)

x_train, x_test, y_train, y_test = train_test_split(df, df['Close'], test_size=0.2, random_state=0, shuffle=False)

def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

WINDOW_SIZE=20
BATCH_SIZE=32

train_data = windowed_dataset(scaled_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(scaled_train, WINDOW_SIZE, BATCH_SIZE, False)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
    
    model = Sequential([
    # 1차원 feature map 생성
    #Conv1D(filters=32, kernel_size=5,
    #       padding="causal",
    #       activation="relu",
    #       input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, input_shape = (1, 5), activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=3)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)
history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

model.load_weights(filename)
pred = model.predict(t_test)
pred.shape

plt.figure(figsize=(12, 9))
plt.plot(scaled_test[20:, 3], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()

def performance_measure(x):
    mse = mean_squared_error(x['actual'], x['pred'])
    rmse = np.sqrt(mse)
    return rmse

result_df = pd.DataFrame({'Date': samsung_test_data.reset_index().loc[20:, 'Date'].values, 'pred': pred.ravel(), 'actual': scaled_test[20:, 3]})
result_df['Month'] = result_df.Date.dt.month
rmse_df = pd.DataFrame(result_df.groupby('Month').apply(performance_measure)).reset_index().rename(columns={0:'RMSE'})
print(rmse_df)
