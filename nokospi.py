import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn. preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

samsung=pd.read_csv('삼성전자 내역.csv', engine='python', header=0, index_col=0, sep=',')
sk=pd.read_csv('SK하이닉스 내역.csv',  engine='python', header=0, index_col=0, sep=',')
db=pd.read_csv('DB하이텍 내역.csv',  engine='python', header=0, index_col=0, sep=',')
kospi=pd.read_csv('코스피 내역.csv',  engine='python', header=0, index_col=0, sep=',')




#정렬을 일자별 오름차순으로 변경
samsung=samsung.sort_values(['날짜'], ascending=['True'])
sk=sk.sort_values(['날짜'], ascending=['True'])
db=db.sort_values(['날짜'], ascending=['True'])
kospi=kospi.sort_values(['날짜'], ascending=['True'])


#필요한 컬럼
samsung=samsung[['종가', '오픈', '고가', '저가']]
sk=sk[['종가', '오픈', '고가', '저가']]
db=db[['종가', '오픈', '고가', '저가']]
kospi=kospi[['종가', '오픈', '고가', '저가']]


#콤마 제거 후 문자를 정수로 변환
for i in range(len(samsung.index)) :
    for j in range(len(samsung.iloc[i])) :
        samsung.iloc[i, j]=int(samsung.iloc[i, j].replace(',', ''))

for i in range(len(sk.index)) :
    for j in range(len(sk.iloc[i])) :
        sk.iloc[i, j]=int(sk.iloc[i, j].replace(',', ''))

for i in range(len(db.index)) :
    for j in range(len(db.iloc[i])) :
        db.iloc[i, j]=int(db.iloc[i, j].replace(',', ''))
        
for i in range(len(kospi.index)) :
    for j in range(len(kospi.iloc[i])) :
        kospi.iloc[i, j]=int(float(kospi.iloc[i, j].replace(',', '')))



samsung_x=samsung[['종가', '고가', '저가']]
samsung_y=samsung[['오픈']]

# 9월 1일 데이터 삭제
samsung_x.drop(samsung_x.index[-1], inplace=True)
sk.drop(sk.index[-1], inplace=True)
samsung_y.drop(samsung_y.index[-1], inplace=True)


#to numpy
samsung_x=samsung_x.to_numpy()
samsung_y=samsung_y.to_numpy()
sk_x=sk.to_numpy()
db_x=db.to_numpy()




#데이터 스케일링
scaler1=MinMaxScaler()
scaler1.fit(samsung_x)
samsung_x=scaler1.transform(samsung_x)

scaler2=MinMaxScaler()
scaler2.fit(sk_x)
sk_x=scaler2.transform(sk_x)

scaler3=MinMaxScaler()
scaler3.fit(db_x)
db_x=scaler3.transform(db_x)



# x 데이터 다섯개씩 자르기
def split_data(x, size) :
    data=[]
    for i in range(x.shape[0]-size+1) :
        data.append(x[i:i+size,:])
    return np.array(data)

size=5
samsung_x=split_data(samsung_x, size)
sk_x=split_data(sk_x, size)
db_x=split_data(db_x, size)

sk_x=sk_x[:samsung_x.shape[0],:]
db_x=db_x[:samsung_x.shape[0],:]


# y 데이터 추출
samsung_y=samsung_y[size+1:, :]



# predict 데이터 추출
samsung_x_predict=samsung_x[-1]
sk_x_predict=sk_x[-1]
db_x_predict=db_x[-1]


samsung_x=samsung_x[:-2, :, :]
sk_x=sk_x[:-2, :, :]
db_x=db_x[:-2, :, :]


print(samsung_x.shape) 
print(sk_x.shape) 
print(db_x.shape) 

print(samsung_y.shape) 

samsung_x=samsung_x.astype('float32')
samsung_y=samsung_y.astype('float32')
samsung_x_predict=samsung_x_predict.astype('float32')
sk_x=sk_x.astype('float32')
sk_x_predict=sk_x_predict.astype('float32')
db_x=db_x.astype('float32')
db_x_predict=db_x_predict.astype('float32')

np.save('samsung_x.npy', arr=samsung_x)
np.save('samsung_x_predict.npy', arr=samsung_x_predict)
np.save('samsung_y.npy', arr=samsung_y)
np.save('sk_x.npy', arr=sk_x)
np.save('sk_x_predict.npy', arr=sk_x_predict)
np.save('db_x.npy', arr=db_x)
np.save('db_x_predict.npy', arr=db_x_predict)

# train, test 분리
samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test=train_test_split(samsung_x, samsung_y, train_size=0.8)
sk_x_train, sk_x_test, db_x_train, db_x_test=train_test_split(sk_x, db_x, train_size=0.8)


samsung_x_predict=samsung_x_predict.reshape(1,5,3)
sk_x_predict=sk_x_predict.reshape(1,5,4)
db_x_predict=db_x_predict.reshape(1,5,4)



######### 2. LSTM 회귀모델
samsung_input1=Input(shape=(5,3))
samsung_layer1=LSTM(400, activation='relu')(samsung_input1)
samsung_layer2=Dense(900,activation='relu')(samsung_layer1)
samsung_layer2=Dropout(0.1)(samsung_layer2)
samsung_layer4=Dense(200,activation='relu')(samsung_layer2)
samsung_layer5=Dense(20,activation='relu')(samsung_layer4)
samsung_layer6=Dense(10,activation='relu')(samsung_layer5)
samsung_layer7=Dense(5,activation='relu')(samsung_layer6)
samsung_output=Dense(1)(samsung_layer7)

sk_input1=Input(shape=(5,4))
sk_layer1=LSTM(512, activation='relu')(sk_input1)
sk_layer1=Dropout(0.2)(sk_layer1)
sk_layer2=Dense(4096, activation='relu')(sk_layer1)
sk_layer3=Dropout(0.2)(sk_layer2)
sk_layer3=Dense(2048, activation='relu')(sk_layer3)
sk_layer3=Dropout(0.2)(sk_layer3)
sk_layer3=Dense(512, activation='relu')(sk_layer3)
sk_layer3=Dense(256, activation='relu')(sk_layer3)
sk_layer3=Dense(128, activation='relu')(sk_layer3)
sk_layer3=Dense(64, activation='relu')(sk_layer3)
sk_output=Dense(1)(sk_layer3)

db_input1=Input(shape=(5,4))
db_layer1=LSTM(512, activation='relu')(db_input1)
db_layer1=Dropout(0.2)(db_layer1)
db_layer2=Dense(4096, activation='relu')(db_layer1)
db_layer3=Dropout(0.2)(db_layer2)
db_layer3=Dense(2048, activation='relu')(db_layer3)
db_layer3=Dropout(0.2)(db_layer3)
db_layer3=Dense(512, activation='relu')(db_layer3)
db_layer3=Dense(256, activation='relu')(db_layer3)
db_layer3=Dense(128, activation='relu')(db_layer3)
db_layer3=Dense(64, activation='relu')(db_layer3)
db_output=Dense(1)(db_layer3)



merge1=concatenate([samsung_output, sk_output, db_output])

output1=Dense(5000)(merge1)
output1=Dropout(0.1)(output1)
output2=Dense(3000)(output1)
output2=Dropout(0.1)(output2)
output3=Dense(800)(output2)
output4=Dense(30)(output3)
output5=Dense(1)(output4)

model=Model(inputs=[samsung_input1, sk_input1, db_input1], outputs=output5)

model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss',  patience=100)
modelpath='D:\loss\bestmodel.hdf5'
cp=ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True)
hist=model.fit([samsung_x_train, sk_x_train, db_x_train], samsung_y_train, epochs=500, batch_size=100, validation_split=0.2, callbacks=[es, cp])


#4. 평가, 예측
loss=model.evaluate([samsung_x_test, sk_x_test, db_x_test], samsung_y_test, batch_size=100)
samsung_y_predict=model.predict([samsung_x_predict, sk_x_predict, db_x_predict])

print("loss : ", loss)
print("2021.09.01. 수요일 삼성전자 시가 :" , samsung_y_predict)

#5. 그래프
plot_figure = plt.figure(figsize=(30, 10))
plot_rst = plot_figure.add_subplot(111)
plot_rst.plot(samsung_y, label='Real')
plot_rst.plot(samsung_y_predict, label='Predict')
plot_rst.legend()
plt.show()


#rmse
rmse = np.sqrt(mean_squared_error(samsung_y_predict, samsung_y))
print("RMSE of train: %.3f"%np.sqrt(hist.history['loss'][-1]))
print("RMSE of val  : %.3f"%np.sqrt(hist.history['val_loss'][-1]))
