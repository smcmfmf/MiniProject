import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score

csv_list = os.listdir('./MiniProject/miniData/')
# print(csv_list)

male_data_list = []
female_data_list = []

# 남 여 데이터 분리
for filename in csv_list:
    if '_m_' in filename:
        male_data_list.append(filename)
    if '_w_' in filename:
        female_data_list.append(filename)

# print(male_data_list)
# print(female_data_list)

# 데이터 시각화
# df = pd.read_csv('./MiniProject/miniData/'+male_data_list[15], header=None)
# data = df.to_numpy()
# zero_mask = (data == 0)

# plt.figure(figsize=(8, 6))
# sns.heatmap(data, mask=zero_mask, cmap="Blues", cbar=True, annot=False, fmt=".2f", linewidths=0.5)

# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         if data[i, j] == 0:
#             plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='red'))
            
# plt.show()

train_data_name = [20220201, 20220202, 20220203, 20220204, 20220205, 20220206, 20220207, 20220208, 20220209, 20220210,
                   20220211, 20220212, 20220213, 20220214, 20220215, 20220216, 20220217, 20220218, 20220219]

# train, test 데이터 분리 함수
def get_train_test_data(file_list, train_name):
    train_data = []
    test_data = []

    for filename in file_list:
        matched = False
        for i in train_name:
            if str(i) in filename:
                train_data.append(filename)
                matched = True
                break
        if not matched:
            test_data.append(filename)
    return train_data, test_data

male_train_data, male_test_data = get_train_test_data(male_data_list, train_data_name)

# print(male_train_data[0])
# print(male_test_data[0])

female_train_data, female_test_data = get_train_test_data(female_data_list,train_data_name)

# print(female_train_data[0])
# print(female_test_data[0])

def labeling_data(file_list):
    data = []
    for i in range(len(file_list)-1):
        data.append(file_list[i+1])
    return data

# 남성 데이터 라벨링
male_train_label = labeling_data(male_train_data)
male_test_label = labeling_data(male_test_data)

#여성 데이터 라벨링

female_train_label = labeling_data(female_train_data)
female_test_label = labeling_data(female_test_data)

# print(male_train_label)


# 진짜 데이터 가져오기
# 데이터 가져오는 함수수
def get_data(file_list):
    data = []
    for filename in file_list:
        df = pd.read_csv('./MiniProject/miniData/'+filename, header=None)
        data.append(df.to_numpy())
    return data

# 데이터 가져오기

# 남성 데이터
male_train_df = get_data(male_train_data)
male_train_label_df = get_data(male_train_label)
male_test_df = get_data(male_test_data)
male_test_label_df = get_data(male_test_label)

# 여성 데이터
female_train_df = get_data(female_train_data)
female_train_label_df = get_data(female_train_label)
female_test_df = get_data(female_test_data)
female_test_label_df = get_data(female_test_label)

# 데이터 전처리

def data_preprocessing(data):
    data = np.expand_dims(np.array(data), axis=-1)
    return data

# 남성 데이터 전처리

m_X_train = data_preprocessing(male_train_df[:-1])
m_y_train = data_preprocessing(male_train_label_df)
m_X_test = data_preprocessing(male_test_df[:-1])
m_y_test = data_preprocessing(male_test_label_df)

# 여성 데이터 전처리

f_X_train = data_preprocessing(female_train_df[:-1])
f_y_train = data_preprocessing(female_train_label_df)
f_X_test = data_preprocessing(female_test_df[:-1])
f_y_test = data_preprocessing(female_test_label_df)

# CNN 모델 생성

# 남성 모델
# male_model = models.Sequential([
#     layers.Input(shape=m_X_train.shape[1:]),
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2), padding='same'),
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.UpSampling2D((2, 2)),
#     layers.Conv2D(1, (3, 3), activation='linear', padding='same'),
#     layers.Cropping2D(cropping=((0, 0), (0, 1)))
# ])

male_model = models.Sequential([
    layers.Input(shape=m_X_train.shape[1:]),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu'),
    layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu'),
    
    layers.Conv2D(1, (3, 3), activation='linear', padding='same'),
    # layers.Cropping2D(((0,0),(0,1)))  # 출력 shape 맞추기
    layers.ZeroPadding2D(((0, 0), (0, 1)))
])

male_model.compile(optimizer='adam', loss='mse')
# model.summary()

male_model.fit(m_X_train, m_y_train, epochs=20, batch_size=8, validation_data=(m_X_test, m_y_test))

m_y_pred_test = male_model.predict(m_X_test)

# 여성 모델
# female_model = models.Sequential([
#     layers.Input(shape=f_X_train.shape[1:]),
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2), padding='same'),
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.UpSampling2D((2, 2)),
#     layers.Conv2D(1, (3, 3), activation='linear', padding='same'),
#     layers.Cropping2D(cropping=((0, 0), (0, 1)))
# ])

female_model = models.Sequential([
    layers.Input(shape=f_X_train.shape[1:]),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu'),
    layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu'),
    
    layers.Conv2D(1, (3, 3), activation='linear', padding='same'),
    # layers.Cropping2D(((0,0),(0,1)))  # 출력 shape 맞추기
    layers.ZeroPadding2D(((0, 0), (0, 1)))
])

female_model.compile(optimizer='adam', loss='mse')
# model.summary()

female_model.fit(f_X_train, f_y_train, epochs=20, batch_size=8, validation_data=(f_X_test, f_y_test))

f_y_pred_test = female_model.predict(f_X_test)

# 예측 결과 평가

#남성 성능 출력
m_mse = mean_squared_error(m_y_test.flatten(), m_y_pred_test.flatten())
m_r2 = r2_score(m_y_test.flatten(), m_y_pred_test.flatten())
print(f"[Male] Test MSE: {m_mse:.4f}")
print(f"[Male] Test R^2: {m_r2:.4f}")

# 여성 성능 출력/
f_mse = mean_squared_error(f_y_test.flatten(), f_y_pred_test.flatten())
f_r2 = r2_score(f_y_test.flatten(), f_y_pred_test.flatten())
print(f"[Female] Test MSE: {f_mse:.4f}")
print(f"[Female] Test R^2: {f_r2:.4f}")

# 예측 결과 시각화


plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title("[Male] Input")
plt.imshow(m_X_test[0].squeeze(), cmap='Blues')
plt.subplot(1, 3, 2)
plt.title("[Male] True Label")
plt.imshow(m_y_test[0].squeeze(), cmap='Blues')
plt.subplot(1, 3, 3)
plt.title("[Male] Prediction")
plt.imshow(m_y_pred_test[0].squeeze(), cmap='Blues')

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title("[female] Input")
plt.imshow(f_X_test[0].squeeze(), cmap='Blues')
plt.subplot(1, 3, 2)
plt.title("[female] True Label")
plt.imshow(f_y_test[0].squeeze(), cmap='Blues')
plt.subplot(1, 3, 3)
plt.title("[female] Prediction")
plt.imshow(f_y_pred_test[0].squeeze(), cmap='Blues')
plt.show()