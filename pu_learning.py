# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import xgboost as xgb
import numpy as np
import lightgbm as lgb

data=pd.read_csv('dataset.csv', encoding='utf-8')

# Заменяем NaN на 0
data.fillna(0.0, inplace=True)

# В таблице все числа должны быть положительными
for j in data.columns:
    data.loc[data[j]<0,j]*=(-1)

# Создаем тестовую и тренировочную выборки 
x_data=data.iloc[:,2:]
y_data=data.iloc[:,1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=42, shuffle=False)

# Тренируем модель
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

# Оцениваем модель
def evaluate_results(y_test, y_predict):
    print('Результаты классификации:')
    f1 = f1_score(y_test, y_predict, average='micro')
    print("f1: %.2f%%" % (f1 * 100.0)) 
    rec = recall_score(y_test, y_predict, average='micro')
    print("recall: %.2f%%" % (rec * 100.0)) 
    prc = precision_score(y_test, y_predict, average='micro')
    print("precision: %.2f%%" % (prc * 100.0)) 
    acc = accuracy_score(y_test, y_predict)
    print("accuracy: %.2f%%" % (acc * 100.0)) 

evaluate_results(y_test, y_predict)

# Positive and unlabeled (PU) обучение
mod_data = data.copy()
# Получаем индексы положительных экземпляров
pos_ind = np.where(mod_data.iloc[:,1].values == 1)[0]
# Перемешиваем их
np.random.shuffle(pos_ind)
# Оставляем 30% положительных экземпляров - они и будут маркированными
pos_sample_len = int(np.ceil(0.3 * len(pos_ind)))
print(f'В качестве положительных используется {pos_sample_len}/{len(pos_ind)} экземляров, остальные немаркированные')
pos_sample = pos_ind[:pos_sample_len]
# Создаем целевую колонку, в которой (1) - положительные экземпляры, а (-1) - немаркированные
mod_data = data.copy()
mod_data['class_test'] = -1
mod_data.loc[pos_sample,'class_test'] = 1
print('Целевые переменные:\n', mod_data.iloc[:,-1].value_counts())

y_data = mod_data.iloc[:,-1]
df_orig_positive  = mod_data.iloc[y_data.values == 1]
df_orig_unlabeled = mod_data.iloc[y_data.values != 1]

x_data_pos = df_orig_positive.iloc[:,2:-1].values
x_data_unl = df_orig_unlabeled.iloc[:,2:-1].values

len_pos = x_data_pos.shape[0] # размер положительных экземляров
len_unlabeled = x_data_unl.shape[0] # размер немаркированных экземпляров
learners_num = 128 # количество обучений
bootstrap_sample_size = len_pos # случайный размер bootstrap выборки

# Создаем маркированный набор для каждого шага обучения
train_labels = np.zeros(shape=(len_pos + bootstrap_sample_size,))
# Заполняем первую часть набора положительными маркированными экземплярами 
train_labels[:len_pos] = 1.0
# Массив из количества предсказаний
n_oob = np.zeros(shape=(len_unlabeled,))
# Массив с результатами предсказаний
f_oob = np.zeros(shape=(len_unlabeled, 2))

for i in range(learners_num):
    # Bootstrap-повторная выборка
    bootstrap_sample = np.random.choice(np.arange(len_unlabeled), replace=True, size=bootstrap_sample_size)
    # Положительная выборка + bootstrap немаркированная выборка
    data_bootstrap = np.concatenate((x_data_pos,x_data_unl[bootstrap_sample, :]), axis=0)
    # Тренируем модель
    model = lgb.LGBMClassifier()
    model.fit(data_bootstrap, train_labels)
    idx_oob = sorted(set(range(len_unlabeled)) - set(np.unique(bootstrap_sample)))
    # Трансдуктивное обучение экземпляров oob
    f_oob[idx_oob] += model.predict_proba(x_data_unl[idx_oob])
    n_oob[idx_oob] += 1
    if(i%10 == 0): print(f'Обучение на стадии {i}/{learners_num}')
        
predicted = f_oob[:, 1]/n_oob

df_orig_predicted = df_orig_unlabeled.copy()
df_orig_predicted['pred'] = [1 if x > 0.5 else 0 for x in predicted]
df_orig_positive.loc[:,'pred'] = 1
df_outcome = pd.concat([df_orig_positive,df_orig_predicted])

df_outcome.to_csv('prediction.csv')

evaluate_results(df_orig_predicted.iloc[:,1].values, df_orig_predicted.iloc[:,-1].values)