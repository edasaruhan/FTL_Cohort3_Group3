import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv('Augmented_Crop_recommendationV2.csv')

for col in df.select_dtypes(include = np.number):
  if col == 'irrigation_frequency':
    continue
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3-Q1
  lower = Q1 - 1.5 * IQR
  upper = Q3 + 1.5 * IQR
  df[col] = np.clip(df[col], lower, upper)


df['irrigation_frequency'] = df['irrigation_frequency'] - 1

df['irrigation_frequency'].value_counts()

df['Nutrient_Index'] = (df['N'] + df['P'] + df['K']) / 3

df.drop(columns=['N', 'P', 'K','organic_matter','rainfall','sunlight_exposure','fertilizer_usage','crop_density','pest_pressure','water_source_type','frost_risk','urban_area_proximity','wind_speed','water_usage_efficiency'], inplace=True)

for column in df['soil_type']:
  if column == 1:
    df['soil_type'] = df['soil_type'].replace(1, 'Sandy')
  elif column == 2:
    df['soil_type'] = df['soil_type'].replace(2, 'Loamy')
  elif column == 3:
    df['soil_type'] = df['soil_type'].replace(3, 'Clay')

for column in df['growth_stage']:
  if column == 1:
    df['growth_stage'] = df['growth_stage'].replace(1, 'Seedling')
  elif column == 2:
    df['growth_stage'] = df['growth_stage'].replace(2, 'Vegetative')
  elif column == 3:
    df['growth_stage'] = df['growth_stage'].replace(3, 'Flowering')

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encode_list = ['label', 'growth_stage', 'soil_type']

for col in encode_list:
  encoded_data = encoder.fit_transform(df[[col]])
  encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
  df = pd.concat([df, encoded_df], axis=1)
  df.drop(columns=[col], inplace=True)


df['irrigation_frequency'] = df['irrigation_frequency'].apply(lambda x: 0 if x <= 3 else (1 if x == 4 else 2))

X = df.drop('irrigation_frequency', axis=1)
y = df['irrigation_frequency']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_xgb = XGBClassifier(colsample_bytree =  1.0, learning_rate = 0.2, max_depth = 7, n_estimators = 200, subsample = 1.0)

model_xgb.fit(X_train, y_train)

pickle.dump(model_xgb, open("model.pkl", "wb"))