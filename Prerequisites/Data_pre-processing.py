import pandas as pd


df = pd.read_csv('housing.csv')

# df.drop(['longitude','latitude'], axis=1, inplace=True)



df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())


df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
df['ocean_proximity_INLAND'] = df['ocean_proximity_INLAND'].astype(int)
df['ocean_proximity_ISLAND'] = df['ocean_proximity_ISLAND'].astype(int)
df['ocean_proximity_NEAR BAY'] = df['ocean_proximity_NEAR BAY'].astype(int)
df['ocean_proximity_NEAR OCEAN'] = df['ocean_proximity_NEAR OCEAN'].astype(int)

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
df['income_per_room'] = df['median_income'] / df['total_rooms']
df['income_per_person'] = df['median_income'] / df['population']




# df.drop(['total_bedrooms','total_rooms','population'], axis=1, inplace=True)
# df.drop(['ocean_proximity_ISLAND'], axis=1, inplace=True)



y = df['median_house_value']
X = df.drop('median_house_value', axis=1)


one_hot_cols = [col for col in X.columns if col.startswith('ocean_proximity_')]


def normalize_columns(df, exclude_cols):
    df = df.copy()
    for col in df.columns:
        if col not in exclude_cols:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    return df

X_normalized = normalize_columns(X, exclude_cols=one_hot_cols)


data = X_normalized.copy()
data['median_house_value'] = y



data = data.sample(frac=1, random_state=42).reset_index(drop=True)


split_index = int(0.8 * len(data))
train_data = data.iloc[:split_index]
val_data = data.iloc[split_index:]


X_train = train_data.drop('median_house_value', axis=1)
y_train = train_data['median_house_value']
X_val = val_data.drop('median_house_value', axis=1)
y_val = val_data['median_house_value']


X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
