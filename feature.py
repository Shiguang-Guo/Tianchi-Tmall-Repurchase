import os
import pandas as pd

data_format1_path = './data/data_format1/'
feature_path = './feature/'

user_log = pd.read_csv(os.path.join(data_format1_path, 'user_log_format1.csv'), dtype={'time_stamp': 'str'})
user_info = pd.read_csv(os.path.join(data_format1_path, 'user_info_format1.csv'))
df_train = pd.read_csv(os.path.join(data_format1_path, 'train_format1.csv'))
df_test = pd.read_csv(os.path.join(data_format1_path, 'test_format1.csv'))

print(user_log.head())
print(user_info.head())
print(df_train.head())
print(df_test.head())
# df_train['origin'] = 'train'
# df_test['origin'] = 'test'
# df_train = pd.concat([df_train, df_test], ignore_index=True, sort=False).drop(['prob'], axis=1)

df_train = pd.merge(df_train, user_info, on='user_id', how='left')
print(df_train.head())
total_logs_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"]]).count().reset_index()[
    ["user_id", "seller_id", "item_id"]]
print(total_logs_temp.head())
total_logs_temp.rename(columns={"seller_id": "merchant_id", "item_id": "total_logs"}, inplace=True)
print(total_logs_temp.head())
df_train = pd.merge(df_train, total_logs_temp, on=["user_id", "merchant_id"], how="left")
print(df_train.head())
unique_item_ids_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["item_id"]]).count().reset_index()[
        ["user_id", "seller_id", "item_id"]]
print(unique_item_ids_temp.head())
unique_item_ids_temp1 = unique_item_ids_temp.groupby(
    [unique_item_ids_temp["user_id"], unique_item_ids_temp["seller_id"]]).count().reset_index()
print(unique_item_ids_temp1.head())
unique_item_ids_temp1.rename(columns={"seller_id": "merchant_id", "item_id": "unique_item_ids"}, inplace=True)
print(unique_item_ids_temp1.head())
df_train = pd.merge(df_train, unique_item_ids_temp1, on=["user_id", "merchant_id"], how="left")
print(df_train.head())
categories_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["cat_id"]]).count().reset_index()[
        ["user_id", "seller_id", "cat_id"]]
print(categories_temp.head())
categories_temp1 = categories_temp.groupby(
    [categories_temp["user_id"], categories_temp["seller_id"]]).count().reset_index()
print(categories_temp1.head())
categories_temp1.rename(columns={"seller_id": "merchant_id", "cat_id": "categories"}, inplace=True)
print(categories_temp1.head())
df_train = pd.merge(df_train, categories_temp1, on=["user_id", "merchant_id"], how="left")
print(df_train.head())
browse_days_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["time_stamp"]]).count().reset_index()[
        ["user_id", "seller_id", "time_stamp"]]
print(browse_days_temp.head())
browse_days_temp1 = browse_days_temp.groupby(
    [browse_days_temp["user_id"], browse_days_temp["seller_id"]]).count().reset_index()
print(browse_days_temp1.head())
browse_days_temp1.rename(columns={"seller_id": "merchant_id", "time_stamp": "browse_days"}, inplace=True)
print(browse_days_temp1.head())
df_train = pd.merge(df_train, browse_days_temp1, on=["user_id", "merchant_id"], how="left")
print(df_train.head())
one_clicks_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["action_type"]]).count().reset_index()[
        ["user_id", "seller_id", "action_type", "item_id"]]
print(one_clicks_temp.head())
one_clicks_temp.rename(columns={"seller_id": "merchant_id", "item_id": "times"}, inplace=True)
print(one_clicks_temp.head())
one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
print(one_clicks_temp.head())
one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
print(one_clicks_temp.head())
one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
print(one_clicks_temp.head())
four_features = one_clicks_temp.groupby(
    [one_clicks_temp["user_id"], one_clicks_temp["merchant_id"]]).sum().reset_index()
print(four_features.head())
four_features = four_features.drop(["action_type", "times"], axis=1)
df_train = pd.merge(df_train, four_features, on=["user_id", "merchant_id"], how="left")
print(df_train.head())

df_train = df_train.fillna(method='ffill')
df_train.to_csv(os.path.join(feature_path, 'features_train.csv'), index=False)

df_test = pd.merge(df_test, user_info, on="user_id", how="left")
df_test = pd.merge(df_test, total_logs_temp, on=["user_id", "merchant_id"], how="left")
df_test = pd.merge(df_test, unique_item_ids_temp1, on=["user_id", "merchant_id"], how="left")
df_test = pd.merge(df_test, categories_temp1, on=["user_id", "merchant_id"], how="left")
df_test = pd.merge(df_test, browse_days_temp1, on=["user_id", "merchant_id"], how="left")
df_test = pd.merge(df_test, four_features, on=["user_id", "merchant_id"], how="left")
df_test = df_test.fillna(method='bfill')
df_test = df_test.fillna(method='ffill')
print(df_test.head())
X1 = df_test.drop(['user_id', 'merchant_id', 'prob'], axis=1)
print(X1.head())
X1.to_csv(os.path.join(feature_path, 'features_test.csv'), index=False)
