import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

model_type = 'XGBClassifier'
feature_path = './feature/'
data_format1_path = './data/data_format1/'
prediction_path = './prediction/'
submission = pd.read_csv(os.path.join(data_format1_path, 'test_format1.csv'))

matrix = pd.read_csv(os.path.join(feature_path, 'features_train.csv'))
train_y = matrix['label']
train_X = matrix.drop(['user_id', 'merchant_id', 'label'], axis=1)

test_data = pd.read_csv(os.path.join(feature_path, 'features_test.csv'))

if model_type == 'MLPClassifier':
    model = MLPClassifier(solver='lbfgs', activation='relu', alpha=0.1, random_state=0,
                          hidden_layer_sizes=[10, 10]).fit(train_X, train_y)
elif model_type == 'XGBClassifier':
    X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.25)
    model = xgb.XGBClassifier(
        max_depth=5,
        n_estimators=500,
        min_child_weight=100,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=42,
        alpha=1,
        learning_rate=0.1
    )
    model.fit(
        X_train, y_train,
        eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=True,
        # 早停法，如果auc在10epoch没有进步就stop
        early_stopping_rounds=10
    )

prob = model.predict_proba(test_data)
submission['prob'] = pd.Series(prob[:, 1])
submission.to_csv(os.path.join(prediction_path, 'pred_ml.csv'), index=False)
