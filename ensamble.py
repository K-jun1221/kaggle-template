import numpy as np
import pandas as pd
import statsmodels.api as sm

ensemble_3_train = pd.read_csv("./input/ensemble-3/train.csv")
ensemble_2_train = pd.read_csv("./input/ensemble-2/train.csv")
ensemble_1_train = pd.read_csv("./input/ensemble-1/train.csv")
train = pd.read_csv("./input/train.csv")

ensemble_3_predict = pd.read_csv("../input/ensemble-3/predict.csv")
ensemble_2_predict = pd.read_csv("../input/ensemble-2/predict.csv")
ensemble_1_predict = pd.read_csv("../input/ensemble-1/predict.csv")
test = pd.read_csv("./input/test.csv")

ensemble_3_train = ensemble_3_train.rename(columns={"prediction": "prediction3"})
ensemble_2_train = ensemble_2_train.rename(columns={"prediction": "prediction2"})
ensemble_1_train = ensemble_1_train.rename(columns={"prediction": "prediction1"})

ensemble_3_predict = ensemble_3_predict.rename(columns={"prediction": "prediction3"})
ensemble_2_predict = ensemble_2_predict.rename(columns={"prediction": "prediction2"})
ensemble_1_predict = ensemble_1_predict.rename(columns={"prediction": "prediction1"})

df_train = pd.concat(
  [
    train.loc[:,["id", "target"]],
    ensemble_3.loc[:,["prediction3"]],
    ensemble_2.loc[:,["prediction2"]],
    ensemble_1.loc[:,["prediction1"]]
  ],
  axis=1
)
df_predict = pd.concat(
  [
    test.loc[:,["id"]],
    ensemble_3_predict.loc[:,["prediction3"]],
    ensemble_2_predict.loc[:,["prediction2"]],
    ensemble_1_predict.loc[:,["prediction1"]]
  ],
  axis=1
)

x_train = df_train.loc[:, ['prediction3', 'prediction2', "prediction1"]]
y_train = df_train["target"].values

x_predict = df_predict.loc[:, ['prediction3', 'prediction2', "prediction1"]].values

logr = sm.OLS(y_train, x_train)
logr = logr.fit()
print(logr.summary())

y_predict = logr.predict(x_predict)
test["prediction"] = y_predict
test.to_csv('submission.csv', index=False)