import h2o
import pickle
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split

h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()

#nypd.to_pickle("nypd_df.pkl")

nypd = pd.read_pickle('nypd_df.pkl')

#print(nypd.head())

black = nypd.loc[nypd['PERP_RACE'] == "BLACK"]
print(black.head())

black = black[['PD_DESC', 'LAW_CODE', 'AGE_GROUP', 'ARREST_BORO', 'PERP_SEX', 'LAW_CAT_CD']]
train, test = train_test_split(black, test_size=0.2)

predictors = train.columns.values[:-1]
target = train.columns.values[-1]

print(predictors)
print(target)

rf_v1 = H2ORandomForestEstimator(
    model_id="rf_covType_v1",
    ntrees=200,
    stopping_rounds=2,
    score_each_iteration=True,
    seed=1000000)

h_train = h2o.H2OFrame(train)
h_test = h2o.H2OFrame(test)

rf_v1.train(list(predictors), str(target), training_frame=h_train, validation_frame=h_test)

print(rf_v1)


# To look at validation statistics, we can use the scoring history function.

# In[ ]:

print(rf_v1.score_history())

