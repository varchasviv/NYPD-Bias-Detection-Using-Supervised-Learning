import h2o
import pickle
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.model_selection import train_test_split

h2o.init(max_mem_size = "10G")             #specify max number of bytes. uses all cores by default.
h2o.remove_all()

#nypd.to_pickle("nypd_df.pkl")

nypd = pd.read_pickle('nypd_df.pkl')
nypd = nypd[['PD_DESC', 'PERP_RACE', 'LAW_CODE', 'AGE_GROUP', 'ARREST_BORO', 'PERP_SEX', 'LAW_CAT_CD']]
nypd.dropna(inplace=True)

#print(nypd.head())

black = nypd.loc[nypd['PERP_RACE'] == "BLACK"]
print(black.head())
black.drop(['PERP_RACE'], axis=1, inplace=True)

train, test = train_test_split(black, test_size=0.2)

predictors = train.columns.values[:-1]
target = train.columns.values[-1]

print(predictors)
print(target)

rf_v1 = H2ORandomForestEstimator(
    model_id="rf_covType_v1",
    ntrees=200,
    stopping_rounds=7,
    score_each_iteration=True,
    max_depth=50,
    seed=1000000)

h_train = h2o.H2OFrame(train)
h_test = h2o.H2OFrame(test)

rf_v1.train(list(predictors), str(target), training_frame=h_train, validation_frame=h_test)

print(rf_v1)


# To look at validation statistics, we can use the scoring history function.

# In[ ]:

print(rf_v1.score_history())

# save the model
h2o.save_model(model=rf_v1, path="rf_v1", force=True)

# print model_path
# /tmp/mymodel/DeepLearning_model_python_1441838096933

# load the model
# saved_model = h2o.load_model(model_path)

