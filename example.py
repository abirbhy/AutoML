import gym
import gym_deepline
from stable_baselines.common.vec_env import DummyVecEnv
from gym_deepline.agents.DDQNatml_weighted_obs import *
import os
import warnings
warnings.filterwarnings("ignore")
import time
import json
import yaml

start_time = time.time()

prim_list = ['NumericData', 'imputer', 'ImputerMedian', 'imputerIndicator', 'OneHotEncoder',
             'LabelEncoder', 'ImputerEncoderPrim', 'ImputerOneHotEncoderPrim',
             'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'StandardScaler',
             'QuantileTransformer', 'PowerTransformer', 'Normalizer', 'KBinsDiscretizerOrdinal', 'KBinsDiscretizerOneHot',
             'VarianceThreshold', 'UnivariateSelectChiKbest', 'f_classifKbest', 'mutual_info_classifKbest',
             'f_regressionKbest', 'mutual_info_regressionKbest', 'UnivariateSelectChiPercentile', 'f_classifPercentile',
             'mutual_info_classifPercentile', 'f_regressionPercentile', 'mutual_info_regressionPercentile',
             'UnivariateSelectChiFPR', 'f_classifFPR', 'f_regressionFPR', 'UnivariateSelectChiFDR', 'f_classifFDR',
             'f_regressionFDR', 'UnivariateSelectChiFWE', 'f_classifFWE', 'f_regressionFWE', 'RFE_RandomForest',
             'RFE_GradientBoosting', 'RFE_SVR', 'RFE_RandomForestReg',
             'PolynomialFeatures', 'InteractionFeatures', 'PCA_LAPACK', 'PCA_ARPACK', 'PCA_Randomized', 'IncrementalPCA',
             'KernelPCA', 'TruncatedSVD', 'FastICA', 'RandomTreesEmbedding',
             'RF_classifier', 'AdaBoostClassifier', 'BaggingClassifier', 'BernoulliNBClassifier', 'ComplementNBClassifier',
             'DecisionTreeClassifier', 'ExtraTreesClassifier', 'GaussianNBClassifier', 'GaussianProcessClassifierPrim',
             'GradientBoostingClassifier', 'KNeighborsClassifierPrim', 'LinearDiscriminantAnalysisPrim', 'LinearSVC',
             'LogisticRegression', 'LogisticRegressionCV', 'MultinomialNB', 'NearestCentroid',
             'PassiveAggressiveClassifier', 'QuadraticDiscriminantAnalysis', 'RidgeClassifier', 'RidgeClassifierCV',
             'SGDClassifier', 'SVC', 'XGBClassifier', 'BalancedRandomForestClassifier', 'EasyEnsembleClassifier',
             'RUSBoostClassifier', 'LGBMClassifier',
             'ARDRegression', 'LinearRegression', 'Ridge', 'LassoLars', 'ElasticNet', 'SGDRegressor', 'AdaBoostRegressor',
             'BaggingRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor',
             'DecisionTreeRegressor', 'KNeighborsRegressor', 'GaussianProcessRegressor', 'SVR', 'LinearSVR',
             'LGBMRegressor', 'XGBRegressor',
             'MajorityVoting', 'RandomForestMeta', 'AdaBoostClassifierMeta', 'AdaBoostRegressorMeta',
             'ExtraTreesMetaClassifier', 'ExtraTreesRegressorMeta', 'GradientBoostingClassifierMeta',
             'GradientBoostingRegressorMeta', 'XGBClassifierMeta', 'XGBRegressorMeta', 'RandomForestRegressorMeta'
             ]


def train_deepline(env, log_dir, datasets_indices):
    env.set_env_params(prim_list, lj_list=datasets_indices, embedd_size=15, log_pipelines=True)
    info = env.state_info
    env = AtmlMonitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    kwargs = dict(layers=[256, 128, 64, 8], state_info=info)
    model = DqnAtml(CustomPolicy, env, verbose=1, policy_kwargs=kwargs, prioritized_replay=True,
                    learning_rate=0.00005, gamma=0.98)
    env.envs[0].env.observation.model = model

    print('Start Training')
    model.learn(total_timesteps=10, log_interval=100)
    model.save(log_dir + "/last_model")
    return model


def test_deepline(env, model, datasets_idx):
    obs = env.reset()
    env.set_env_params(prim_list, datasets_idx, embedd_size=15, log_pipelines=True)
    env.observation.model = model
    x_train = env.observation.X_train.copy(deep=True)
    y_train = env.observation.Y_train.copy()
    x_test = env.observation.X_test.copy(deep=True)
    y_test = env.observation.Y_test.copy()
    model.set_env(env)

    ds = env.observation.learning_job.name
    print('Testing dataset: {}'.format(ds))

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.observation.pipe_run.produce(x_test)
            score = env.observation.pipe_run.learning_job.metric.evaluate(y_test.copy(), env.observation.pipe_run.produce_outputs['predictions'])
            print('Score: {}'.format(score))

            path = os.path.dirname(os.path.realpath(__file__)) + '/envs/pipelines'
            with open(path + '/pipelines_run_log.json') as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    if (key.startswith(ds)):
                        for key1, value1 in value.items():
                            if (key1 == 'score'):
                                if (1 > value1 > score):
                                    print(key)
                        #if (key1['name'] == ds) & (key1['score'] > score):
                            #print(key1.pipeline_id)
                        #print('k1', key1, 'v1', value1)
                    #if value.name == ds & value.score > score:
                        #print(value)


if __name__ == '__main__':
    log_dir = 'logs/'
    env = gym.make('deepline-v0')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_indices = list(range(0))
    test_indices = [1]

    num_training_steps = 150000  # change to 50,000-150,000 for better results!
    model = train_deepline(env, log_dir, train_indices)
    test_deepline(env, model, test_indices)

    print("--- %s seconds ---" % (time.time() - start_time))
