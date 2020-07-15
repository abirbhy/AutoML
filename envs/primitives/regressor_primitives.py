from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor,\
    GradientBoostingRegressor
from sklearn.linear_model import ARDRegression, LinearRegression, Ridge, LassoLars, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from gym_deepline.envs.Primitives import primitive
from copy import deepcopy
import pandas as pd
import numpy as np
np.random.seed(1)

def handle_data(data):
    new_data = {}
    if len(data) == 1:
        new_data = deepcopy(data[0])
        # new_data['X'].columns = list(map(str, list(range(new_data['X'].shape[1]))))
    else:
        concatenated_df = pd.DataFrame()
        for d_input in data.values():
            df2 = deepcopy(d_input['X'][d_input['X'].columns.difference(concatenated_df.columns)])
            concatenated_df = pd.concat([concatenated_df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        # concatenated_df = concatenated_df.T.drop_duplicates().T
        new_data = deepcopy(data[0])
        new_data['X'] = concatenated_df.infer_objects()
        # new_data['X'].columns = list(map(str, list(range(new_data['X'].shape[1]))))
    cols = list(new_data['X'].columns)
    for i in range(len(cols)):
        col = cols[i]
        col = col.replace('[', 'abaqa')
        col = col.replace(']', 'bebab')
        col = col.replace('<', 'cfckc')
        col = col.replace('>', 'dmdad')
        cols[i] = col
    new_data['X'].columns = cols
    new_data['X'] = new_data['X'].loc[:, ~new_data['X'].columns.duplicated()]

    return new_data


class ARDRegressionPrim(primitive):
    def __init__(self, random_state=0):
        super(ARDRegressionPrim, self).__init__(name='ARDRegression')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Bayesian ARD regression. Fit the weights of a regression model, using an ARD prior. The weights of the regression model are assumed to be in Gaussian distributions. Also estimate the parameters lambda (precisions of the distributions of the weights) and alpha (precision of the distribution of the noise). The estimation is done by an iterative procedures (Evidence Maximization)"
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ARDRegression()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


# Added by me Begin
class LinearRegressionPrim(primitive):
    def __init__(self, random_state=0):
        super(LinearRegressionPrim, self).__init__(name='LinearRegression')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Linear Regression fits a linear model with coefficients to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LinearRegression()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class RidgePrim(primitive):
    def __init__(self, random_state=0):
        super(RidgePrim, self).__init__(name='Ridge')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients. The ridge coefficients minimize a penalized residual sum of squares: The complexity parameter alpha>0 controls the amount of shrinkage: the larger the value of , the greater the amount of shrinkage and thus the coefficients become more robust to collinearity."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = Ridge(alpha=0.5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class LassoLarsPrim(primitive):
    def __init__(self, random_state=0):
        super(LassoLarsPrim, self).__init__(name='LassoLars')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "LassoLars is a lasso model implemented using the LARS algorithm, and unlike the implementation based on coordinate descent, this yields the exact solution, which is piecewise linear as a function of the norm of its coefficients."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LassoLars(alpha=0.1)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class ElasticNetPrim(primitive):
    def __init__(self, random_state=0):
        super(ElasticNetPrim, self).__init__(name='ElasticNet')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "ElasticNet is a linear regression model trained with both l1 and l2-norm regularization of the coefficients. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. We control the convex combination of l1 and l2 using the l1_ratio parameter. Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both. A practical advantage of trading-off between Lasso and Ridge is that it allows Elastic-Net to inherit some of Ridge’s stability under rotation."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ElasticNet(alpha=0.1, l1_ratio=0.7)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class SGDRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(SGDRegressorPrim, self).__init__(name='SGDRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "The class SGDRegressor implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. SGDRegressor is well suited for regression problems with a large number of training samples (> 10.000), for other problems we recommend Ridge, Lasso, or ElasticNet."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = SGDRegressor(loss='squared_loss', penalty='l2')
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


# End
class AdaBoostRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(AdaBoostRegressorPrim, self).__init__(name='AdaBoostRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "An AdaBoost regressor. An AdaBoost [1] regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = AdaBoostRegressor(random_state=random_state)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class BaggingRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(BaggingRegressorPrim, self).__init__(name='BaggingRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "A Bagging regressor. A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [1]. If samples are drawn with replacement, then the method is known as Bagging [2]. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [3]. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [4]."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = BaggingRegressor(random_state=random_state, n_jobs=5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output

# Added by me Begin
class RandomForestRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(RandomForestRegressorPrim, self).__init__(name='RandomForestRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "In RandomForestRegressor classe, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = RandomForestRegressor(random_state=random_state, n_jobs=5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class ExtraTreesRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(ExtraTreesRegressorPrim, self).__init__(name='ExtraTreesRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "In extremely randomized trees (ExtraTreesRegressor classe), randomness goes one step further in the way splits are computed. As in random forests, a random subset of candidate features is used, but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense of a slightly greater increase in bias."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = ExtraTreesRegressor(random_state=random_state, n_jobs=5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class GradientBoostingRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(GradientBoostingRegressorPrim, self).__init__(name='GradientBoostingRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "GradientBoostingRegressor supports a number of different loss functions for regression which can be specified via the argument loss; the default loss function for regression is least squares ('ls')."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = GradientBoostingRegressor(random_state=random_state, n_estimators=5, learning_rate=0.1, max_depth=1, loss='ls')
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class DecisionTreeRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(DecisionTreeRegressorPrim, self).__init__(name='DecisionTreeRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Decision trees can be applied to regression problems, using the DecisionTreeRegressor class. The fit method will take as argument arrays X and y"
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = DecisionTreeRegressor()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class KNeighborsRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(KNeighborsRegressorPrim, self).__init__(name='KNeighborsRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "Neighbors-based regression can be used in cases where the data labels are continuous rather than discrete variables. The label assigned to a query point is computed based on the mean of the labels of its nearest neighbors. KNeighborsRegressor implements learning based on the k nearest neighbors of each query point, where k is an integer value specified by the user."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = KNeighborsRegressor(n_jobs=5)
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class GaussianProcessRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(GaussianProcessRegressorPrim, self).__init__(name='GaussianProcessRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "The GaussianProcessRegressor implements Gaussian processes (GP) for regression purposes. For this, the prior of the GP needs to be specified. The prior mean is assumed to be constant and zero (for normalize_y=False) or the training data’s mean (for normalize_y=True). The prior’s covariance is specified by passing a kernel object. The hyperparameters of the kernel are optimized during fitting of GaussianProcessRegressor by maximizing the log-marginal-likelihood (LML) based on the passed optimizer. As the LML may have multiple local optima, the optimizer can be started repeatedly by specifying n_restarts_optimizer. The first run is always conducted starting from the initial hyperparameter values of the kernel; subsequent runs are conducted from hyperparameter values that have been chosen randomly from the range of allowed values. If the initial hyperparameters should be kept fixed, None can be passed as optimizer. The noise level in the targets can be specified by passing it via the parameter alpha, either globally as a scalar or per datapoint. Note that a moderate noise level can also be helpful for dealing with numeric issues during fitting as it is effectively implemented as Tikhonov regularization, i.e., by adding it to the diagonal of the kernel matrix. An alternative to specifying the noise level explicitly is to include a WhiteKernel component into the kernel, which can estimate the global noise level from the data."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = GaussianProcessRegressor()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output

class SVRPrim(primitive):
    def __init__(self, random_state=0):
        super(SVRPrim, self).__init__(name='SVR')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "The model produced by Support Vector Regression depends only on a subset of the training data, because the cost function ignores samples whose prediction is close to their target."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = SVR()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class LinearSVRPrim(primitive):
    def __init__(self, random_state=0):
        super(LinearSVRPrim, self).__init__(name='LinearSVR')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "We make use of the epsilon-insensitive loss, i.e. errors of less than epsilon are ignored. This is the form that is directly optimized by LinearSVR."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LinearSVR()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class LGBMRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(LGBMRegressorPrim, self).__init__(name='LGBMRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "LightGBM is a gradient boosting framework that uses tree based learning algorithms."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = LGBMRegressor()
        self.accept_type = 'c_r'

    def can_accept(self, data):
        return self.can_accept_c(data, 'Regression')

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name+"Pred"])
        final_output = {0: output}
        return final_output


class XGBRegressorPrim(primitive):
    def __init__(self, random_state=0):
        super(XGBRegressorPrim, self).__init__(name='XGBRegressor')
        self.hyperparams = []
        self.type = 'Regressor'
        self.description = "XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples."
        self.hyperparams_run = {'default': True}
        self.random_state = random_state
        self.model = XGBRegressor(random_state=random_state, n_jobs=5)
        self.accept_type = 'xgb'

    def can_accept(self, data):
        # data = handle_data(data)
        if data['X'].empty:
            return False
        cols = data['X']
        num_cols = data['X']._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        if not data['learning_job'].task == 'Regression':
            return False
        elif not len(cat_cols) == 0:
            return False
        return True

    def is_needed(self, data):
        # data = handle_data(data)
        return True

    def fit(self, data):
        data = handle_data(data)
        self.model.fit(data['X'], data['Y'])

    def produce(self, data):
        output = handle_data(data)
        output['predictions'] = self.model.predict(output['X'])
        output['X'] = pd.DataFrame(output['predictions'], columns=[self.name + "Pred"])
        final_output = {0: output}
        return final_output
# End
