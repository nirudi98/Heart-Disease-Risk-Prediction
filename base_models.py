from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


class BaseModels:
    def __init__(self):
        self.base_list = None

    # get a list of models to evaluate
    def get_models(self):
        self.base_list = dict()
        self.base_list['lr'] = LogisticRegression()
        self.base_list['bayes'] = GaussianNB()
        self.base_list['MLP'] = MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10), max_iter=2000, random_state=1000)
        self.base_list['knn9'] = KNeighborsClassifier(12)
        self.base_list['knn27'] = KNeighborsClassifier(27)
        self.base_list['cart'] = DecisionTreeClassifier(criterion="entropy", max_depth=5)
        self.base_list['cart_gini'] = DecisionTreeClassifier(criterion="gini", max_depth=10)
        self.base_list['svm_linear'] = SVC(gamma="auto", kernel="linear", probability=True)
        self.base_list['RF500'] = RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10)
        self.base_list['RF200'] = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=10)
        self.base_list['LDA'] = LinearDiscriminantAnalysis()
        self.base_list['ET10'] = ExtraTreesClassifier(max_depth=10)
        self.base_list['ET500'] = ExtraTreesClassifier(n_estimators=500, max_depth=10)
        self.base_list['ET1000'] = ExtraTreesClassifier(n_estimators=1000)
        self.base_list['AB'] = AdaBoostClassifier()
        self.base_list['XGB'] = xgb.XGBClassifier(use_label_encoder=False)
        self.base_list['XGB500'] = xgb.XGBClassifier(n_estimators=500, use_label_encoder=False)
        self.base_list['XGB1000'] = xgb.XGBClassifier(n_estimators=1000, use_label_encoder=False)
        self.base_list['XGB2000'] = xgb.XGBClassifier(n_estimators=2000, use_label_encoder=False)
        self.base_list['GBM'] = GradientBoostingClassifier(n_estimators=200, max_features="sqrt", learning_rate=0.5)
        self.base_list['SDA'] = SGDClassifier(max_iter=1000, tol=1e-4)
        return self.base_list


if __name__ == '__main__':
    base = BaseModels()
    model_list = base.get_models()
    print(model_list)