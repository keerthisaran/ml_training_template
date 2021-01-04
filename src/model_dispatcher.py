import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models={'xgb':xgb.XGBClassifier(n_jobs=2,
                                max_depth=7,
                                n_estimators=200),
        'rf':RandomForestClassifier(n_estimators=100)}
