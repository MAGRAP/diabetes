import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter


def cleaning(diabetes):
    
    """ diabetes is a dataframe """
    
    # Primero quitamos las columnas que no queremos
    
    diabetes = diabetes.drop([
                            # Valores de segunda medida
                            "diagnosdm", "tiempoadm","fecha2","peso2","talla2","cintura2","ntratami2",
                            "X25oh2","urico2","crea2","colester2","triglice2","hdl2","ldl2","glucemia2",
                            "microalc2", "cistatin2","fibrinog2","pcr2","hbglicos2","insulina2","homa2",
                            "antiagr2","diureti2","betablo2","alfablo2","calcioa2","ieca2", "tiempo.censo",
                            "araii2","tas_s2","tad_s2","fc_s2","diferenciafechas", "imc2", "sm2", "hta2", 
                            "ncrit_sm2", "epi2", "dislipe2", "Unnamed: 0", "fechaglucometria", "fnacimien",
                            "fecha1", "nsagrado",
                              
                            # tratamientos
                            "dislipe1", "antiagr1", "diureti1", "betablo1", "alfablo1", "calcioa1", "ieca1", "araii1",
                            
                            # otros
                            "peso1", "colester1", "progres_microalc"
                              ], axis=1)
    
    # Nos aseguramos de que las columnas tengan sus tipos de clase correspondiente
    
    for key in diabetes.keys(): 
        if diabetes[key].dtype == object:    
            diabetes[key] = diabetes[key].astype(float)
            
        if diabetes[key].dtype == bool:    
            diabetes[key] = diabetes[key].astype(int)
            
    return diabetes

def complete_data(diabetes):
    
    "If value is Nan and column is hbglicos delete row"
    
    """    
    diabetes = diabetes.dropna(subset=['hbglicos1'])
    diabetes = diabetes.reset_index(drop=True)
    """
    
    "If value is Nan, input the average of the column"
    
    imputer = SimpleImputer(strategy="median")
    imputer.fit(diabetes)
    X = imputer.transform(diabetes)
    
    #Hay que volver a transformar a dataFrame, porque ahora diabetes es una matriz sin nombres 
    diabetes = pd.DataFrame(X, columns=diabetes.columns, index=diabetes.index)
    
    return diabetes

def standarize(data):
    
    "Use standarization to scale the data"
    
    standard_scaler = StandardScaler()
    standard_scaler.fit(data)
    X = standard_scaler.transform(data)
    
    # Hay que volver a transformar a dataFrame, porque ahora diabetes es una matriz sin nombres     
    data_scaled = pd.DataFrame(X, columns=data.columns, index=data.index)
    
    # Los valores booleanos se vuelven a meter, ya que no se cambian
    data_scaled[["sexo", "ecv", "diabete2"]] = data[["sexo", "ecv", "diabete2"]]
      
    return data_scaled

def get_features_lables(df):
    
    "Separate the features and labels from the database"
    
    features = df.drop(['diabete2'], axis=1)
    labels = df["diabete2"]
    
    return features, labels


def train_and_test(X, y, grid_model, n_iterations):
    
    """ 
    Divide the data in different train and test splits,
    search for each split a model with grid search,
    compute the metrics for each split and return the average metric
    return also the best model
    """
    
    recall = 0
    precision = 0
    accuracy = 0
    auc = 0
    
    sss = StratifiedShuffleSplit(n_splits=n_iterations, test_size=0.1, random_state=3)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        # Fit the grid to the train data for this iteration
        grid.fit(X_train, y_train)

        # Get the best estimator
        clf = grid.best_estimator_
        
        # Compute the predicitions
        grid_predictions = grid.predict(X_test)
        y_pred = grid_predictions
        
        # Compute and save metrics
        precision += precision_score(y_test, y_pred)
        recall += recall_score(y_test, y_pred)
        accuracy += accuracy_score(y_test, y_pred)
        auc += roc_auc_score(y_test, y_pred)
        
    precision = precision / n_iterations 
    accuracy = accuracy / n_iterations
    recall = recall / n_iterations
    auc = auc / n_iterations
    
    return clf, accuracy, precision, recall, auc

def compute_confusion_matrix(acc, prec, recall, length):
    
    """Compute the confusion matrix from a recall, precision and accuracy"""
    
    a = np.array([
        [1 - prec, - prec, 0, 0],
        [1 - recall, 0, - recall, 0],
        [1 - acc, - acc, - acc, 1 - acc],
        [1, 1, 1, 1]
    ])

    b = np.array([0, 0, 0, length])
    
    # Solve the linear problem
    x = np.linalg.solve(a, b)
    
    tp, fp, fn, tn = x[0], x[1], x[2], x[3]
    tp, fp, fn, tn = round(tp), round(fp), round(fn), round(tn)
    
    print("\nconfusion matrix")
    print("[tp, fn ]")
    print("[fp, tn ]")
    print("\n[{} {}]".format(tp,fn))
    print("[{} {}]".format(fp,tn))

# Load Data
""" 
LOAD CLEAN DATA
"""

diabetes_orig = pd.read_csv("diabetes.csv", decimal=',')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 3000)

# Eliminamos las columnas no necesarias y fijamos los tipos de clases para cada una
diabetes = cleaning(diabetes_orig.copy())

# Completamos los datos faltantes
diabetes = complete_data(diabetes)

diabetes_scaled = diabetes.copy()
diabetes_scaled = standarize(diabetes_scaled)

# Separar diabetes y diabetes scaled en features y labels

X, y = get_features_lables(diabetes)
X_scaled, y_scaled = get_features_lables(diabetes_scaled)

from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.svm import LinearSVC
import seaborn as sns

# Calculate the correlation matrix and take the absolute value
corr_matrix = X_scaled.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)

# List column names of highly correlated features (r > 0.9)
to_drop = list()

for col in tri_df.keys():
    for row in tri_df.keys():
        if tri_df[col][row] > 0.9:
            to_drop.append((col, row))
        
for row in to_drop:
    print(row)

columns_to_drop = ['homa1', 'gluc_media', 'permentrop', 'conga2', 'apen']

# Drop the features in the to_drop list
X = X.drop(['homa1', 'gluc_media', 'permentrop', 'conga2', 'apen'], axis=1)
X_scaled = X_scaled.drop(['homa1', 'gluc_media', 'permentrop', 'conga2', 'apen'], axis=1)

mutual_info = dict(zip(X.columns,
                    mutual_info_classif(X, y, n_neighbors = 3, random_state = 21)
                    ))

mutual_info_ordered = dict(sorted(mutual_info.items(),  key=lambda x: x[1], reverse=True))

selector = RFE(LinearSVC(), n_features_to_select = 1, step=1)
selector = selector.fit(X_scaled, y_scaled)

rfe_selection = dict(zip(X_scaled.columns,
                    selector.ranking_
                    ))
rfe_selection_ordered = dict(sorted(rfe_selection.items(),  key=lambda x: x[1], reverse=False))

importance_values = dict()

for key in rfe_selection_ordered.keys():
    importance_values[key] = mutual_info_ordered[key] / rfe_selection_ordered[key] 

importance_values_ordered = dict(sorted(importance_values.items(),  key=lambda x: x[1], reverse=True))

list_importance_values = list()

for col in importance_values_ordered.keys():
    list_importance_values.append(col)

top = 5

X = X[list_importance_values[:top]]
X_scaled = X_scaled[list_importance_values[:top]]

import imblearn
print(imblearn.__version__)

from imblearn.over_sampling import SMOTE

# transform the dataset

oversample = SMOTE(sampling_strategy=0.2, k_neighbors = 5)
X, y = oversample.fit_resample(X, y)
X_scaled, y_scaled = oversample.fit_resample(X_scaled, y_scaled)

# summarize class distribution
counter = Counter(y)
print(counter)

from sklearn.svm import SVC

# defining parameter range

"""param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1],
              'kernel': ['rbf', 'sigmoid', 'linear', 'poly']
            }"""

param_grid = {'C': [10, 100],
              'gamma': [0.01, 0.1],
              'kernel': ['rbf', 'sigmoid']
            }
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1, scoring = 'f1', cv=10)

svm, acc, prec, recall, auc = train_and_test(X_scaled, y_scaled, grid, 10)

print("BEST CLASSIFIER :", svm)
print("\naccuracy: {:.2f}".format(acc))
print("precision: {:.2f}".format(prec))
print("recall: {:.2f}".format(recall))
print("auc: {:.2f}".format(auc))

compute_confusion_matrix(acc, prec, recall, len(y))

from sklearn.ensemble import RandomForestClassifier

# defining parameter range

param_grid = {'n_estimators': [400],
              'criterion': ["gini"],
              'max_depth': [10]
            }
 
grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 10, scoring = 'f1', cv=10)

rf, acc, prec, recall, auc = train_and_test(X, y, grid, 10)

print("BEST CLASSIFIER :", rf)
print("\naccuracy: {:.2f}".format(acc))
print("precision: {:.2f}".format(prec))
print("recall: {:.2f}".format(recall))
print("auc: {:.2f}".format(auc))

compute_confusion_matrix(acc, prec, recall, len(y))