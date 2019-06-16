import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
def classifier_compare(X, y, clf_1, clf_2, clf_3, split_size):
    def draw_heatmap_linear(acc, acc_desc, C_list, parameter):
        plt.figure(figsize = (2,4))
        ax = sns.heatmap(acc, annot=True, fmt='.3f', yticklabels = C_list, xticklabels=[])
        ax.collections[0].colorbar.set_label("accuracy")
        ax.set(ylabel=parameter)
        plt.title(acc_desc + ' w.r.t '+ parameter)
        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.show()
    
    _df = pd.DataFrame(index = [type(clf_1[0]).__name__, type(clf_2[0]).__name__, type(clf_3[0]).__name__])
    
    partitions = [str(int(i * 100)) + '/' + str(int(100 - i * 100)) for i in split_size]
    
    clf_1_final = {i : {} for i in partitions}
    clf_2_final = {i : {} for i in partitions}
    clf_3_final = {i : {} for i in partitions}
    
    for i in split_size:
        key = str(int(i * 100)) + '/' + str(int(100 - i * 100))
        clf_1_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        clf_2_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        clf_3_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        training_acc_clf_1, validation_acc_clf_1, testing_acc_clf_1 = [], [], []
        training_acc_clf_2, validation_acc_clf_2, testing_acc_clf_2 = [], [], []
        training_acc_clf_3, validation_acc_clf_3, testing_acc_clf_3 = [], [], []
        best_param_1, best_param_2, best_param_3= [], [], []
        for rs in range(3):
            # split data
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = i, random_state = rs)

            # cv on the data
            clf = GridSearchCV(clf_1[0], clf_1[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_1.append(train_scores[clf.best_index_])
            validation_acc_clf_1.append(val_scores[clf.best_index_])
            best_param_1.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_1.append(accuracy_score(Y_test, Y_pred))
            # draw heatmap
            draw_heatmap_linear(train_scores.reshape(-1,1), \
                                type(clf_1[0]).__name__ + "train accuracy", \
                                list(clf_1[1].values())[0],\
                                list(clf_1[1].keys())[0])
            draw_heatmap_linear(val_scores.reshape(-1,1), \
                                type(clf_1[0]).__name__ + "val accuracy",\
                                list(clf_1[1].values())[0],\
                                list(clf_1[1].keys())[0])


            clf = GridSearchCV(clf_2[0], clf_2[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_2.append(train_scores[clf.best_index_])
            validation_acc_clf_2.append(val_scores[clf.best_index_])
            best_param_2.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_2.append(accuracy_score(Y_test, Y_pred))
            # draw heatmap
            draw_heatmap_linear(train_scores.reshape(-1,1), \
                                type(clf_2[0]).__name__ + "train accuracy", \
                                list(clf_2[1].values())[0], \
                                list(clf_2[1].keys())[0])
            draw_heatmap_linear(val_scores.reshape(-1,1), \
                                type(clf_2[0]).__name__ + "val accuracy", \
                                list(clf_2[1].values())[0], \
                                list(clf_2[1].keys())[0])

            # decision tree part

            # cv on the data
            clf = GridSearchCV(clf_3[0], clf_3[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_3.append(train_scores[clf.best_index_])
            validation_acc_clf_3.append(val_scores[clf.best_index_])
            best_param_3.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_3.append(accuracy_score(Y_test, Y_pred))
            # draw heatmap
            draw_heatmap_linear(train_scores.reshape(-1,1), \
                                type(clf_3[0]).__name__ + "train accuracy", \
                                list(clf_3[1].values())[0], \
                                list(clf_3[1].keys())[0])
            draw_heatmap_linear(val_scores.reshape(-1,1), \
                                type(clf_3[0]).__name__ + "val accuracy", \
                                list(clf_3[1].values())[0], \
                                list(clf_3[1].keys())[0])
            clf_1_result['train'] = training_acc_clf_1
            clf_1_result['validation'] = validation_acc_clf_1
            clf_1_result['test'] = testing_acc_clf_1
            clf_1_result['best_params'] = best_param_1
            
            clf_2_result['train'] = training_acc_clf_2
            clf_2_result['validation'] = validation_acc_clf_2
            clf_2_result['test'] = testing_acc_clf_2
            clf_2_result['best_params'] = best_param_2
            
            clf_3_result['train'] = training_acc_clf_3
            clf_3_result['validation'] = validation_acc_clf_3
            clf_3_result['test'] = testing_acc_clf_3
            clf_3_result['best_params'] = best_param_3
            
            clf_1_final[key] = clf_1_result
            clf_2_final[key] = clf_2_result
            clf_3_final[key] = clf_3_result
    
    return {type(clf_1[0]).__name__: clf_1_final,
            type(clf_2[0]).__name__: clf_2_final, 
            type(clf_3[0]).__name__: clf_3_final}
def generate_results(output):
    df = pd.Series(output).apply(lambda x: pd.DataFrame(x).T)
    out = df.iloc[0].append(df.iloc[1]).append(df.iloc[2])
    new_ind = pd.Series([i for i in out.index]) + pd.Series([
        ' RF', ' RF', ' RF',
        ' LOGREG ', ' LOGREG ', ' LOGREG ',
        ' BST-DT ', ' BST-DT ', ' BST-DT '
    ])
    out.index = new_ind
    out['best_params'] = out['best_params']\
    .apply(lambda x: {list(x[0].keys())[0] : pd.Series([list(i.values())[0] for i in x]).unique().tolist()})
    out['mean_train_acc'] = out['train'].apply(lambda x: round(np.mean(x),4))
    out['mean_validation_acc'] = out['validation'].apply(lambda x: round(np.mean(x),4))
    out['mean_test_acc'] = out['test'].apply(lambda x: round(np.mean(x),4))
    out['train_acc_var'] = out['train'].apply(lambda x: round(np.std(x)**2,4))
    out['test_acc_var'] = out['test'].apply(lambda x: round(np.std(x)**2,4))
    out = out[['mean_train_acc','mean_validation_acc', 'mean_test_acc',\
         'train_acc_var', 'test_acc_var', 'best_params']]
    return out


def classifier_compare_var(X, y, clf_1, clf_2, clf_3, split_size):
    _df = pd.DataFrame(index = [type(clf_1[0]).__name__, type(clf_2[0]).__name__, type(clf_3[0]).__name__])
    
    partitions = [str(int(i * 100)) + '/' + str(int(100 - i * 100)) for i in split_size]
    
    clf_1_final = {i : {} for i in partitions}
    clf_2_final = {i : {} for i in partitions}
    clf_3_final = {i : {} for i in partitions}
    
    for i in split_size:
        key = str(int(i * 100)) + '/' + str(int(100 - i * 100))
        clf_1_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        clf_2_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        clf_3_result = {'train': [], 'validation': [], 
                        'test' : [],'best_params': []}
        training_acc_clf_1, validation_acc_clf_1, testing_acc_clf_1 = [], [], []
        training_acc_clf_2, validation_acc_clf_2, testing_acc_clf_2 = [], [], []
        training_acc_clf_3, validation_acc_clf_3, testing_acc_clf_3 = [], [], []
        best_param_1, best_param_2, best_param_3= [], [], []
        for rs in range(3):
            # split data
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = i, random_state = rs)

            # cv on the data
            clf = GridSearchCV(clf_1[0], clf_1[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_1.append(train_scores[clf.best_index_])
            validation_acc_clf_1.append(val_scores[clf.best_index_])
            best_param_1.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_1.append(accuracy_score(Y_test, Y_pred))


            clf = GridSearchCV(clf_2[0], clf_2[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_2.append(train_scores[clf.best_index_])
            validation_acc_clf_2.append(val_scores[clf.best_index_])
            best_param_2.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_2.append(accuracy_score(Y_test, Y_pred))

            # decision tree part

            # cv on the data
            clf = GridSearchCV(clf_3[0], clf_3[1], return_train_score= True, cv = 5, n_jobs= -1)
            clf.fit(X_train, Y_train)
            train_scores = clf.cv_results_['mean_train_score']
            val_scores = clf.cv_results_['mean_test_score']
            # append trainnig validation and testing accuarcy
            training_acc_clf_3.append(train_scores[clf.best_index_])
            validation_acc_clf_3.append(val_scores[clf.best_index_])
            best_param_3.append(clf.best_params_)
            Y_pred = clf.predict(X_test)
            testing_acc_clf_3.append(accuracy_score(Y_test, Y_pred))
            clf_1_result['train'] = training_acc_clf_1
            clf_1_result['validation'] = validation_acc_clf_1
            clf_1_result['test'] = testing_acc_clf_1
            clf_1_result['best_params'] = best_param_1
            
            clf_2_result['train'] = training_acc_clf_2
            clf_2_result['validation'] = validation_acc_clf_2
            clf_2_result['test'] = testing_acc_clf_2
            clf_2_result['best_params'] = best_param_2
            
            clf_3_result['train'] = training_acc_clf_3
            clf_3_result['validation'] = validation_acc_clf_3
            clf_3_result['test'] = testing_acc_clf_3
            clf_3_result['best_params'] = best_param_3
            
            clf_1_final[key] = clf_1_result
            clf_2_final[key] = clf_2_result
            clf_3_final[key] = clf_3_result
    
    return {type(clf_1[0]).__name__: clf_1_final,
            type(clf_2[0]).__name__: clf_2_final, 
            type(clf_3[0]).__name__: clf_3_final}