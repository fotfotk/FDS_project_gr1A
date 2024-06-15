import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from IPython.display import display
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau

def evaluation_metrics(clf, y, X):

    y_test_pred    = clf.predict(X) #label predictions

    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()
    
    # evaluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tn + tp) / (tn + tp + fn + fp)
    recall      = tp / (tp + fn)
    f1          = 2*precision*recall/(precision+recall)
    
    y_test_predict_proba  = clf.predict_proba(X)
    y_test_predict_proba = y_test_predict_proba[:,1]
    fp_rates, tp_rates, _ =  roc_curve(y, y_test_predict_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    error_rate = (fn + fp) / (fn + fp + tn + tp)


    return [accuracy, precision, recall, specificity, f1, roc_auc, error_rate]


def test_model(X, y, ax, weight):
    # Preparation of performance overview for each split
    df_performance_k = pd.DataFrame(columns = ['accuracy','precision','recall',
                                            'specificity','F1','roc_auc', 'error_rate'])

# try model for k between 1 and 20. 
    for i in range(1, 21): 
        df_performance = pd.DataFrame(columns = ['fold','accuracy','precision','recall',
                                            'specificity','F1','roc_auc', 'error_rate'])
        
        fold = 0

        # Loop over all splits
        for train_index, test_index in skf.split(X,y):
        
            # Get the relevant subsets for training and testing
            X_test  = X.iloc[test_index]
            y_test  = y.iloc[test_index]
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]

            # Scale the features
            scaler = MinMaxScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            # Create prediction models and fit them to the training data
            neigh = KNeighborsClassifier(n_neighbors=i, weights=weight)
            # Fit the classifier with data
            neigh.fit(X_train_sc, y_train)

            # Evaluate the model
            eval_metrics = evaluation_metrics(neigh, y_test, X_test_sc)
            df_performance.loc[len(df_performance),:] = [fold]+eval_metrics

            # increase counter for folds
            fold += 1


        df_performance_k.loc[len(df_performance_k),:] = [np.mean(df_performance['accuracy']), np.mean(df_performance['precision']), np.mean(df_performance['recall']), np.mean(df_performance['specificity']), np.mean(df_performance['F1']), np.mean(df_performance['roc_auc']), np.mean(df_performance['error_rate'])]


    #print(df_performance_k)
    
    ax.plot('error_rate', data=df_performance_k)
    return df_performance_k

# Import data
df = pd.read_csv("./data/parkinsons.data")
display(df)

X = df.copy().drop(['status', 'name'], axis=1)
y = df['status']

# Split data into test and training sets with 20:80 split
X_train, X_test_f, y_train, y_test_f = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature importance
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_train)

# Determine the correlation of each feature and the label
corr = []

for i in range(0, X.shape[1]):
    res = kendalltau(X_sc[:,i], y_train)
    corr.append(res.statistic)

# Take absolute value to only have strength of correlation
corr_abs = np.abs(corr)

# Put correlation in a dataframe, add feature names and sort by descending values
df_correlation = pd.DataFrame([corr_abs], columns = X.columns).T.sort_values(by=0, ascending = False)
print(df_correlation)

# Make a plot with the correlation of each feature to the label
plt.plot(df_correlation)
plt.title("Feature selection (Kendall's Tau)")
plt.ylabel('statistic')
plt.xticks(rotation=90)
plt.vlines(x=12.5, ymin=0, ymax=0.5, colors='r')
plt.tight_layout()
plt.savefig('./output/knn/knn_feature_selection.png')

# Prepare datasets for different feature selections
X_kendalltau = X_train.copy().drop(['Shimmer:APQ3', 'Shimmer:DDA', 'HNR', 'D2', 'RPDE', 'MDVP:Fo(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Fhi(Hz)', 'DFA'], axis=1)
X_lasso = X_train[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']]


# Preparation of 5-fold crossvalidation and splitting
n_splits = 5
skf      = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=42)

# Determine the error rate for the different feature selections and distance weights
fig,axs = plt.subplots(2,3,figsize=(9,4), sharey=True, sharex=True)
plt.suptitle('Error rate for different feature selections and weights')
axs[0,0].set_title('All features', fontsize=10)
axs[0,1].set_title('Lasso selection', fontsize=10)
axs[0,2].set_title("Kendall's Tau selection", fontsize=10)
axs[0,0].set_ylabel('error rate', fontsize=8)
axs[1,0].set_ylabel('error rate', fontsize=8)
axs[1,0].set_xlabel('number of neighbors (k)', fontsize=8)
axs[1,1].set_xlabel('number of neighbors (k)', fontsize=8)
axs[1,2].set_xlabel('number of neighbors (k)', fontsize=8)
fig.text(0.04, 0.75, 'uniform weight', va='center', ha='center', rotation='vertical', fontsize=10)
fig.text(0.04, 0.25, 'distance weight', va='center', ha='center', rotation='vertical', fontsize=10)

test_model(X_train, y_train, ax=axs[0,0], weight='uniform')
test_model(X_lasso, y_train, ax=axs[0,1], weight='uniform')
test_model(X_kendalltau, y_train, ax=axs[0,2], weight='uniform')
test_model(X_train, y_train, ax=axs[1,0], weight='distance')
test_model(X_lasso, y_train, ax=axs[1,1], weight='distance')
test_model(X_kendalltau, y_train, ax=axs[1,2], weight='distance')
plt.savefig('./output/knn/knn_plot_all.png')

# Determine performance metrics for all features, using weighted distance
fig, ax = plt.subplots(1,1, figsize=(12,8))
plt.suptitle("All features, distance weight", fontsize=20)
ax.set_xlabel('number of neighbors (k)', fontsize=20)

df_performance_all = test_model(X=X_train, y=y_train, ax=ax, weight='distance')
ax.plot('accuracy', data=df_performance_all)
ax.plot('precision', data=df_performance_all)
ax.plot('recall', data=df_performance_all)
ax.plot('specificity', data=df_performance_all)
ax.plot('F1', data=df_performance_all)
ax.plot('roc_auc', data=df_performance_all)
ax.legend(fontsize=15)
plt.savefig('./output/knn/knn_plot_all_features_distance.png')

# Determine performance metrics for all features, using uniform distance
fig, ax = plt.subplots(1,1, figsize=(12,8))
plt.suptitle("All features, uniform weight", fontsize=20)
ax.set_xlabel('number of neighbors (k)', fontsize=20)

df_performance_all = test_model(X=X_train, y=y_train, ax=ax, weight='uniform')
ax.plot('accuracy', data=df_performance_all)
ax.plot('precision', data=df_performance_all)
ax.plot('recall', data=df_performance_all)
ax.plot('specificity', data=df_performance_all)
ax.plot('F1', data=df_performance_all)
ax.plot('roc_auc', data=df_performance_all)
ax.legend(fontsize=15)
plt.savefig('./output/knn/knn_plot_all_features_uniform.png')


# Evaluation dataframe
df_performance = pd.DataFrame(columns = ['model','accuracy','precision','recall',
                                            'specificity','F1','roc_auc', 'error_rate'])


# Evaluation of All features (train)
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_train)
X_test_f_sc  = scaler.transform(X_test_f)
neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_train, X_sc)
df_performance.loc[len(df_performance),:] = ['All features (train)']+eval_metrics


# Evaluation of Lasso selection (train)
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_lasso)
X_test_f_lasso = X_test_f[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']]
X_test_f_sc  = scaler.transform(X_test_f_lasso)
neigh = KNeighborsClassifier(n_neighbors=9, weights='distance')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_train, X_sc)
df_performance.loc[len(df_performance),:] = ['Lasso selection (train)']+eval_metrics


# Evaluation of Kendall selection (train)
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_kendalltau)
X_test_f_kendalltau = X_test_f.copy().drop(['Shimmer:APQ3', 'Shimmer:DDA', 'HNR', 'D2', 'RPDE', 'MDVP:Fo(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Fhi(Hz)', 'DFA'], axis=1)
X_test_f_sc  = scaler.transform(X_test_f_kendalltau)
neigh = KNeighborsClassifier(n_neighbors=9, weights='uniform')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_train, X_sc)
df_performance.loc[len(df_performance),:] = ['Kendall selection (train)']+eval_metrics

# Evaluation of All features (test)
# The number of neighbours is defined based on the error rate of figure 'plot_all'
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_train)
X_test_f_sc  = scaler.transform(X_test_f)
neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_test_f, X_test_f_sc)
df_performance.loc[len(df_performance),:] = ['All features (test)']+eval_metrics


# Evaluation of Lasso selection (test) 
# The number of neighbours is defined based on the error rate of figure 'plot_all'
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_lasso)
X_test_f_lasso = X_test_f[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']]
X_test_f_sc  = scaler.transform(X_test_f_lasso)
neigh = KNeighborsClassifier(n_neighbors=9, weights='distance')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_test_f, X_test_f_sc)
df_performance.loc[len(df_performance),:] = ['Lasso selection (test)']+eval_metrics


# Evaluation of Kendall selection (test)
# The number of neighbours is defined based on the error rate of figure 'plot_all'
scaler = MinMaxScaler()
X_sc = scaler.fit_transform(X_kendalltau)
X_test_f_kendalltau = X_test_f.copy().drop(['Shimmer:APQ3', 'Shimmer:DDA', 'HNR', 'D2', 'RPDE', 'MDVP:Fo(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Fhi(Hz)', 'DFA'], axis=1)
X_test_f_sc  = scaler.transform(X_test_f_kendalltau)
neigh = KNeighborsClassifier(n_neighbors=9, weights='uniform')
neigh.fit(X_sc, y_train)
eval_metrics = evaluation_metrics(neigh, y_test_f, X_test_f_sc)
df_performance.loc[len(df_performance),:] = ['Kendall selection (test)']+eval_metrics

# Convert all but the first columns into floats and round
df_performance.iloc[:, 1:] = df_performance.iloc[:, 1:].astype(float).round(3)
df_performance.to_csv('./output/knn/knn_performance.csv')



