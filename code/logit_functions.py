# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:36:19 2024

@author: Fotini Kyriazi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import RandomOverSampler

def sigmoid(x):
    ''' Returns the result of a sigmoid function '''
    return 1/(1 + np.exp(-x))

def create_plot(plot_type, ax, xdata, datay=None):
    '''
    Helper function to define plot types.
    Parameters:
        plot_type (str): the desired plot type. Can be: histplot, scatterplot, regplot or qqplot.
        ax (Axes object): the ax to plot on.
        xdata (array): the data to plot on x axis.
        datay (array, optional): what to plot on the y axis. Default is None.
    Output: 
        Plots the corresponding plot.
    '''
    if plot_type == 'histplot':
        sns.histplot(xdata, ax=ax)
    elif plot_type == 'scatterplot' and datay is not None:
        sns.scatterplot(x=xdata, y=datay, ax=ax)
    elif plot_type == 'regplot' and datay is not None:
        sns.regplot(x=xdata, y=datay, logistic=True, ax=ax)
    elif plot_type == 'qqplot':
        sm.qqplot(xdata, line='q', ax=ax)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def generate_plots(plot_type, path, colnames, labelx, labely, title, figsize, datax, datay=None):
    ''' 
    Helper function to create individual plots based on type. 
    Parameters:
        plot_type (str): the desired plot type. Can be: histplot, scatterplot, regplot or qqplot.
        datax (DataFrame): the data to be plotted on the x axis.
        datay (1D array, optional): the data to be plotted on the y axis. Default is None.
        colnames (list): list of column names, eg. X.columns. 
        labelx, labely (str): the x,y axis label.
        path (str): the path for saving the figure.
        title (str): the title of the figure.
        figsize (tuple): figure dimensions.    
    Returns:
        None
    '''
    # Ensure that datax is a dataframe
    if not isinstance(datax, pd.DataFrame):
        datax = pd.DataFrame(datax, columns=colnames)

    # First figure with a 3x4 grid
    fig1, axes1 = plt.subplots(3, 4, figsize=figsize)
    axes1 = axes1.flatten()
    for i in range(12):
        create_plot(plot_type, axes1[i], datax.iloc[:, i], datay)
        axes1[i].set(title=colnames[i], xlabel=labelx, ylabel=labely)
    fig1.tight_layout()
    fig1.suptitle(title+' (1/2)', y=1.02, fontsize=15)
    fig1.savefig(path+'_part1.png', bbox_inches='tight', dpi=400)

    # Second figure with 2 rows of 4 and one row of 2
    fig2, axes2 = plt.subplots(3, 4, figsize=figsize)
    # Remove the last two axes to create a 2x4 and 1x2 grid
    fig2.delaxes(axes2[2, 2])
    fig2.delaxes(axes2[2, 3])
    axes2 = axes2.flatten()

    for i in range(12, 22):
        create_plot(plot_type, axes2[i-12], datax.iloc[:,i], datay)
        axes2[i-12].set(title=colnames[i], xlabel=labelx, ylabel=labely)

    fig2.tight_layout()
    fig2.suptitle(title+' (2/2)', y=1.02, fontsize=15)
    fig2.savefig(path+'_part2.png', bbox_inches='tight')


def calc_vif(data,features):
    ''' 
    Helper function to calculate the variance inflation factor (VIF).
    Parameters:
        data (2-D array): contains the data for which to calculate the VIF.
        features (list or DataFrame Index object): the feature names.
    Returns:
        vif_df: the dataframe of VIF for all features.
  '''
    col_vif = [] # placeholder to store the vif values
    for i in range(len(features)):
        vif = variance_inflation_factor(data,i)
        col = features[i]
        col_vif.append((col,vif))
    vif_df = pd.DataFrame(data=col_vif,columns=['Feature','VIF'])
    return vif_df


def calc_vif_thresh(data, features, threshold=15):
    '''
    Selects the best features based on Variance Inflation Factor (VIF).
    Parameters:
        data (2-D array): The input array containing the (scaled and transformed) data.
        features (list): List of feature names, for which the VIF is to be computed.
        threshold (float): The threshold value for VIF. Features with VIF greater than this threshold will be removed.
    Returns:
        selected_features (Index): Index containing the selected features.
        perf (DataFrame): DataFrame containing the VIF values for the features below the threshold.
    '''
    selected_features = features.copy()
    removed_features = []
    removed_feat_vif = []  # placeholder to hold the removed feature names and their VIF while removing
    perf = calc_vif(data, selected_features).sort_values(by=['VIF'], ascending=False)
    while any(perf['VIF'] > threshold):
        worst_vif_val = perf['VIF'].max()
        worst_column = perf.loc[perf['VIF'] == worst_vif_val, 'Feature'].to_list()
        removed_feat_vif.append([worst_column, worst_vif_val])
        removed_features.extend(worst_column)
        selected_features = selected_features.drop(worst_column)
        perf = calc_vif(data, selected_features).sort_values(by=['VIF'], ascending=False)
    return selected_features, perf, removed_features, removed_feat_vif


def draw_roc_curve(figsize, n_splits, fpr_folds, tpr_folds, tprs_mean, auc_folds, title):
    '''
    Helper function to draw an ROC curve.
    Parameters:
        figsize (tuple): the figure dimensions in inches.
        n_splits (int): the number of folds in k-fold cross validation.
        fpr_folds (array): the false positive rate for each fold.
        tpr_folds (array): the true positive rate for each fold.
        tprs_mean (array): the mean true positive rate for each fold.
        auc_folds (array): the area under the curve for each fold.
        title (str): the figure title.
    Returns:
        fig (Matplotlib figure): a figure object.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    # Draw the curves of each fold:
    for i in range(n_splits):
        ax.step(fpr_folds[i], tpr_folds[i], label='ROC fold %d (AUC = %0.2f)' % (i, auc_folds[i]))
    
    # Plot the mean curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs_mean, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_folds)
    ax.step(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Chance level (AUC = 0.50)', lw=2)
    std_tpr = np.std(tprs_mean, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1,label=r'$\pm$ 1 std. dev.')
    ax.set(ylim = [-0.05, 1.05],xlabel="False Positive Rate",ylabel="True Positive Rate",title=title)
    ax.legend(loc="lower right")

    return fig

def draw_pr_curve(figsize, n_splits, recall_folds, precision_folds, avg_pr, y_real_concat, y_proba_concat, title):
    '''
    Helper function to draw an PR curve.
    Parameters:
        figsize (tuple): the figure dimensions in inches.
        n_splits (int): the number of folds in k-fold cross validation.
        recall_folds (array): the recall rate for each fold.
        precision_folds (array): the precision rate for each fold.
        avg_pr (array): the average precision for each fold.
        y_real_concat (array): the true (concatenated) label values.
        y_proba_concat (array): the concatenated probabilities of the label y.
        title (str): the figure title.
    Returns:
        fig (Matplotlib figure): a figure object.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    # Draw the curves for each fold
    for i in range(n_splits):
        ax.step(recall_folds[i], precision_folds[i], label='PR fold %d (AP = %.2f)' % (i, avg_pr[i]))
    
    # Plot the mean curve
    precision, recall, _ = precision_recall_curve(y_real_concat, y_proba_concat)
    std_precision = np.std(precision, axis=0)
    pr_upper = precision + std_precision
    pr_lower = precision - std_precision
    lab = 'Mean PR (AP=%.2f $\pm$ %0.2f)' % (average_precision_score(y_real_concat, y_proba_concat), std_precision)
    ax.step(recall, precision, label=lab, lw=2, color='blue')
    chance_level = np.sum(y_real_concat) / len(y_real_concat)
    ax.plot([0, 1], [chance_level, chance_level], linestyle='--', color='black', label='Chance level (PR = %.2f)'%(chance_level), lw=2)
    ax.fill_between(recall,pr_lower,pr_upper,color="grey",alpha=0.1,label=r"$\pm$ 1 std. dev.")
    ax.set(ylim = [-0.05, 1.05],xlabel="Recall",ylabel="Precision",title=title)
    ax.legend(loc="lower right")
    
    return fig

def calculate_metrics(y_test, y_pred, y_score):
    '''
    Helper function to calculate performance metrics.
    Parameters:
        y_test: the true label.
        y_pred: the predicted label.
        y_score: the probability of the predicted label.
    Returns:
        accuracy, recall, specificity, F1, ROC AUC as a list.
    '''
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0) # get the recall for the negative class
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    return [acc,precision,recall,specificity,f1,roc_auc]

def logit_with_cv(X, y, n_splits, num_inner_cv, skf, hp_params, max_iter, roctitle, prtitle, figsize=(10,7), random_state=2024, oversampling=False):
    '''
    Function to perform logistic regression with k-fold cross validation.
    Parameters:
        X (DataFrame): the data to train the model on.
        y (Series): the label of the data.
        n_splits (int): the number of splits for k-fold nested cross validation.
        num_inner_cv (int): the number of inner splits. 
        skf: the classifier, in this case stratified.
        hp_params (list): the defined hyperparameters to tune for.
        max_iter (int): the number of iterations.
        roctitle (str): the title for the ROC figure.
        prtitle (str): the title for the PR figure.
        figsize (tuple): the figure size, default is 10,7 inches.
        random_state (int): random seed, defaults to 2024.
        oversampling (bool): whether to apply random oversampling, defaults to False
    Returns:
        df_perf (DataFrame): the performance metric for the model.
        df_coef_norm (DataFrame): the normalised feature coefficients.
        df_coeff_summary (DataFrame): the mean normalised feature coefficients per fold.
        best_params (DataFrame): the best hyperparameters in each fold.
        figroc, figpr (Matplotlib figures): the ROC and PR figures.
    '''
    # Prepare dataframes to store coefficients and performance
    df_perf = pd.DataFrame(columns = ['accuracy','precision','recall','specificity','F1','roc_auc']).astype('float64')
    df_coef_norm = pd.DataFrame(index = X.columns, columns = np.arange(n_splits)).astype('float64')
    arr_intercept = np.zeros(shape=(n_splits,))
    best_params_cv = pd.DataFrame(index = hp_params.keys(), columns = np.arange(n_splits))

    # Prepare lists for ROC figure
    tprs_mean = []
    fpr_folds = []
    tpr_folds = []
    auc_folds = []
    mean_fpr = np.linspace(0, 1, 100)

    # Prepare lists for precision-recall figure
    avg_pr = []
    precision_folds = []
    recall_folds = []
    y_real = []
    y_proba = []

    fold = 0
    for train_idx, test_idx in skf.split(X,y):

        # Get the relevant subsets for training and testing
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx]
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # Random oversampling
        if oversampling==True:
            ros = RandomOverSampler(random_state=random_state)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            
        # Standardize the numerical features - NON-GAUSSIAN DATA!
        minmax = MinMaxScaler()
        X_train_sc = minmax.fit_transform(X_train)
        X_test_sc = minmax.transform(X_test)
        
        # Transform the data with Yeo-Johnson
        yj = PowerTransformer() # defaults to YJ
        X_train_transf = yj.fit_transform(X_train_sc)
        X_test_transf = yj.transform(X_test_sc)

        # Create prediction models and fit them to the training data
        # Logistic regression
        logitclf = LogisticRegression(solver='saga', penalty=None, max_iter=max_iter, random_state = random_state)
        
        # HP tuning within inner CV loop
        GS_clf = GridSearchCV(logitclf, hp_params, cv=num_inner_cv)
        
        # Fit the classifier with data
        GS_clf.fit(X_train_transf, y_train) 
        
        # Best parameters in each fold:
        best_params_cv.iloc[0,fold] = GS_clf.best_estimator_.get_params()['penalty']
        best_params_cv.iloc[1,fold] = GS_clf.best_estimator_.get_params()['C']
         
        # Get the coefficients
        df_LR_coefs = pd.DataFrame(zip(X_train.columns, np.transpose(GS_clf.best_estimator_.coef_[0,:])), columns=['features', 'coef'])
        df_coef_norm.loc[:,fold] = df_LR_coefs['coef'].values/df_LR_coefs['coef'].abs().sum()

        # Get the intercept per fold
        arr_intercept[fold] = GS_clf.best_estimator_.intercept_

        # Prediction
        y_pred = GS_clf.best_estimator_.predict(X_test_transf)
        y_score = GS_clf.best_estimator_.predict_proba(X_test_transf)[:,1] # positive prediction

        # Get the metrics
        df_perf.loc[fold] = calculate_metrics(y_test, y_pred, y_score)    

        # ROC figure
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fpr_folds.append(fpr)
        tpr_folds.append(tpr)
        # Interpolate at mean_fpr given discrete points (fpr, tpr)
        tprs_mean.append(np.interp(mean_fpr, fpr, tpr))
        tprs_mean[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_folds.append(roc_auc)
        
        # Precision recall figure
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        precision_folds.append(precision)
        recall_folds.append(recall)
        avg_pr.append(average_precision_score(y_test, y_score))
        y_real.append(y_test)
        y_proba.append(y_score)
        
        fold += 1
    
    # Unfortunately, some parameters might be equally frequent. The first column of the resulting matrix is taken to somehow make a selection.
    best_params = best_params_cv.mode(axis=1)[0]
    model_intercept = np.mean(arr_intercept)
    y_real_concat = np.concatenate(y_real)
    y_proba_concat = np.concatenate(y_proba)
    df_coeff_summary = pd.concat([df_coef_norm.mean(axis=1),df_coef_norm.std(axis=1)],axis=1).rename({0:'mean',1:'sd'},axis='columns')
    figroc = draw_roc_curve(figsize,n_splits,fpr_folds,tpr_folds,tprs_mean,auc_folds,roctitle)
    figpr = draw_pr_curve(figsize,n_splits,recall_folds,precision_folds,avg_pr,y_real_concat,y_proba_concat,prtitle)

    return df_perf, df_coef_norm, df_coeff_summary, model_intercept, best_params, figroc, figpr