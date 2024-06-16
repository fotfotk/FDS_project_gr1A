####################
# @author: Gui Basso
####################

# IMPORT PACKAGES
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

###---------------------USER DEFINED FUNCTIONS---------------------------###
# Utility function to plot the diagonal line
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# Function to complete evaluation metrics for Random Forest performance assessment
def evaluation_metrics(clf, y_true, X, legend_entry='my legendEntry'):
    y_pred = clf.predict(X)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # ROC Curve and area under the curve
    y_predict_proba = clf.predict_proba(X)[:, 1]
    fp_rates, tp_rates, _ = roc_curve(y_true, y_predict_proba)
    roc_auc = auc(fp_rates, tp_rates)

    plt.plot(fp_rates, tp_rates, label=legend_entry)

    return [accuracy, precision, recall, specificity, f1, roc_auc]

###---------------------------RANDOM FOREST MODEL--------------------------###
# IMPORT DATA
if __name__ == "__main__" :
    df = pd.read_csv("./data/parkinsons.data", index_col=0)
    # Take out target feature
    X = df.copy().drop('status', axis=1)
    y = df['status']

    # Initialize fold variable
    fold = 0
    # Create plot for ROC Curve
    fig, axs = plt.subplots(figsize=(6,5))

    # 5-fold stratified cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Performance overview data frame
    RF_performance = pd.DataFrame(columns=['fold', 'dataset', 'clf', 'accuracy', 'precision', 'recall', 'specificity', 'F1', 'roc_auc'])

    # Loop to apply Random Forest model to each split
    for train_index, test_index in kf.split(X, y):
        # Training and testing subsets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # RANDOM FOREST MODEL
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        # Evaluation of metrics using the function
        train_metrics = evaluation_metrics(rf, y_train, X_train, legend_entry=f'Train Fold {fold}')
        RF_performance.loc[len(RF_performance), :] = [fold, 'Train', 'RF'] + train_metrics

        test_metrics = evaluation_metrics(rf, y_test, X_test, legend_entry=f'Test Fold {fold}')
        RF_performance.loc[len(RF_performance), :] = [fold, 'Test', 'RF'] + test_metrics

        fold += 1

    ###---------------PLOTTING AND PERFORMANCE -----------------------###
    # ROC Curve Plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    add_identity(axs, color="r", ls="--", label='random\nclassifier')
    plt.title('ROC Curve - Random Forest with all features included')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output/randomforest/RF_roc_curves_without_lasso.png",bbox_inches='tight')

    # Performance output
    performance_summary = RF_performance.groupby(['dataset', 'clf']).agg(['mean', 'std'])
    print(performance_summary)

    # Feature Importance
    f_i = list(zip(X.columns, rf.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    imp = plt.figure(figsize=(7,5))
    
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance - Random Forest')
    plt.savefig("./output/randomforest/feature_importance_without_lasso.png",bbox_inches='tight')
