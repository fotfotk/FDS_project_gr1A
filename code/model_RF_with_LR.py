#######
# @author: Gui Bassso
#######

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


# Import data
if __name__ == "__main__" :
    df = pd.read_csv("./data/parkinsons.data",index_col=0)
    X  = df.copy().drop('status', axis = 1)
    y  = df['status']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a Lasso Regression model
    lasso_model = Lasso(alpha=0.1)  # Alpha is the regularization strength

    # Training the model
    lasso_model.fit(X_train, y_train)

    # Making predictions on the test set
    predictions = lasso_model.predict(X_test)

    # Calculating the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

    # Printing the coefficients of the model
    #print("Coefficients:", lasso_model.coef_)

    coefficients = lasso_model.coef_
    feature_names = df.columns.tolist()

    # Printing coefficients along with feature names
    #for feature, coef in zip(feature_names, coefficients):
        #print(feature, ":", coef)

    # Identifying the most important features
    important_features = [feature for feature, coef in zip(feature_names, coefficients) if coef != 0]
    print("Important Features:", important_features)

    # Assuming df is your DataFrame containing the features used in the model

    # Analyzing Coefficients
    coefficients = lasso_model.coef_
    feature_names = df.columns.tolist()  # Extracting feature names from DataFrame columns

    # Filtering out features with non-zero coefficients
    non_zero_features_indices = [i for i, coef in enumerate(coefficients) if coef != 0]
    selected_features = [feature_names[i] for i in non_zero_features_indices]

    # Creating a new DataFrame with selected features
    df_selected = df[selected_features]

    ###---------------------------RANDOM FOREST MODEL--------------------------###
    #IMPORT DATA
    X  = df_selected.copy()
    y  = df['status']

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
    plt.title('ROC Curve - Random Forest with Lasso Regression Feature Selection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./output/randomforest/RF_LR_roc_curves.png',bbox_inches='tight')

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
    plt.savefig('./output/randomforest/feature_importance_with_lasso.png',bbox_inches='tight')
