import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_validate
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)

df = pd.read_excel("proje_data.xlsx")
df.columns = df.columns.str.strip()
df.columns
df.head()

# Verinin düzenlenmesi
df.drop("Unnamed: 0", axis=1, inplace=True)
df.head()

df.to_csv("data.csv")

# Check Dataframe
def check_df(dataframe, count=5):
    print("#" * 30, "Head", "#" * 30)
    print(dataframe.head(count))
    print("#" * 30, "Tail", "#" * 30)
    print(dataframe.tail(count))
    print("#"*30, "Shape", "#"*30)
    print(dataframe.shape)
    print("#" * 30, "Information", "#" * 30)
    print(dataframe.info())
    print("#" * 30, "Describe", "#" * 30)
    print(dataframe.describe().T)
    print("#" * 30, "NA", "#" * 30)
    print(dataframe.isnull().sum())

check_df(df)

df.head()

# Label Encoding
le = LabelEncoder()
df["Cinsiyet"] = le.fit_transform(df["Cinsiyet"]) # 0 => Erkek - 1 => Kadın
df["İnfluencer olma istegi"] = le.fit_transform(df["İnfluencer olma istegi"]) # 0 => Evet - 1 => Hayır
df["Kendinizi yalnız hissediyor musunuz"] = le.fit_transform(df["Kendinizi yalnız hissediyor musunuz"]) # 0 => Evet - 1 => Hayır

# one hot
categorical_columns = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_columns)

##################################
# Model & Prediction
##################################

# Logistic Regression

y = df["Kendinizi yalnız hissediyor musunuz"]
X = df.drop(columns=["Kendinizi yalnız hissediyor musunuz"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_ # 1.45456067
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

# Model Evaluation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Accuary Score: {0}".format(acc), size= 10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))

# Accuracy: 0.80
# Precision: 0.82
# Recall: 0.92
# f1-Score: 0.87

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.8481249999999999

# Model Validation: Holdout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.70, random_state=45
)

log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.82
# Recall: 0.87
# f1-score: 0.84

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='LogisticRegression (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Cross Validation
y = df["Kendinizi yalnız hissediyor musunuz"]
X = df.drop(columns=["Kendinizi yalnız hissediyor musunuz"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_result = cross_validate(log_model,
                           X, y,
                           cv= 10,
                           scoring= ["accuracy", "precision", "recall", "f1", "roc_auc"])

log_model_accuracy_value = cv_result["test_accuracy"].mean()
# 0.7739130434782608

cv_result["test_precision"].mean()
# 0.8032576539387686

cv_result["test_recall"].mean()
# 0.89375

log_f1_value = cv_result["test_f1"].mean()
# 0.84439711564

cv_result["test_roc_auc"].mean()
# 0.79375

# Prediction for a New Observation

X.columns

random_user = X.sample(1)
log_model.predict(random_user)

X.columns
print(X_train.columns)

y
# Save Model
#joblib.dump(log_model, "logistic_model.pkl")


################## Light GBM ##################

lgbm_model = LGBMClassifier(random_state=17, verbose=-1)
lgbm_model.get_params()

cv_result_lgbm = cross_validate(lgbm_model, X, y, cv= 10, scoring= ["accuracy", "f1", "roc_auc"])

cv_result_lgbm["test_accuracy"].mean() # 0.7347
cv_result_lgbm["test_f1"].mean() # 0.8136
cv_result_lgbm["test_roc_auc"].mean() # 0.7294

lgbm_params = {
    "learning_rate": [0.01, 0.1, 0.001],
    "n_estimators": [100, 300, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_result_lgbm = cross_validate(lgbm_final, X, y, cv= 10, scoring=["accuracy", "f1", "roc_auc"])

lgbm_model_accuracy_value = cv_result_lgbm["test_accuracy"].mean() # 0.7782
lgbm_f1_value = cv_result_lgbm["test_f1"].mean() # 0.8557
cv_result_lgbm["test_roc_auc"].mean() # 0.7870

#joblib.dump(lgbm_final, "lgbm_model.pkl")

