import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

df = pd.read_csv('data.csv')

df['screen_time'] = (
    1.2 * (8 - df['sleep_hours']) +
    2.0 * (1 / (df['daily_steps'] + 1)) +
    0.8 * (df['resting_hr'] / 100) +
    np.random.uniform(-0.5, 0.5, len(df))
)
df['screen_time'] = df['screen_time'].clip(0, 12)

df['mood_score'] = (
    0.6 * df['sleep_hours'] +
    0.002 * df['daily_steps'] +
    0.5 * df['water_intake_l'] -
    0.001 * df['calories_consumed'] +
    np.random.uniform(-0.5, 0.5, len(df))
)
df['mood_score'] = np.interp(df['mood_score'], (df['mood_score'].min(), df['mood_score'].max()), (0, 10))

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

num_cols = ['age','bmi','daily_steps','sleep_hours','water_intake_l',
            'calories_consumed','resting_hr','systolic_bp','diastolic_bp','cholesterol']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df[['age','gender','bmi','daily_steps','sleep_hours','water_intake_l',
        'calories_consumed','resting_hr','systolic_bp','diastolic_bp',
        'cholesterol','smoker','alcohol','family_history','screen_time','mood_score']]
y = df['disease_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred, average='macro'))
print("F1 (macro):", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:\n", feat_imp.head(10))

joblib.dump(model, "small_model.pkl", compress=3)
