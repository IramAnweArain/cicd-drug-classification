import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import skops.io as sio
import os

# --- CRITICAL FIX: Create directories automatically ---
# This prevents the "FileNotFoundError" in GitHub Actions
if not os.path.exists("Results"):
    os.makedirs("Results")
if not os.path.exists("Model"):
    os.makedirs("Model")
# ----------------------------------------------------

# 1. Load Data
df = pd.read_csv("drug200.csv")
df = df.sample(frac=1, random_state=42)

# 2. Split Data
X = df.drop("Drug", axis=1)
y = df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create Pipeline
cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# 4. Train
pipe.fit(X_train, y_train)

# 5. Evaluate
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print(f"Accuracy: {accuracy:.2f}")

# 6. Save Metrics
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"Accuracy = {accuracy:.2f}\n")
    outfile.write(f"F1 Score = {f1:.2f}\n")

# 7. Save Confusion Matrix
cm = ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.savefig("Results/model_results.png", dpi=120)

# 8. Save Model
sio.dump(pipe, "Model/drug_pipeline.skops")