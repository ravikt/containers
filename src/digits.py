import platform

import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)
model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=0)
model.fit(X_train, y_train)

print(f"python {platform.python_version()}  scikit-learn {sklearn.__version__}")
print(f"test_acc {model.score(X_test, y_test):.4f}")
