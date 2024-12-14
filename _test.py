import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)

input_ = lucid.Tensor(X_train_sc)
target = lucid.Tensor(y_train)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        return super().forward()
