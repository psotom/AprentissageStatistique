import numpy as np
from collections import Counter

class Node:
    """
    A class representing a node in the decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        # For decision nodes
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # For leaf nodes
        self.value = value

class DecisionTree:
    """
    A class representing a decision tree.
    """
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None, split_criterion='gini'):
        self.max_depth = max_depth                  # Maximum depth of the tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split
        self.n_features = n_features                # Number of features to consider when looking for the best split
        self.root = None                            # Root node of the tree
        self.split_criterion = split_criterion      # Splitting criterion ('gini' or 'middle')

    def fit(self, X, y):
        """
        Build the decision tree.
        """
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _gini(self, y):
        """
        Calculate the Gini impurity for labels y.
        """
        m = len(y)
        counts = np.bincount(y)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _split(self, X_column, threshold):
        """
        Split the dataset based on a feature and a threshold.
        """
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X, y, features):
        """
        Find the best split for a node using Gini impurity.
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        parent_gini = self._gini(y)

        for feature_idx in features:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                y_left, y_right = y[left_idxs], y[right_idxs]
                n = len(y)
                n_left, n_right = len(y_left), len(y_right)
                gini_left, gini_right = self._gini(y_left), self._gini(y_right)
                child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

                gain = parent_gini - child_gini

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _middle_split(self, X, y, features):
        """
        Split at the middle of the feature's range.
        """
        for feature_idx in features:
            X_column = X[:, feature_idx]
            min_value, max_value = X_column.min(), X_column.max()
            if min_value == max_value:
                continue  # Cannot split further
            threshold = (min_value + max_value) / 2
            return feature_idx, threshold
        return None, None  # If no split is possible

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Select random features (M_try)
        feat_idxs = np.random.choice(num_features, self.n_features, replace=False)

        # Find the best split
        if self.split_criterion == 'gini':
            best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        elif self.split_criterion == 'middle':
            best_feat, best_thresh = self._middle_split(X, y, feat_idxs)
        else:
            raise ValueError("Unknown split criterion")

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split the dataset
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """
        Return the most common label in y.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

class RandomForest:
    """
    A class representing a Random Forest model.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, n_features=None, bagging=True, split_criterion='gini'):
        self.n_estimators = n_estimators            # Number of trees in the forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split
        self.n_features = n_features                # Number of features to consider when looking for the best split
        self.bagging = bagging                      # Whether to use bagging
        self.split_criterion = split_criterion      # Splitting criterion ('gini' or 'middle')
        self.trees = []                             # List to store the trees

    def fit(self, X, y):
        """
        Build the forest of trees.
        """
        self.trees = []
        for _ in range(self.n_estimators):
            if self.bagging:
                idxs = np.random.choice(len(X), len(X), replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features,
                                split_criterion=self.split_criterion)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class for X.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Transpose to get predictions for each sample
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Majority vote
        y_pred = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        return np.array(y_pred)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Random Forest classifier
    clf_standard = RandomForest(n_estimators=10, max_depth=10, min_samples_split=2, n_features=None, bagging=True, split_criterion='gini')
    clf_standard.fit(X_train, y_train)
    y_pred_standard = clf_standard.predict(X_test)
    accuracy_standard = accuracy_score(y_test, y_pred_standard)
    print(f"Standard Random Forest Accuracy: {accuracy_standard:.4f}")

    # Simple Random Forest classifier
    clf_simple = RandomForest(n_estimators=10, max_depth=10, min_samples_split=2, n_features=None, bagging=False, split_criterion='middle')
    clf_simple.fit(X_train, y_train)
    y_pred_simple = clf_simple.predict(X_test)

    accuracy_simple = accuracy_score(y_test, y_pred_simple)
    print(f"Simple Model Accuracy: {accuracy_simple:.4f}")
