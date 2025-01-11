import numpy as np
from collections import Counter
import abc

class Node:
    """
    A class representing a node in the decision tree (for both classification and regression).
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        # For decision nodes
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # For leaf nodes
        self.value = value

class BaseDecisionTree(metaclass=abc.ABCMeta):
    """
    Abstract base class for a Decision Tree.
    It holds all the common logic for splitting, building, etc.
    Subclasses must implement the abstract methods:
      - _select_split(...)
      - _leaf_value(...)
    """

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        n_features=None,
        split_criterion=None,
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        self.split_criterion = split_criterion

    def fit(self, X, y, oob_X=None, oob_y=None):
        """
        Build the decision tree.
        """
        # If n_features is not specified, consider all features
        self.cuts = np.zeros(X.shape[1])
        self.total_cuts = 0
        self.oob_X = oob_X
        self.oob_y = oob_y
        self.probs = None
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._build_tree(X, y)

    def predict(self, X):
        """
        Predict values (class labels or regression values) for samples in X.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        num_samples, num_features = X.shape

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Select random subset of features
        feat_idxs = np.random.choice(num_features, self.n_features, replace=False)
        # Find the best feature and threshold to split
        best_feat, best_thresh = self._select_split(X, y, feat_idxs)

        # If we cannot find any effective split => leaf node
        if best_feat is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        self.cuts[best_feat] += 1
        self.total_cuts += 1
        # Otherwise, split and recurse
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _split(self, X_column, threshold):
        """
        Split the dataset based on a feature column and threshold.
        """
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    @abc.abstractmethod
    def _select_split(self, X, y, feat_idxs):
        """
        Calculate the feature and threshold by which to split the data.
        """
        pass

    @abc.abstractmethod
    def _leaf_value(self, y):
        """
        Return the leaf value (e.g. majority label or mean).
        Must be implemented by subclass.
        """
        pass


# ===================================================
#           DECISION TREE CLASSIFIER
# ===================================================

class DecisionTreeClassifier(BaseDecisionTree):
    """
    Decision tree for classification.
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None, split_criterion="gini"):
        super().__init__(max_depth, min_samples_split, n_features, split_criterion)

    def _select_split(self, X, y, feat_idxs):
        """Select a splitting feature and threshold. Gini or middle.
        """

        if self.split_criterion == 'gini':
            best_feat, best_thresh = self._gini_split(X, y, feat_idxs)
        elif self.split_criterion == 'middle':
            best_feat, best_thresh = self._middle_split(X, y, feat_idxs)
        else:
            raise ValueError(f"Unknown split criterion {self.split_criterion}")
        
        if best_feat is not None:
            left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                best_feat = None

        return best_feat, best_thresh
    
    def _gini_split(self, X, y, features):
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

        if best_gain < 1e-15:
            return None, None
        else:
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

    def _leaf_value(self, y):
        """
        Return the most common label in y.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _gini(self, y):
        """
        Calculate the Gini impurity.
        """
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1 - np.sum(probabilities**2)


# ===================================================
#           DECISION TREE REGRESSOR
# ===================================================

class DecisionTreeRegressor(BaseDecisionTree):
    """
    Decision tree for regression.
    """

    def __init__(self, max_depth=None, min_samples_split=2, n_features=None, split_criterion="mse"):
        super().__init__(max_depth, min_samples_split, n_features, split_criterion)

    def _select_split(self, X, y, feat_idxs):
        """Select a splitting feature and threshold. Gini or middle.
        """
        if self.split_criterion == 'mse':
            best_feat, best_thresh = self._mse_split(X, y, feat_idxs)
        elif self.split_criterion == 'middle':
            best_feat, best_thresh = self._oob_middle_split(X, y, feat_idxs)
        else:
            raise ValueError(f"Unknown split criterion {self.split_criterion}")
        
        if best_feat is not None:
            left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                best_feat = None

        
        return best_feat, best_thresh
    
    def _compute_probs_oob(self):
        if self.oob_X is None or self.oob_y is None:
            return
        
        # Compute the parent MSE using OOB data
        parent_mse = self._mse(self.oob_y)
        
        # For each feature, compute MSE gain on OOB data if we split at the middle
        gains = []
        for feature_idx in range(self.oob_X.shape[1]):
            
            X_oob_col = self.oob_X[:, feature_idx]
            if self.oob_X.size == 0:
                gains.append(0.0)
                continue

            min_val, max_val = X_oob_col.min(), X_oob_col.max()

            if min_val == max_val:
                gains.append(0.0)
                continue
            
            threshold = (min_val + max_val) / 2.0 # middle split

            left_oob_idxs = np.argwhere(X_oob_col <= threshold).flatten()
            right_oob_idxs = np.argwhere(X_oob_col > threshold).flatten()

            if len(left_oob_idxs) == 0 or len(right_oob_idxs) == 0:
                # No valid split
                gains.append(0.0)
                continue

            gain = self._split_gain(self.oob_y, left_oob_idxs, right_oob_idxs, parent_mse)
            gains.append(gain)

        self.probs = np.array(gains)
        
    
    def _oob_middle_split(self, X, y, features):
        """
        Pick a feature with probability proportional to the MSE reduction on OOB data,
        then split at the middle of that feature's OOB range.
        """
        
        if self.oob_X is None or self.oob_y is None:
            return self._middle_split(X, y, features)
        

        if self.probs is None:
            self._compute_probs_oob()
        
        probs_node = np.zeros_like(self.probs) - np.inf
        probs_node[features] = self.probs[features]
        probs_node = probs_node / max(max(probs_node), 1e-6)
        probs = np.exp(probs_node) / sum(np.exp(probs_node))

        if np.isnan(probs).any():
            print("There are NaN probabilities")
            print(probs_node)
            print(self.probs[features])

        valid = False
        counter = 0
        while not valid and counter < 10:
            chosen_idx = np.random.choice(len(self.probs), p=probs)
            X_column = X[:, chosen_idx]
            min_value, max_value = X_column.min(), X_column.max()
            if min_value != max_value:
                valid = True  # Cannot split further
            threshold = (min_value + max_value) / 2
            counter += 1
        
        return chosen_idx, threshold

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
    
    def _mse_split(self, X, y, feat_idxs):
        """
        Select the best splitting using mse loss.
        """
        best_feat, best_thresh, best_gain = None, None, -1
        parent_metric = self._mse(y)  # e.g., Gini or MSE at the root

        for feature_idx in feat_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                # Calculate the gain (drop in impurity or MSE)
                gain = self._split_gain(y, left_idxs, right_idxs, parent_metric)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_idx
                    best_thresh = threshold
        
        return best_feat, best_thresh

    def _split_gain(self, y, left_idxs, right_idxs, parent_mse):
        """
        MSE reduction from splitting into left and right.
        """
        y_left, y_right = y[left_idxs], y[right_idxs]
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        mse_left = self._mse(y_left)
        mse_right = self._mse(y_right)
        child_mse = (n_left / n) * mse_left + (n_right / n) * mse_right
        gain = parent_mse - child_mse
        return gain

    def _leaf_value(self, y):
        """
        Return the mean of y.
        """
        return np.mean(y)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y)**2)


# ===================================================
#           ABSTRACT BASE RANDOM FOREST
# ===================================================

class MyBaseRandomForest(metaclass=abc.ABCMeta):
    """
    Abstract base class for a Random Forest.
    Holds the common code for building multiple trees, bagging, etc.
    Subclasses must implement:
      - _create_tree()
      - _aggregate_predictions(...)
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2, 
        n_features=None,
        bagging=True,
        split_criterion="middle",
    ):
        self.n_estimators = n_estimators            # Number of trees in the forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split
        self.n_features_type = n_features
        self.n_features = n_features                # Number of features to consider when looking for the best split
        self.bagging = bagging                      # Whether to use bagging
        self.split_criterion = split_criterion      # Splitting criterion ('gini' or 'middle')
        self.trees = []                             # List to store the trees

    def fit(self, X, y):
        """
        Build the forest of trees.
        """
        if self.n_features_type == "sqrt":
            self.n_features = int(np.sqrt(X.shape[1])) + 1

        self.trees = []
        for _ in range(self.n_estimators):
            if self.bagging:
                idxs = np.random.choice(len(X), len(X), replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
                unique_inbag_idxs = np.unique(idxs)
                oob_mask = np.ones(len(X), dtype=bool)
                oob_mask[unique_inbag_idxs] = False

                oob_X, oob_y = X[oob_mask], y[oob_mask]
            else:
                X_sample, y_sample = X, y
                oob_X, oob_y = None, None
            # Create a new tree (classifier or regressor)
            tree = self._create_tree()
            tree.fit(X_sample, y_sample, oob_X=oob_X, oob_y=oob_y)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict output for X by aggregating predictions from each tree.
        """
        # Get predictions from each tree
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # all_preds.shape => (n_estimators, n_samples)
        return self._aggregate_predictions(all_preds)
    
    def get_cut_probs(self):
        return np.stack([tree.cuts / tree.total_cuts for tree in self.trees])
    
    @abc.abstractmethod
    def _create_tree(self):
        """
        Create a new decision tree (classifier or regressor).
        Must return an instance of DecisionTreeClassifier or DecisionTreeRegressor.
        """
        pass

    @abc.abstractmethod
    def _aggregate_predictions(self, all_preds):
        """
        Aggregate the predictions from each tree (majority vote for classifier,
        average for regressor, etc.).
        """
        pass


# ===================================================
#       RANDOM FOREST CLASSIFIER (INHERIT)
# ===================================================

class MyRandomForestClassifier(MyBaseRandomForest):
    """
    Random Forest for classification, inherits from BaseRandomForest.
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 n_features=None, bagging=True, split_criterion="gini"):
        super().__init__(n_estimators, max_depth, min_samples_split, n_features, bagging, split_criterion)

    def _create_tree(self):
        """
        Create a DecisionTreeClassifier with the current config.
        """
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
        )

    def _aggregate_predictions(self, all_preds):
        """
        Majority vote across n_estimators.
        """
        # Transpose shape => (n_samples, n_estimators)
        all_preds = np.swapaxes(all_preds, 0, 1)
        # For each sample, pick the most common label
        y_pred = [Counter(sample).most_common(1)[0][0] for sample in all_preds]
        return np.array(y_pred)
    



# ===================================================
#       RANDOM FOREST REGRESSOR (INHERIT)
# ===================================================

class MyRandomForestRegressor(MyBaseRandomForest):
    """
    Random Forest for regression, inherits from BaseRandomForest.
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, 
                 n_features=None, bagging=True, split_criterion="mse"):
        super().__init__(n_estimators, max_depth, min_samples_split, n_features, bagging, split_criterion)

    def _create_tree(self):
        """
        Create a DecisionTreeRegressor with the current config.
        """
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
            split_criterion=self.split_criterion,
        )

    def _aggregate_predictions(self, all_preds):
        """
        Average predictions across n_estimators.
        """
        # all_preds shape => (n_estimators, n_samples)
        # Average over axis=0 => shape (n_samples,)
        return np.mean(all_preds, axis=0)


# ===================================================
#                 DEMO / TEST
# ===================================================
if __name__ == "__main__":
    from sklearn.datasets import load_iris, fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error

    # -----------------------------------------
    # DEMO 1: RandomForestClassifier on Iris
    # -----------------------------------------
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = MyRandomForestClassifier(n_estimators=5, max_depth=5, min_samples_split=2,
                                 n_features=None, bagging=True, split_criterion="gini")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForestClassifier accuracy on Iris: {acc:.4f}")

    # -----------------------------------------
    # DEMO 2: RandomForestRegressor on California Housing
    # -----------------------------------------
    boston = fetch_california_housing()
    X_b, y_b = boston.data, boston.target
    print("California housing shape, ", X_b.shape)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size=0.2)

    rfr = MyRandomForestRegressor(n_estimators=5, max_depth=5, min_samples_split=2,
                                n_features=None, bagging=True, split_criterion="mse")
    rfr.fit(X_train_b, y_train_b)
    y_pred_b = rfr.predict(X_test_b)
    mse_b = mean_squared_error(y_test_b, y_pred_b)
    print(f"RandomForestRegressor MSE on California Housing: {mse_b:.4f}")
