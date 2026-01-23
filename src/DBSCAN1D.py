import numpy as np


class DBSCAN1D:
    """
    A one-dimensional implementation of DBSCAN.
    Compatible with sklearn-style interface.
    """

    core_sample_indices_: np.ndarray | None = None
    components_: np.ndarray | None = None
    labels_: np.ndarray | None = None

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def _get_is_core(self, ar):
        """Determine core points based on neighborhood count."""
        left = np.searchsorted(ar, ar - self.eps, side="left")
        right = np.searchsorted(ar, ar + self.eps, side="right")
        return (right - left) >= self.min_samples

    def _assign_core_group_numbers(self, cores):
        """Assign cluster ids to core points based on connectivity."""
        if len(cores) == 0: return np.array([], dtype=int)
        split = np.abs(cores - np.roll(cores, 1)) > self.eps
        split[0] = False  # first element shouldn't trigger a new cluster
        return split.astype(int).cumsum()

    def _get_non_core_labels(self, non_cores, cores, core_nums):
        """Assign non-core points to nearest core if within eps."""
        out = np.full(len(non_cores), -1, dtype=int)
        if len(cores) == 0: return out

        idx_right = np.searchsorted(cores, non_cores)
        idx_left = idx_right - 1
        idx_left = np.clip(idx_left, 0, len(cores)-1)
        idx_right = np.clip(idx_right, 0, len(cores)-1)

        dist_left = np.abs(non_cores - cores[idx_left])
        dist_right = np.abs(non_cores - cores[idx_right])

        nearest = np.where(dist_left <= dist_right, idx_left, idx_right)
        dist_min = np.minimum(dist_left, dist_right)

        connected = dist_min <= self.eps
        out[connected] = core_nums[nearest[connected]]
        return out

    def fit(self, X):
        """Fit DBSCAN1D to 1D data."""
        X = np.asarray(X).flatten()
        sorted_idx = np.argsort(X)
        X_sorted = X[sorted_idx]
        undo_sort = np.argsort(sorted_idx)

        is_core = self._get_is_core(X_sorted)
        cores = X_sorted[is_core]
        non_cores = X_sorted[~is_core]

        core_nums = self._assign_core_group_numbers(cores)
        non_core_nums = self._get_non_core_labels(non_cores, cores, core_nums)

        labels_sorted = np.full(len(X_sorted), -1, dtype=int)
        labels_sorted[is_core] = core_nums
        labels_sorted[~is_core] = non_core_nums

        self.labels_ = labels_sorted[undo_sort]
        self.core_sample_indices_ = np.where(is_core[undo_sort])[0]
        self.components_ = cores

        return self

    def fit_predict(self, X):
        """Fit and return labels."""
        self.fit(X)
        return self.labels_
