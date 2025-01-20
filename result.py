""" Module for bundling algorithm results. """

from typing import List, Optional

import numpy as np


class SplitResult:
    """ Results on either the train or test set. """

    def __init__(
        self,
        pred: np.ndarray,
        conf_logits: np.ndarray,
        conf_probs: np.ndarray,
        beliefs: np.ndarray,
        uncertainty: np.ndarray,
        reject: np.ndarray,
        is_guessing: np.ndarray,
    ) -> None:
        self.pred = pred
        self.conf_logits = conf_logits
        self.conf_probs = conf_probs
        self.beliefs = beliefs
        self.uncertainty = uncertainty
        self.reject = reject
        self.is_guessing = is_guessing

    @classmethod
    def from_logits(
        cls,
        rng: np.random.Generator,
        conf_logits: np.ndarray,
    ):
        """ Create results from logits with random tie-breaking. """

        # Extract predictions
        pred_list: List[int] = []
        guessing: List[bool] = []
        for score_row in conf_logits:
            max_idx = np.flatnonzero(np.isclose(score_row, np.max(score_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute probs for reference
        conf_logits = np.clip(conf_logits, -np.inf, 1e10)
        conf_logits -= np.max(conf_logits, axis=1, keepdims=True)
        logits_exp = np.exp(conf_logits)
        sum_exp = np.sum(logits_exp, axis=1, keepdims=True)
        sum_exp = np.where(sum_exp < 1e-10, 1.0, sum_exp)
        conf_probs = logits_exp / sum_exp

        return cls(
            np.array(pred_list), conf_logits, conf_probs, conf_probs,
            np.zeros(len(pred_list), dtype=float),
            np.zeros(len(pred_list), dtype=bool), np.array(guessing),
        )

    @classmethod
    def from_scores(
        cls,
        rng: np.random.Generator,
        conf_scores: np.ndarray,
    ):
        """ Create results from non-negative scores with random tie-breaking. """

        # Extract prediction
        prob_sum = np.sum(conf_scores, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        conf_probs = conf_scores / prob_sum
        pred_list: List[int] = []
        guessing: List[bool] = []
        for score_row in conf_probs:
            max_idx = np.flatnonzero(np.isclose(score_row, np.max(score_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute logits for reference
        conf_logits = np.log(
            conf_probs,
            out=-np.inf * np.ones_like(conf_probs),
            where=conf_probs > 1e-10,
        )

        return cls(
            np.array(pred_list), conf_logits, conf_probs, conf_probs,
            np.zeros(len(pred_list), dtype=float),
            np.zeros(len(pred_list), dtype=bool), np.array(guessing),
        )

    @classmethod
    def from_scores_with_uncertainty(
        cls,
        rng: np.random.Generator,
        probs: np.ndarray,
        bel: np.ndarray,
        unc: np.ndarray,
    ):
        """ Create result from probabilities, beliefs, and uncertainty. """

        # Normalize probabilities
        prob_sum = np.sum(probs, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        conf_probs = probs / prob_sum

        # Extract predictions from projected probability
        pred_list: List[int] = []
        guessing: List[bool] = []
        for prob_row in conf_probs:
            max_idx = np.flatnonzero(np.isclose(prob_row, np.max(prob_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute logits for reference
        conf_logits = np.log(
            conf_probs,
            out=-np.inf * np.ones_like(conf_probs),
            where=conf_probs > 1e-10,
        )

        return cls(
            np.array(pred_list), conf_logits, conf_probs, bel, unc,
            np.zeros(len(pred_list), dtype=bool), np.array(guessing),
        )

    @classmethod
    def from_scores_with_reject(
        cls,
        rng: np.random.Generator,
        probs: np.ndarray,
        bel: np.ndarray,
        unc: np.ndarray,
        reject: np.ndarray,
    ):
        """ Create result from probabilities, beliefs, and uncertainty. """

        # Normalize probabilities
        prob_sum = np.sum(probs, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        conf_probs = probs / prob_sum

        # Extract predictions from projected probability
        pred_list: List[int] = []
        guessing: List[bool] = []
        for prob_row in conf_probs:
            max_idx = np.flatnonzero(np.isclose(prob_row, np.max(prob_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute logits for reference
        conf_logits = np.log(
            conf_probs,
            out=-np.inf * np.ones_like(conf_probs),
            where=conf_probs > 1e-10,
        )

        return cls(
            np.array(pred_list), conf_logits, conf_probs,
            bel, unc, reject, np.array(guessing),
        )

    def frac_guessing(self) -> float:
        """ Returns the fraction of guessing. """

        return np.count_nonzero(self.is_guessing) / self.is_guessing.shape[0]

    def frac_no_reject(self) -> float:
        """ Returns the fraction of not rejected predictions. """

        return np.count_nonzero(self.reject > 0) / self.reject.shape[0]


class Result:
    """ Results on train and test set. """

    def __init__(
        self,
        train_result: SplitResult,
        test_result: SplitResult,
        holdout_result: Optional[SplitResult] = None,
    ) -> None:
        self.train_result = train_result
        self.test_result = test_result
        self.holdout_result = holdout_result

    def get_holdout_result(self) -> SplitResult:
        """ Return hold-out set result.

        Raises:
            ValueError: If result missing.

        Returns:
            SplitResult: The hold-out set result.
        """

        if not self.holdout_result:
            raise ValueError("Result not available.")
        return self.holdout_result
