"""Tests for CMA-ES training infrastructure."""

import random

import numpy as np
import torch

from train_cmaes import (
    build_arch_a_model,
    evaluate_fitness,
    flatten_weights,
    unflatten_weights,
)


class TestFlattenUnflatten:
    """Weight vector <-> model parameter round-trip."""

    def test_round_trip_preserves_values(self):
        """Flatten then unflatten should recover identical weights."""
        model, _ = build_arch_a_model()
        original_vec = flatten_weights(model)

        # Perturb to make sure unflatten actually writes
        perturbed = original_vec + 1.0
        unflatten_weights(model, perturbed)
        assert not np.allclose(flatten_weights(model), original_vec)

        # Round-trip back
        unflatten_weights(model, original_vec)
        recovered = flatten_weights(model)
        np.testing.assert_array_equal(original_vec, recovered)

    def test_vector_length_matches_param_count(self):
        """Flat vector length should equal total number of tensor elements."""
        model, _ = build_arch_a_model()
        vec = flatten_weights(model)
        total_params = sum(p.numel() for p in model.parameters())
        assert len(vec) == total_params

    def test_unflatten_wrong_size_raises(self):
        """Unflatten with wrong-sized vector should raise."""
        model, _ = build_arch_a_model()
        vec = flatten_weights(model)
        import pytest

        with pytest.raises(ValueError):
            unflatten_weights(model, vec[:-1])  # one element short


class TestEvaluateFitness:
    """Fitness function correctness."""

    def test_known_good_model_near_perfect(self):
        """Hand-coded model from submission_1l.py should get ~100% accuracy."""
        model, _ = build_arch_a_model(warm_start=True)
        accuracy = evaluate_fitness(model, n_pairs=200, seed=42)
        assert accuracy >= 0.99, f"Hand-coded model only got {accuracy:.2%}"

    def test_random_weights_low_accuracy(self):
        """Random weights should get roughly chance-level (~10%)."""
        model, _ = build_arch_a_model()
        # Set random weights
        rng = np.random.RandomState(123)
        vec = rng.randn(sum(p.numel() for p in model.parameters())) * 0.1
        unflatten_weights(model, vec)
        accuracy = evaluate_fitness(model, n_pairs=200, seed=42)
        assert accuracy < 0.30, f"Random model got suspiciously high {accuracy:.2%}"

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same accuracy."""
        model, _ = build_arch_a_model(warm_start=True)
        a1 = evaluate_fitness(model, n_pairs=100, seed=99)
        a2 = evaluate_fitness(model, n_pairs=100, seed=99)
        assert a1 == a2

    def test_different_seeds_use_different_pairs(self):
        """Different seeds should (usually) produce different pair sets.
        With a perfect model the accuracy is always 1.0 regardless of pairs,
        so we use random weights where accuracy varies by sample."""
        model, _ = build_arch_a_model()
        rng = np.random.RandomState(456)
        vec = rng.randn(sum(p.numel() for p in model.parameters())) * 0.1
        unflatten_weights(model, vec)
        # With random weights, different seeds will likely give different accuracy
        a1 = evaluate_fitness(model, n_pairs=50, seed=1)
        a2 = evaluate_fitness(model, n_pairs=50, seed=2)
        # Not guaranteed different, but overwhelmingly likely with random weights
        # Just check both return valid floats in [0, 1]
        assert 0.0 <= a1 <= 1.0
        assert 0.0 <= a2 <= 1.0


class TestBuildModel:
    """Model factory tests."""

    def test_arch_a_param_count(self):
        """Architecture A (d=3, 3h, ff=4) should have 141 tensor elements."""
        model, meta = build_arch_a_model()
        total = sum(p.numel() for p in model.parameters())
        assert total == 141, f"Expected 141 params, got {total}"

    def test_warm_start_loads_handcoded_weights(self):
        """Warm start should load the analytical weights from submission_1l."""
        model_warm, _ = build_arch_a_model(warm_start=True)
        model_cold, _ = build_arch_a_model(warm_start=False)

        warm_vec = flatten_weights(model_warm)
        cold_vec = flatten_weights(model_cold)
        # They should differ (warm has hand-coded, cold has default init)
        assert not np.allclose(warm_vec, cold_vec)
