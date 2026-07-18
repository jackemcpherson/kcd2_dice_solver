#!/usr/bin/env python3
import pytest
from collections import Counter
from src.kcd2_dice_solver.main import (
    config,
    is_scoring_combination,
    score_trip,
    get_straight_score,
    valid_groups_from_count,
    best_score_from_counter,
    score_hold,
    generate_holds,
    outcome_is_scoring,
    compute_bust_probability,
    average_score,
)


# --------------------------------------
# Tests for Scoring Function Components
# --------------------------------------
class TestScoringFunctions:
    def test_is_scoring_combination(self):
        # Single scorers (1's and 5's)
        assert is_scoring_combination([1]) is True
        assert is_scoring_combination([5]) is True
        assert is_scoring_combination([2, 3, 4, 6]) is False
        assert is_scoring_combination([1, 2, 3, 4, 6]) is True
        
        # Three of a kind
        assert is_scoring_combination([2, 2, 2]) is True
        assert is_scoring_combination([3, 3, 3, 4]) is True
        assert is_scoring_combination([6, 6]) is False
        
        # Straights
        assert is_scoring_combination([1, 2, 3, 4, 5]) is True
        assert is_scoring_combination([2, 3, 4, 5, 6]) is True
        assert is_scoring_combination([1, 2, 3, 4, 5, 6]) is True
        assert is_scoring_combination([1, 3, 4, 5, 6]) is False  # Not a straight

        # Mixed cases
        assert is_scoring_combination([1, 1, 1, 5, 5]) is True  # Three 1's and two 5's
        assert is_scoring_combination([2, 3, 4, 6, 6]) is False  # No scoring dice

    def test_score_trip(self):
        # Three of a kind
        assert score_trip(3, 1) == 1000  # Three 1's
        assert score_trip(3, 2) == 200   # Three 2's
        assert score_trip(3, 5) == 500   # Three 5's
        assert score_trip(3, 6) == 600   # Three 6's
        
        # Four of a kind (doubles the score)
        assert score_trip(4, 1) == 2000  # Four 1's
        assert score_trip(4, 3) == 600   # Four 3's
        
        # Five of a kind (4x the score)
        assert score_trip(5, 1) == 4000  # Five 1's
        assert score_trip(5, 4) == 1600  # Five 4's
        
        # Six of a kind (8x the score)
        assert score_trip(6, 1) == 8000  # Six 1's
        assert score_trip(6, 2) == 1600  # Six 2's

    def test_get_straight_score(self):
        # Valid straights
        assert get_straight_score({1, 2, 3, 4, 5}) == config.small_straight_score
        assert get_straight_score({2, 3, 4, 5, 6}) == config.large_straight_score
        assert get_straight_score({1, 2, 3, 4, 5, 6}) == config.full_straight_score
        
        # Invalid straights
        assert get_straight_score({1, 2, 3, 4, 6}) is None
        assert get_straight_score({1, 3, 4, 5, 6}) is None
        assert get_straight_score({1, 2, 3}) is None

    def test_valid_groups_from_count(self):
        # Test with singles
        cnt = Counter({1: 2, 5: 1})
        groups = valid_groups_from_count(cnt)
        assert (Counter({1: 1}), 100) in groups
        assert (Counter({1: 2}), 200) in groups
        assert (Counter({5: 1}), 50) in groups
        
        # Test with three of a kind
        cnt = Counter({3: 3, 5: 1})
        groups = valid_groups_from_count(cnt)
        assert (Counter({3: 3}), 300) in groups
        assert (Counter({5: 1}), 50) in groups
        
        # Test with straight
        cnt = Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
        groups = valid_groups_from_count(cnt)
        straight_group = (Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1}), config.small_straight_score)
        assert straight_group in groups
        assert (Counter({1: 1}), 100) in groups
        assert (Counter({5: 1}), 50) in groups

    def test_best_score_from_counter(self):
        # Simple cases
        assert best_score_from_counter(Counter()) == 0  # Empty counter
        assert best_score_from_counter(Counter({1: 1})) == 100  # Single 1
        assert best_score_from_counter(Counter({5: 2})) == 100  # Two 5's
        
        # Three of a kind
        assert best_score_from_counter(Counter({2: 3})) == 200  # Three 2's
        
        # Mixed cases
        assert best_score_from_counter(Counter({1: 2, 5: 1})) == 250  # Two 1's and one 5
        assert best_score_from_counter(Counter({3: 3, 5: 1})) == 350  # Three 3's and one 5
        
        # Non-scoring dice
        assert best_score_from_counter(Counter({2: 2, 3: 1})) is None  # No scoring dice

    def test_score_hold(self):
        # Empty hold
        assert score_hold([]) == 0
        
        # Singles
        assert score_hold([1]) == 100
        assert score_hold([5]) == 50
        assert score_hold([1, 5]) == 150
        assert score_hold([1, 1]) == 200
        
        # Three of a kind
        assert score_hold([2, 2, 2]) == 200
        assert score_hold([4, 4, 4]) == 400
        assert score_hold([1, 1, 1]) == 1000
        
        # Four of a kind
        assert score_hold([3, 3, 3, 3]) == 600
        
        # Mixed cases
        assert score_hold([1, 1, 1, 5]) == 1050
        assert score_hold([2, 2, 2, 5]) == 250
        
        # Straights
        assert score_hold([1, 2, 3, 4, 5]) == config.small_straight_score
        assert score_hold([2, 3, 4, 5, 6]) == config.large_straight_score
        assert score_hold([1, 2, 3, 4, 5, 6]) == config.full_straight_score
        
        # Invalid combinations
        assert score_hold([2, 3, 4]) is None
        assert score_hold([6, 6]) is None

    def test_generate_holds(self):
        # Single Die
        holds = generate_holds([1])
        assert ([1], 100) in holds
        holds = generate_holds([5])
        assert ([5], 50) in holds
        holds = generate_holds([2])
        assert len(holds) == 0
        
        # Multiple Dice with Singles
        holds = generate_holds([1, 5])
        assert ([1], 100) in holds
        assert ([5], 50) in holds
        assert ([1, 5], 150) in holds
        
        # Three of a Kind
        holds = generate_holds([3, 3, 3])
        assert ([3, 3, 3], 300) in holds
        
        # Straight
        holds = generate_holds([1, 2, 3, 4, 5])
        straight_hold = ([1, 2, 3, 4, 5], config.small_straight_score)
        assert straight_hold in holds
        
        # Complex case
        holds = generate_holds([1, 1, 3, 3, 3, 5])
        assert ([1], 100) in holds
        assert ([1, 1], 200) in holds
        assert ([3, 3, 3], 300) in holds
        assert ([5], 50) in holds
        # Look for combined holds
        combined_found = False
        for hold, score in holds:
            if sorted(hold) == [1, 1, 3, 3, 3, 5] and score == 550:
                combined_found = True
                break
        assert combined_found


# --------------------------------------
# Tests for Probability Functions
# --------------------------------------
class TestProbabilityFunctions:
    def test_outcome_is_scoring(self):
        # Should match is_scoring_combination behavior
        assert outcome_is_scoring((1,)) is True
        assert outcome_is_scoring((5,)) is True
        assert outcome_is_scoring((2, 3, 4, 6)) is False
        
        # Three of a kind
        assert outcome_is_scoring((2, 2, 2)) is True
        assert outcome_is_scoring((6, 6)) is False
        
        # Straights
        assert outcome_is_scoring((1, 2, 3, 4, 5)) is True
        assert outcome_is_scoring((2, 3, 4, 5, 6)) is True
        assert outcome_is_scoring((1, 2, 3, 4, 5, 6)) is True

    def test_compute_bust_probability(self):
        # Edge cases
        assert compute_bust_probability(0) == 0.0
        
        # Known values (from cache)
        assert compute_bust_probability(1) == 2/3  # 4 out of 6 faces don't score
        assert compute_bust_probability(2) == 4/9
        
        # Test bounds for all reasonable dice counts
        for n in range(1, 7):
            bust_prob = compute_bust_probability(n)
            assert 0 <= bust_prob <= 1
            # Bust probability should decrease as dice count increases
            if n > 1:
                assert bust_prob < compute_bust_probability(n-1)

    def test_average_score(self):
        # Edge cases
        assert average_score(0) == 0.0
        
        # Known value for single die
        assert average_score(1) == 25.0  # (100 + 50)/6
        
        # Test that average score increases with more dice
        for n in range(1, 6):
            assert average_score(n) < average_score(n+1)
        
        # Test values are in a reasonable range
        for n in range(1, 7):
            avg = average_score(n)
            # Very rough bounds: single die minimum is 0, maximum is 100
            assert 0 <= avg <= 1000 * n


# --------------------------------------
# Integration Tests
# --------------------------------------
class TestIntegrationScenarios:
    def test_complete_turn_scenario(self):
        """Test a complete turn with multiple decisions."""
        # Start with 6 dice, roll [1, 1, 2, 3, 5, 6]
        roll1 = [1, 1, 2, 3, 5, 6]
        holds1 = generate_holds(roll1)
        
        # We should be able to hold [1, 1, 5] for 250 points
        hold1_found = False
        for hold, score in holds1:
            if sorted(hold) == [1, 1, 5] and score == 250:
                hold1_found = True
                break
        assert hold1_found
        
        # After holding [1, 1], we have 4 dice left
        # Let's say we roll [2, 3, 4, 6]
        roll2 = [2, 3, 4, 6]
        holds2 = generate_holds(roll2)
        
        # No scoring combinations
        assert len(holds2) == 0
        
        # In real game, this would be a bust, but let's try another scenario
        # Holding [1, 1, 5] from first roll, we'd have 3 dice left
        # If we roll [1, 3, 5]
        roll3 = [1, 3, 5]
        holds3 = generate_holds(roll3)
        
        # We should be able to hold [1, 5] for additional 150 points
        hold3_found = False
        for hold, score in holds3:
            if sorted(hold) == [1, 5] and score == 150:
                hold3_found = True
                break
        assert hold3_found
        
        # Total score would be 250 + 150 = 400

    def test_straight_detection_and_scoring(self):
        """Test correct identification and scoring of all straight types."""
        # Small straight
        roll1 = [1, 2, 3, 4, 5, 6]
        holds1 = generate_holds(roll1)
        
        small_straight_found = False
        for hold, score in holds1:
            if sorted(hold) == [1, 2, 3, 4, 5] and score == config.small_straight_score:
                small_straight_found = True
                break
        assert small_straight_found
        
        # Large straight
        large_straight_found = False
        for hold, score in holds1:
            if sorted(hold) == [2, 3, 4, 5, 6] and score == config.large_straight_score:
                large_straight_found = True
                break
        assert large_straight_found
        
        # Full straight
        full_straight_found = False
        for hold, score in holds1:
            if sorted(hold) == [1, 2, 3, 4, 5, 6] and score == config.full_straight_score:
                full_straight_found = True
                break
        assert full_straight_found
        
        # Modified order should still be detected
        roll2 = [3, 1, 5, 2, 4, 6]
        holds2 = generate_holds(roll2)
        
        small_straight_found = False
        for hold, score in holds2:
            if sorted(hold) == [1, 2, 3, 4, 5] and score == config.small_straight_score:
                small_straight_found = True
                break
        assert small_straight_found


if __name__ == "__main__":
    pytest.main(["-xvs"])