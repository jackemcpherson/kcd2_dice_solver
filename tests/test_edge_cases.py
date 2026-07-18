#!/usr/bin/env python3
import pytest
import time
import random
from collections import Counter
from src.kcd2_dice_solver.main import (
    score_hold,
    generate_holds,
    recommend_move,
    config,
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        # Empty hold
        assert score_hold([]) == 0
        
        # Empty roll
        assert generate_holds([]) == []
        
        # Recommendation with empty roll
        with pytest.raises(Exception):
            recommend_move([], 0, 0, 0, 0)

    def test_max_values(self):
        """Test with maximum possible values."""
        # Six of a kind
        assert score_hold([1, 1, 1, 1, 1, 1]) == 8000  # Six 1's
        assert score_hold([6, 6, 6, 6, 6, 6]) == 4800  # Six 6's
        
        # Very high scores
        result = recommend_move([1, 1, 1, 1, 1, 1], 6, 5000, 10000, 12000)
        assert result["decision"] == "score and pass"  # Should bank such a high score

    def test_unusual_dice_counts(self):
        """Test with non-standard numbers of dice."""
        # More than 6 dice (unusual but should handle gracefully)
        holds = generate_holds([1, 2, 3, 4, 5, 6, 1])
        assert len(holds) > 0
        
        # Check that recommendation works with different dice counts
        for dice_count in [1, 2, 3, 4, 5, 6, 7]:
            roll = [random.randint(1, 6) for _ in range(dice_count)]
            try:
                result = recommend_move(roll, dice_count, 0, 0, 0)
                assert isinstance(result, dict)
                assert "decision" in result
            except Exception as e:
                pytest.fail(f"Failed with {dice_count} dice: {e}")

    def test_extreme_game_states(self):
        """Test with extreme game states."""
        # Extreme score disparities
        result = recommend_move([1, 5, 2, 3, 4, 6], 6, 0, 0, 10000)
        assert result["decision"] == "score and continue"  # Should be aggressive when far behind
        
        result = recommend_move([1, 5, 2, 3, 4, 6], 6, 0, 10000, 0)
        assert result["decision"] in ["score and pass", "score and continue"]  # Might be conservative when far ahead
        
        # Very close end game
        result = recommend_move([1, 5], 2, 0, 1950, 1950, 0, 2000)
        assert result["decision"] == "score and continue"  # Should be aggressive in close endgame


class TestStressTests:
    """Performance and stress tests."""

    def test_performance_generate_holds(self):
        """Measure performance of hold generation for complex cases."""
        start_time = time.time()
        
        # Generate all possible rolls
        for _ in range(100):
            roll = [random.randint(1, 6) for _ in range(6)]
            holds = generate_holds(roll)
        
        duration = time.time() - start_time
        assert duration < 5.0, f"Generate holds took too long: {duration}s"

    def test_performance_recommend_move(self):
        """Measure performance of move recommendation."""
        start_time = time.time()
        
        # Generate all possible rolls
        for _ in range(50):
            roll = [random.randint(1, 6) for _ in range(6)]
            current_points = random.randint(0, 500)
            player_score = random.randint(0, 1500)
            opponent_score = random.randint(0, 1500)
            recommend_move(roll, 6, current_points, player_score, opponent_score)
        
        duration = time.time() - start_time
        assert duration < 10.0, f"Recommend move took too long: {duration}s"

    def test_consistency_recommendations(self):
        """Test that recommendations are consistent for the same inputs."""
        # Fixed roll and game state
        roll = [1, 2, 3, 4, 5, 6]
        
        # Get recommendation multiple times
        results = []
        for _ in range(5):
            result = recommend_move(roll, 6, 100, 500, 600)
            results.append((result["decision"], tuple(sorted(result["hold_dice"]))))
        
        # All recommendations should be the same
        assert len(set(results)) == 1, "Inconsistent recommendations for same input"

    def test_all_roll_combinations(self):
        """Test a representative sample of all possible roll combinations."""
        # Test 50 random rolls
        for _ in range(50):
            roll = [random.randint(1, 6) for _ in range(6)]
            holds = generate_holds(roll)
            
            # Verify each hold is valid
            for hold, score in holds:
                hold_score = score_hold(hold)
                assert hold_score is not None, f"Invalid hold: {hold}"
                assert hold_score == score, f"Incorrect score for {hold}: expected {hold_score}, got {score}"
                
                # Verify each hold is a subset of the roll
                roll_counter = Counter(roll)
                hold_counter = Counter(hold)
                for face, count in hold_counter.items():
                    assert roll_counter[face] >= count, f"Hold {hold} not a subset of roll {roll}"


if __name__ == "__main__":
    pytest.main(["-xvs"])