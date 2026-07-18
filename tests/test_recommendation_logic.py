#!/usr/bin/env python3
import pytest
from src.kcd2_dice_solver.main import (
    expected_value,
    recommend_move,
    config,
)


class TestStrategicDecisions:
    """Test the strategic recommendation logic of the dice solver."""

    def test_expected_value_basic(self):
        """Test basic expected value calculation."""
        # With no points at risk, EV should be at least 0
        ev = expected_value(6, 0, 0, 0)
        assert ev >= 0

        # With some points at risk, EV should be at least those points
        ev = expected_value(6, 200, 0, 0)
        assert ev >= 200

        # With a lot of points at risk and few dice, might be better to bank
        ev = expected_value(1, 500, 0, 0)
        assert ev == 500  # Should recommend banking

    def test_expected_value_risk_adjustments(self):
        """Test that expected value adjusts risk based on game state."""
        # When behind, should be more aggressive
        ev_behind = expected_value(3, 200, 500, 800)
        ev_neutral = expected_value(3, 200, 500, 500)
        ev_ahead = expected_value(3, 200, 800, 500)

        # When behind, EV should be higher (more willing to risk)
        assert ev_behind > ev_neutral
        # When ahead, EV should be lower (less willing to risk)
        assert ev_neutral > ev_ahead

    def test_expected_value_endgame(self):
        """Test end-game considerations in expected value calculation."""
        # When close to winning, should be more aggressive unless can win by banking
        ev_normal = expected_value(3, 200, 500, 500)
        ev_close = expected_value(3, 200, 1800, 1500, target_score=2000)
        
        # When close to winning but not enough to bank, should be more aggressive
        assert ev_close > ev_normal
        
        # When can win by banking, should bank
        ev_can_win = expected_value(3, 300, 1800, 1500, target_score=2000)
        assert ev_can_win == 300  # Should bank to win

    def test_recommend_move_basic(self):
        """Test basic recommendation logic."""
        # Simple case: recommend holding 1's
        result = recommend_move([1, 2, 3, 4, 5, 6], 6, 0, 0, 0)
        assert result["decision"] in ["score and continue", "score and pass"]
        assert 1 in result["hold_dice"]
        
        # No scoring dice should recommend bust
        result = recommend_move([2, 3, 4, 6, 2, 4], 6, 0, 0, 0)
        assert result["decision"] == "bust"
        
        # Three of a kind should recommend continuing
        result = recommend_move([3, 3, 3, 4, 5, 6], 6, 0, 0, 0)
        assert result["decision"] == "score and continue"
        assert sorted(result["hold_dice"]) == [3, 3, 3]

    def test_recommend_move_high_stakes(self):
        """Test recommendations with significant points at stake."""
        # With high current points, should be more conservative
        result = recommend_move([1, 2, 3, 4, 5, 6], 6, 500, 0, 0)
        
        # If holding just one die with high current points, might recommend passing
        alt_result = recommend_move([1, 2, 3, 4, 6, 2], 6, 500, 0, 0)
        
        # At least one of these should recommend passing to preserve points
        assert result["decision"] == "score and pass" or alt_result["decision"] == "score and pass"

    def test_recommend_move_endgame(self):
        """Test end-game recommendations."""
        # Can win scenario - should always bank
        result = recommend_move(
            [1, 1, 1, 5, 5, 6],  # 1150 points worth
            6, 
            0,  # Current turn points
            900,  # Player score 
            1000,  # Opponent score
            0,  # No devil dice
            2000  # Target score
        )
        assert result["decision"] == "score and pass"
        
        # Close to winning but can't bank yet - should be aggressive
        result = recommend_move(
            [1, 1, 5, 6, 2, 3],  # 250 points worth
            6, 
            0, 
            1700,  # Need 300 to win
            1800, 
            0,
            2000
        )
        assert result["decision"] == "score and continue"
        
        # With enough points to win - should bank immediately
        result = recommend_move(
            [1, 1, 5],  # 250 points worth
            3,
            100,  # Already have 100 points this turn
            1900,  # Total would be 2250, enough to win
            1800,
            0,
            2000
        )
        assert result["decision"] == "score and pass"
        assert "win the game" in result["rationale"].lower()

    def test_recommend_move_devil_dice(self):
        """Test how devil dice affect recommendations."""
        # Standard case without devil dice
        std_result = recommend_move([1, 1, 5, 5, 6, 3], 6, 0, 0, 0, 0)
        
        # Same case but with devil dice
        devil_result = recommend_move([1, 1, 5, 5, 6, 3], 6, 0, 0, 0, 2)
        
        # Devil dice should make recommendation more conservative
        if std_result["decision"] == "score and continue":
            # Either decision changes or EV is lower
            if devil_result["decision"] == std_result["decision"]:
                std_ev = float(std_result["rationale"].split("continuing EV: ")[1].split(" ")[0])
                devil_ev = float(devil_result["rationale"].split("continuing EV: ")[1].split(" ")[0])
                assert devil_ev < std_ev
            else:
                assert devil_result["decision"] == "score and pass"

    def test_recommend_move_rationale_consistency(self):
        """Test that rationales are consistently formatted and meaningful."""
        # Test various scenarios to ensure rationales are properly populated
        scenarios = [
            # Basic case
            ([1, 2, 3, 4, 5, 6], 6, 0, 0, 0),
            # High current points
            ([1, 5, 6, 3, 2, 4], 6, 400, 0, 0),
            # End game
            ([1, 1, 5], 3, 0, 1900, 1800, 0, 2000),
            # Devil dice
            ([1, 5, 2, 3, 4, 6], 6, 0, 0, 0, 2),
            # No scoring dice
            ([2, 3, 4, 6, 2, 4], 6, 0, 0, 0),
        ]
        
        for args in scenarios:
            result = recommend_move(*args)
            
            # Rationale should be populated
            assert result["rationale"], f"Empty rationale for scenario: {args}"
            
            # Rationale should end with proper punctuation
            assert result["rationale"][-1] in [".", "!", "?"], f"Improper punctuation in rationale: {result['rationale']}"
            
            # Rationale should be appropriate for the decision
            if result["decision"] == "score and pass":
                assert "bank" in result["rationale"].lower() or "win" in result["rationale"].lower()
            elif result["decision"] == "score and continue":
                assert "ev: " in result["rationale"].lower() or "continuing" in result["rationale"].lower()
            elif result["decision"] == "bust":
                assert "no scoring" in result["rationale"].lower()


if __name__ == "__main__":
    pytest.main(["-xvs"])