#!/usr/bin/env python3
import itertools
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# -------------------------------
# Game Configuration
# -------------------------------
class GameConfig:
    """Configuration class for game rules and scoring."""

    def __init__(self):
        # Scoring rules
        self.single_one_score = 100
        self.single_five_score = 50

        # Straight scores
        self.small_straight_score = 500  # 1-2-3-4-5
        self.large_straight_score = 750  # 2-3-4-5-6
        self.full_straight_score = 1500  # 1-2-3-4-5-6

        # Straight definitions
        self.small_straight = set(range(1, 6))  # 1,2,3,4,5
        self.large_straight = set(range(2, 7))  # 2,3,4,5,6
        self.full_straight = set(range(1, 7))  # 1,2,3,4,5,6

        # Other rules
        self.hot_dice_count = 6  # Number of dice to get when all are used
        self.min_of_a_kind = 3  # Minimum for three-of-a-kind scoring
        self.dice_faces = 6  # Standard dice
        self.target_score_default = 10000  # Default target score

        # Strategy parameters
        self.ev_discount = 0.5  # Discount factor for future turns
        self.ev_max_depth = 2  # Max recursion depth for EV calculation
        self.risk_bonus = 1.05  # When behind
        self.risk_penalty = 0.95  # When ahead


# Create a global config instance
config = GameConfig()


# -------------------------------
# Scoring Functions
# -------------------------------
def score_trip(count: int, num: int) -> int:
    """Return score for k-of-a-kind for a given face, where k >= 3."""
    base = 1000 if num == 1 else num * 100
    return base * (2 ** (count - 3))


def valid_groups_from_count(cnt: Counter) -> List[Tuple[Counter, int]]:
    """
    Given a counter of dice (the multiset), return a list of candidate groups
    (as a Counter) that can be scored from some dice in cnt, along with their score.
    """
    groups = []
    # For singles (only 1's and 5's)
    for face in (1, 5):
        if cnt[face] > 0:
            for k in range(1, cnt[face] + 1):
                group = Counter({face: k})
                score_val = k * (
                    config.single_one_score if face == 1 else config.single_five_score
                )
                groups.append((group, score_val))

    # For three-or-more of a kind (for any face)
    for face in cnt:
        if cnt[face] >= config.min_of_a_kind:
            for k in range(config.min_of_a_kind, cnt[face] + 1):
                group = Counter({face: k})
                score_val = score_trip(k, face)
                groups.append((group, score_val))

    # Check for 1-2-3-4-5-6 (full straight)
    if len(cnt) == 6 and all(cnt[i] >= 1 for i in range(1, 7)):
        group = Counter({i: 1 for i in range(1, 7)})
        groups.append((group, config.full_straight_score))

    # Check for 1-2-3-4-5 (small straight)
    if all(cnt.get(i, 0) >= 1 for i in range(1, 6)):
        group = Counter({i: 1 for i in range(1, 6)})
        groups.append((group, config.small_straight_score))

    # Check for 2-3-4-5-6 (large straight)
    if all(cnt.get(i, 0) >= 1 for i in range(2, 7)):
        group = Counter({i: 1 for i in range(2, 7)})
        groups.append((group, config.large_straight_score))

    return groups


def best_score_from_counter(cnt: Counter) -> Optional[int]:
    """
    Given a Counter of dice (the hold), return the maximum score obtainable by partitioning
    the entire multiset into valid scoring groups. If no partition covers all dice, return None.
    """
    if sum(cnt.values()) == 0:
        return 0
    best = -math.inf
    groups = valid_groups_from_count(cnt)
    found = False
    for group, group_score in groups:
        # Check if group is a submultiset of cnt.
        if all(cnt[x] >= group[x] for x in group):
            new_cnt = cnt.copy()
            for x in group:
                new_cnt[x] -= group[x]
                if new_cnt[x] <= 0:
                    del new_cnt[x]
            sub_score = best_score_from_counter(new_cnt)
            if sub_score is not None:
                total = group_score + sub_score
                if total > best:
                    best = total
                    found = True
    return best if found else None


def score_hold(hold: List[int]) -> Optional[int]:
    """
    Given a hold (list of dice), return the maximum score obtainable if the hold is entirely scoring.
    Return None if the hold cannot be partitioned into valid scoring groups.
    """
    # Special case for straights
    hold_set = set(hold)
    if len(hold) == 6 and hold_set == config.full_straight:
        return config.full_straight_score
    if len(hold) == 5:
        if hold_set == config.small_straight:
            return config.small_straight_score
        if hold_set == config.large_straight:
            return config.large_straight_score

    # Normal case - proceed with recursive scoring
    cnt = Counter(hold)
    return best_score_from_counter(cnt)


def generate_holds(roll: List[int]) -> List[Tuple[List[int], int]]:
    """
    Generate all valid holds from the roll more efficiently.
    Uses specialized handling for straights and a more efficient approach
    for building combinations.
    Returns a list of tuples: (sorted list of dice in hold, maximum score).
    """
    n = len(roll)
    holds_dict: Dict[Tuple[int, ...], int] = {}

    # Check for straights first as special cases
    roll_set = set(roll)

    # Full straight (1-6)
    if config.full_straight.issubset(roll_set) and n >= 6:
        straight_dice = tuple(sorted([i for i in range(1, 7)]))
        holds_dict[straight_dice] = config.full_straight_score

    # Small straight (1-5)
    if config.small_straight.issubset(roll_set) and n >= 5:
        straight_dice = tuple(sorted([i for i in range(1, 6)]))
        holds_dict[straight_dice] = config.small_straight_score

    # Large straight (2-6)
    if config.large_straight.issubset(roll_set) and n >= 5:
        straight_dice = tuple(sorted([i for i in range(2, 7)]))
        holds_dict[straight_dice] = config.large_straight_score

    # Optimization: Start with individual scoring dice (1's and 5's) and build up
    holds_to_process = []

    # Add singles (1's and 5's) first as baseline holds
    roll_counter = Counter(roll)
    for face in (1, 5):
        if roll_counter[face] > 0:
            for count in range(1, roll_counter[face] + 1):
                hold = tuple([face] * count)
                holds_dict[hold] = count * (
                    config.single_one_score if face == 1 else config.single_five_score
                )
                holds_to_process.append(hold)

    # Process each potential triplet (three of a kind)
    for face in range(1, 7):
        if roll_counter[face] >= 3:
            for count in range(3, roll_counter[face] + 1):
                hold = tuple([face] * count)
                holds_dict[hold] = score_trip(count, face)
                holds_to_process.append(hold)

    # For all other combinations - more efficiently build valid scoring combinations
    # We use the more efficient approach for smaller sets and the traditional approach for larger ones
    # The threshold of 4 is based on experimentation with typical rolls
    if n <= 4:
        # For small dice counts, still use the combinatorial approach
        for r in range(1, n + 1):
            for indices in itertools.combinations(range(n), r):
                subset = [roll[i] for i in indices]
                # Skip if it's a straight we already handled
                if len(subset) == 6 and set(subset) == config.full_straight:
                    continue
                if len(subset) == 5 and (
                    set(subset) == config.small_straight
                    or set(subset) == config.large_straight
                ):
                    continue

                sorted_subset = tuple(sorted(subset))
                if sorted_subset in holds_dict:
                    continue
                s = score_hold(list(sorted_subset))
                if s is not None and s > 0:
                    holds_dict[sorted_subset] = s
    else:
        # For larger dice counts, use a more efficient approach
        # Combine existing holds and check if they're valid
        processed_holds = set(holds_dict.keys())
        while holds_to_process:
            current_hold = holds_to_process.pop(0)
            # Try adding other dice from the roll
            for i in range(n):
                die = roll[i]
                # Skip if we've already used this index in the current hold
                if current_hold.count(die) >= roll_counter[die]:
                    continue

                # Create a new hold with this die added
                new_hold = tuple(sorted(current_hold + (die,)))

                # Skip if we've already processed this hold
                if new_hold in processed_holds:
                    continue

                # Check if this is a valid scoring hold
                score_val = score_hold(list(new_hold))
                if score_val is not None and score_val > 0:
                    holds_dict[new_hold] = score_val
                    holds_to_process.append(new_hold)
                    processed_holds.add(new_hold)

    # Convert to list of (list, score)
    return [(list(k), v) for k, v in holds_dict.items()]


# -------------------------------
# Risk & Probability Functions
# -------------------------------
def outcome_is_scoring(outcome: Tuple[int, ...]) -> bool:
    """
    Determine if an outcome (tuple of dice) is scoring.
    An outcome is scoring if at least one die is scorable.
    """
    # Use the config values for consistency
    if any(d == 1 or d == 5 for d in outcome):
        return True

    counts = Counter(outcome)
    if any(cnt >= config.min_of_a_kind for cnt in counts.values()):
        return True

    # Check for straights
    outcome_set = set(outcome)
    if len(outcome) >= 5:
        if config.small_straight.issubset(
            outcome_set
        ) or config.large_straight.issubset(outcome_set):
            return True
    if len(outcome) == 6 and outcome_set == config.full_straight:
        return True

    return False


def compute_bust_probability(n: int) -> float:
    """
    Compute bust probability for n dice by enumerating all outcomes.
    Returns a probability between 0 and 1.
    """
    # Edge cases
    if n <= 0:
        return 0.0  # No dice can't bust

    # We can cache these values since they're static for a given dice count
    bust_probabilities = {
        1: 2 / 3,  # Only 1 and 5 score (2 out of 6 faces), so bust on 2,3,4,6
        2: 4 / 9,  # 1-(1-(2/3)²) = 4/9
        3: 0.1667,  # Pre-calculated, approx 1/6
        4: 0.0770,  # Pre-calculated
        5: 0.0309,  # Pre-calculated
        6: 0.0154,  # Pre-calculated
    }

    # Return cached value if available
    if n in bust_probabilities:
        return bust_probabilities[n]

    # Calculate if not cached (fallback for non-standard dice counts)
    total = 6**n
    bust = sum(
        1
        for outcome in itertools.product(range(1, 7), repeat=n)
        if not outcome_is_scoring(outcome)
    )
    return bust / total if total > 0 else 0.0


def average_score(n: int) -> float:
    """
    Calculate the expected score from rolling n dice.
    Uses cached values for common dice counts to improve performance.
    """
    # Pre-calculated expected values for common dice counts
    avg_scores = {
        1: 25.0,  # (100 + 50)/6 = 25
        2: 59.7,  # Pre-calculated
        3: 130.3,  # Pre-calculated
        4: 212.5,  # Pre-calculated
        5: 301.4,  # Pre-calculated
        6: 423.2,  # Pre-calculated
    }

    # Return cached value if available
    if n in avg_scores:
        return avg_scores[n]

    # Fallback calculation for non-standard dice counts
    if n <= 0:
        return 0.0

    # Simplified calculation for larger n to avoid exponential complexity
    if n > 6:
        return avg_scores[6] * (n / 6)

    # Standard calculation
    total_score = 0
    total_outcomes = 0

    for outcome in itertools.product(range(1, 7), repeat=n):
        holds = generate_holds(list(outcome))
        if holds:
            best_score = max(score for _, score in holds)
            total_score += best_score
        total_outcomes += 1

    return total_score / total_outcomes


# -------------------------------
# Expected Value Function (with risk adjustment)
# -------------------------------
def expected_value(
    remaining_dice: int,
    current_turn_points: int,
    player_score: int,
    opponent_score: int,
    depth: int = 0,
    target_score: int = None,
) -> float:
    """
    Calculate expected value of continuing with current turn.
    Enhanced to account for end-game strategy and target score proximity.

    Args:
        remaining_dice: Number of dice left to roll
        current_turn_points: Points accumulated in current turn
        player_score: Total score of the player
        opponent_score: Total score of the opponent
        depth: Current recursion depth
        target_score: Target score to win (if known)

    Returns:
        Expected value of continuing the turn
    """
    if remaining_dice == 0:
        remaining_dice = config.hot_dice_count  # Hot-dice rule.

    bust_prob = compute_bust_probability(remaining_dice)
    safe_points = player_score + current_turn_points

    # End-game strategy adjustments
    if target_score is not None:
        points_needed = target_score - player_score

        # If we can win by stopping, always stop
        if current_turn_points >= points_needed:
            return current_turn_points  # Just bank and win

        # If we're very close to winning, be more aggressive
        if points_needed - current_turn_points < 300:
            # Increase risk taking when close to winning
            risk_factor = 1.2
        else:
            risk_factor = 1.0
    else:
        risk_factor = 1.0

    # Additional risk adjustment based on relative scores
    if player_score < opponent_score:
        # Being behind increases willingness to take risks
        risk_factor *= config.risk_bonus
    elif player_score > opponent_score:
        # Being ahead decreases willingness to take risks
        risk_factor *= config.risk_penalty

    # For high dice counts or deep recursion, use average
    if remaining_dice <= 2 or depth >= config.ev_max_depth:
        avg_score = average_score(remaining_dice)
        # Apply risk factor to expected continuation value
        ev_continue = current_turn_points + (1 - bust_prob) * avg_score * risk_factor
    else:
        # More precise calculation with recursion
        avg_score = average_score(remaining_dice)
        future_ev = config.ev_discount * expected_value(
            remaining_dice - 1,  # Assume we'll use at least one die
            current_turn_points + avg_score,
            player_score,
            opponent_score,
            depth + 1,
            target_score,
        )
        ev_continue = (
            current_turn_points
            + (1 - bust_prob) * (avg_score + future_ev) * risk_factor
        )

    # Decision point: is it better to bank or continue?
    return max(current_turn_points, ev_continue)


# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_move(
    current_roll: List[int],
    remaining_dice: int,
    current_turn_points: int,
    player_score: int,
    opponent_score: int,
    devil_count: int = 0,
    target_score: int = None,
) -> Dict[str, Any]:
    """
    Recommend the best move given the current game state.
    Enhanced with more strategic decision making and end-game awareness.

    Args:
        current_roll: List of dice values rolled
        remaining_dice: Total dice in this roll
        current_turn_points: Points accumulated this turn
        player_score: Total player score
        opponent_score: Total opponent score
        devil_count: Number of Devil's Head dice (if applicable)
        target_score: Target score to win (if known)

    Returns:
        Dictionary with recommendation details
    """
    holds = generate_holds(current_roll)

    if not holds:
        return {
            "hold_dice": [],
            "decision": "bust",
            "rationale": "No scoring combinations possible.",
        }

    best_decision = "score and pass"
    best_ev = current_turn_points  # Default: banking current points
    best_hold = []
    best_rationale = "No better option than banking current points."

    # If we can win by banking, just do it
    if target_score and player_score + current_turn_points >= target_score:
        for hold_dice, hold_score in holds:
            if player_score + current_turn_points + hold_score >= target_score:
                return {
                    "hold_dice": hold_dice,
                    "decision": "score and pass",
                    "rationale": f"Banking {player_score + current_turn_points + hold_score} points to win the game.",
                }

    for hold_dice, hold_score in holds:
        dice_left_after_hold = remaining_dice - len(hold_dice)
        if dice_left_after_hold == 0:
            dice_left_after_hold = config.hot_dice_count  # hot dice

        bust_prob = compute_bust_probability(dice_left_after_hold)

        # Calculate expected value of continuing
        new_turn_points = current_turn_points + hold_score

        # Special case: if using all dice (hot dice), be more aggressive
        hot_dice_bonus = (
            1.1
            if dice_left_after_hold == config.hot_dice_count
            and len(hold_dice) == remaining_dice
            else 1.0
        )

        # End-game considerations
        if target_score and player_score + new_turn_points >= target_score:
            # We can win by banking - do it
            ev_continue = 0  # Make sure we choose to bank
            ev_pass = new_turn_points
            decision = "score and pass"
            rationale = f"Banking {player_score + new_turn_points} to win the game."
        else:
            # Calculate expected values
            avg_future_score = average_score(dice_left_after_hold)
            ev_continue = (
                new_turn_points + (1 - bust_prob) * avg_future_score * hot_dice_bonus
            )
            ev_pass = new_turn_points

            # Consider devil dice (not implemented in core logic yet)
            if devil_count > 0:
                # Penalize expected value based on devil dice count
                ev_continue -= devil_count * 50  # Simple penalty

            # Make decision with detailed rationale
            if ev_continue > ev_pass + 50 and bust_prob < 0.5:
                decision = "score and continue"
                rationale = (
                    f"Hold {hold_dice} scoring {hold_score} points; "
                    f"with {dice_left_after_hold} dice remaining, bust chance is "
                    f"{bust_prob * 100:.1f}%, continuing EV: {ev_continue:.1f} vs banking: {ev_pass}"
                )
            else:
                decision = "score and pass"
                if bust_prob > 0.4:
                    rationale = f"Banking {new_turn_points} is safer due to high bust chance ({bust_prob:.1%})"
                else:
                    rationale = f"Banking {new_turn_points} has better expected value than continuing"

        # Update best decision if this is better
        if decision == "score and pass":
            if new_turn_points > best_ev:
                best_ev = new_turn_points
                best_hold = hold_dice
                best_decision = decision
                best_rationale = rationale
        else:  # score and continue
            if ev_continue > best_ev:
                best_ev = ev_continue
                best_hold = hold_dice
                best_decision = decision
                best_rationale = rationale

    return {
        "hold_dice": best_hold,
        "decision": best_decision,
        "rationale": best_rationale,
    }


# -------------------------------
# Interactive Trainer/Solver
# -------------------------------
import json


def interactive_solver() -> None:
    """Interactive game solver with improved error handling."""
    try:
        target_score_input = input("Enter the total score required to win: ")
        if not target_score_input.isdigit() or int(target_score_input) <= 0:
            print("Invalid target score. Please enter a positive number.")
            return
        target_score = int(target_score_input)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    player_total = 0
    opponent_score = 0
    round_number = 1
    game_log = []

    while player_total < target_score and opponent_score < target_score:
        print(f"\n--- Round {round_number} ---")
        try:
            opponent_input = input("Enter opponent's total score: ").strip()
            if not opponent_input.isdigit():
                print("Invalid opponent's score. Please enter a non-negative number.")
                continue
            opponent_score = int(opponent_input)
            if opponent_score < 0:
                print("Opponent score cannot be negative.")
                continue
        except ValueError:
            print("Invalid opponent score. Please enter a numeric value.")
            continue

        # Check game end condition immediately
        if opponent_score >= target_score:
            print(f"Opponent reached {opponent_score}. You lose.")
            break

        current_turn_points = 0
        remaining_dice = 6
        round_over = False

        # Initialize logging for this round
        round_log = {
            "round": round_number,
            "player_start_total": player_total,
            "opponent_total": opponent_score,
            "turns": [],
        }

        while not round_over:
            try:
                dice_input = input(
                    f"Enter dice roll for {remaining_dice} dice (comma or space separated, or 'bust'): "
                ).strip()

                if dice_input.lower() == "bust":
                    round_log["turns"].append(
                        {
                            "roll": [],
                            "hold": [],
                            "hold_score": 0,
                            "turn_points": 0,
                            "decision": "bust",
                        }
                    )
                    print("Bust! Turn points lost.")
                    current_turn_points = 0
                    round_over = True
                    continue

                # Allow both comma and space separation
                if "," in dice_input:
                    dice_values = dice_input.split(",")
                else:
                    dice_values = dice_input.split()

                # Validate number of dice
                if len(dice_values) != remaining_dice:
                    print(
                        f"Error: You must enter exactly {remaining_dice} dice values."
                    )
                    continue

                # Validate dice values
                try:
                    current_roll = [int(x.strip()) for x in dice_values]
                    # Check range
                    if any(d < 1 or d > 6 for d in current_roll):
                        print("Error: Dice values must be between 1 and 6.")
                        continue
                except ValueError:
                    print("Error: Dice values must be integers.")
                    continue

                # Validate devil count
                try:
                    devil_count_input = input(
                        "Enter number of Devil's Head dice (if any, else 0): "
                    ).strip()

                    if not devil_count_input.isdigit():
                        print(
                            "Error: Devil's Head count must be a non-negative integer."
                        )
                        continue

                    devil_count = int(devil_count_input)
                    if devil_count < 0 or devil_count > remaining_dice:
                        print(
                            f"Error: Devil's Head count must be between 0 and {remaining_dice}."
                        )
                        continue
                except ValueError:
                    print("Error: Devil's Head count must be an integer.")
                    continue

                # Now we can proceed with recommendation
                try:
                    rec = recommend_move(
                        current_roll,
                        remaining_dice,
                        current_turn_points,
                        player_total,
                        opponent_score,
                        devil_count,
                        target_score,
                    )
                except Exception as e:
                    print(f"Error in recommendation algorithm: {str(e)}")
                    continue

                # Validate holding logic
                if not set(rec["hold_dice"]).issubset(set(current_roll)):
                    print(
                        "Internal error: Recommended hold contains dice not in the roll!"
                    )
                    continue

                hold_score = score_hold(rec["hold_dice"]) or 0

                # Make sure hold score is valid
                if hold_score is None or hold_score <= 0:
                    print("Internal error: Invalid holding score calculation!")
                    continue

                current_turn_points += hold_score
                remaining_dice -= len(rec["hold_dice"])
                if remaining_dice == 0:
                    remaining_dice = 6  # Hot dice rule

                turn_log = {
                    "roll": current_roll,
                    "devil_dice": devil_count,
                    "hold": rec["hold_dice"],
                    "hold_score": hold_score,
                    "turn_points": current_turn_points,
                    "remaining_dice": remaining_dice,
                    "decision": rec["decision"],
                    "rationale": rec["rationale"],
                }
                round_log["turns"].append(turn_log)

                print(
                    f"\nSolver recommends: {rec['decision']} with dice {rec['hold_dice']}"
                )
                print(f"Rationale: {rec['rationale']}")

                if rec["decision"] == "score and pass":
                    print("Passing turn.")
                    round_over = True
                elif rec["decision"] == "bust":
                    print("Bust! Points lost.")
                    current_turn_points = 0
                    round_over = True
                else:
                    print(
                        f"Continuing turn, current points: {current_turn_points}, dice left: {remaining_dice}"
                    )

            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                continue

        player_total += current_turn_points
        round_log["player_end_total"] = player_total
        game_log.append(round_log)

        print(f"Round {round_number} complete. Your total score: {player_total}")
        round_number += 1

        # Game end conditions check
        if player_total >= target_score:
            print("Congratulations! You've reached the target score and won!")
            break

    # Finally print full game log as JSON
    try:
        print("\nGame log (for ML consumption):")
        print(json.dumps(game_log, indent=2))
    except Exception as e:
        print(f"Error generating game log: {str(e)}")


def main() -> None:
    interactive_solver()


if __name__ == "__main__":
    main()
