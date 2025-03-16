#!/usr/bin/env python3
import random
from collections import Counter
import itertools
import math
from typing import Any, Dict, List, Tuple, Optional

# -------------------------------
# Scoring rules:
#   - Single 1: 100; single 5: 50.
#   - Three of a kind: 1's score 1000; for others, face * 100.
#     Each additional die beyond three doubles the base triple score.
#   - Straights: 1-2-3-4-5 = 500; 2-3-4-5-6 = 750; 1-2-3-4-5-6 = 1500.
# -------------------------------

def score_trip(count: int, num: int) -> int:
    """Return score for k-of-a-kind for a given face, where k >= 3."""
    base = 1000 if num == 1 else num * 100
    return base * (2 ** (count - 3))

# --- Exhaustive hold scoring functions ---

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
                score_val = k * (100 if face == 1 else 50)
                groups.append((group, score_val))
    # For three-or-more of a kind (for any face)
    for face in cnt:
        if cnt[face] >= 3:
            for k in range(3, cnt[face] + 1):
                group = Counter({face: k})
                score_val = score_trip(k, face)
                groups.append((group, score_val))
    # Straights (only valid if exactly the required dice are used)
    # Check for 1-2-3-4-5-6
    if all(cnt.get(i, 0) >= 1 for i in range(1,7)):
        group = Counter({i:1 for i in range(1,7)})
        groups.append((group, 1500))
    # Check for 1-2-3-4-5 (only if available)
    if all(cnt.get(i, 0) >= 1 for i in range(1,6)):
        group = Counter({i:1 for i in range(1,6)})
        groups.append((group, 500))
    # Check for 2-3-4-5-6
    if all(cnt.get(i, 0) >= 1 for i in range(2,7)):
        group = Counter({i:1 for i in range(2,7)})
        groups.append((group, 750))
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
    cnt = Counter(hold)
    return best_score_from_counter(cnt)

def generate_holds(roll: List[int]) -> List[Tuple[List[int], int]]:
    """
    Generate all valid holds from the roll.
    Each hold is a subset (order ignored) of dice from roll that is completely scoring.
    Returns a list of tuples: (sorted list of dice in hold, maximum score).
    """
    n = len(roll)
    holds_dict: Dict[Tuple[int, ...], int] = {}
    for r in range(1, n+1):
        for indices in itertools.combinations(range(n), r):
            subset = [roll[i] for i in indices]
            sorted_subset = tuple(sorted(subset))
            if sorted_subset in holds_dict:
                continue
            s = score_hold(list(sorted_subset))
            if s is not None and s > 0:
                holds_dict[sorted_subset] = s
    # Convert to list of (list, score)
    return [(list(k), v) for k, v in holds_dict.items()]

# -------------------------------
# Risk & Probability Functions
# -------------------------------

def outcome_is_scoring(outcome: Tuple[int, ...]) -> bool:
    """
    Determine if an outcome (tuple of dice) is scoring.
    An outcome is scoring if at least one nonempty subset of it scores.
    (For bust probability, we consider an outcome scoring if it contains at least one scoring die.)
    """
    # A simple check: if any die is 1 or 5, or if there are at least three of any face,
    # or if it qualifies as a straight.
    if any(d in (1,5) for d in outcome):
        return True
    counts = Counter(outcome)
    if any(cnt >= 3 for cnt in counts.values()):
        return True
    if len(outcome) >= 5:
        s = set(outcome)
        if set(range(1,6)).issubset(s) or set(range(2,7)).issubset(s):
            return True
    if len(outcome) == 6 and set(outcome) == set(range(1,7)):
        return True
    return False

def compute_bust_probability(n: int) -> float:
    """
    Compute bust probability for n dice by enumerating all outcomes.
    Returns a probability between 0 and 1.
    """
    total = 6 ** n
    bust = sum(1 for outcome in itertools.product(range(1,7), repeat=n)
               if not outcome_is_scoring(outcome))
    return bust / total if total > 0 else 0.0

def average_score(n: int) -> float:
    if n == 1:
        return (1/6)*100 + (1/6)*50  # Exactly 25 points
    # use existing logic for other cases (n > 1)
    total_score = 0
    outcomes = itertools.product(range(1, 7), repeat=n)
    for outcome in outcomes:
        holds = generate_holds(list(outcome))
        if holds:
            best_score = min(score for _, score in holds)
            total_score += best_score
    return total_score / (6 ** n)


# -------------------------------
# Expected Value Function (with risk penalty)
# -------------------------------

def expected_value(
    remaining_dice: int,
    current_turn_points: int,
    player_score: int,
    opponent_score: int,
    depth: int = 0,
    max_depth: int = 2,
    discount: float = 0.5,
) -> float:
    if remaining_dice == 0:
        remaining_dice = 6  # Hot-dice rule.

    bust_prob = compute_bust_probability(remaining_dice)

    # For 1 or 2 dice, explicitly calculate EV without recursion
    if remaining_dice <= 2 or depth >= max_depth:
        avg_score = average_score(remaining_dice)
        ev_continue = current_turn_points + (1 - bust_prob) * avg_score
    else:
        avg_score = average_score(remaining_dice)
        future_ev = discount * expected_value(
            remaining_dice,
            current_turn_points + avg_score,
            player_score,
            opponent_score,
            depth + 1,
            max_depth,
            discount,
        )
        ev_continue = current_turn_points + (1 - bust_prob) * (avg_score + future_ev)

    # Slight strategic adjustments based on scores
    if player_score < opponent_score:
        ev_continue *= 1.05  # Small aggression bonus
    elif player_score > opponent_score:
        ev_continue *= 0.95  # Small conservative penalty

    return max(current_turn_points, ev_continue)



# -------------------------------
# Recommendation Function
# -------------------------------

def recommend_move(current_roll: List[int], remaining_dice: int,
                   current_turn_points: int, player_score: int, opponent_score: int,
                   devil_count: int = 0) -> Dict[str, Any]:
    holds = generate_holds(current_roll)
    best_decision = "score and pass"
    best_ev = -math.inf
    best_hold = []
    rationale = ""

    for hold_dice, hold_score in holds:
        dice_left_after_hold = remaining_dice - len(hold_dice)
        if dice_left_after_hold == 0:
            dice_left_after_hold = 6  # hot dice
        bust_prob = compute_bust_probability(dice_left_after_hold)
        avg_future_score = average_score(dice_left_after_hold)

        # Explicit simplified EV calculation (no recursion at small dice count)
        ev_continue = current_turn_points + hold_score + (1 - bust_prob) * average_score(dice_left_after_hold)
        ev_pass = current_turn_points + hold_score

        # Explicit small-margin check to avoid risky plays:
        if ev_continue > ev_pass + 50 and bust_prob < 0.5:
            decision = "score and continue"
            ev = ev_continue
            rationale_tmp = (f"Hold {hold_dice} scoring {hold_score} points; with "
                             f"{dice_left_after_hold} dice remaining, bust chance is "
                             f"{bust_prob*100:.1f}%, EV continuing: {ev_continue:.1f} vs banking: {ev_pass}")
        else:
            decision = "score and pass"
            ev = ev_pass
            rationale_pass = f"Banking {ev_pass} is safer due to high bust chance ({bust_prob:.1%})"

        if ev > best_ev:
            best_ev = ev
            best_hold = hold_dice
            best_decision = decision
            rationale = rationale_pass if decision == "score and pass" else rationale

    return {"hold_dice": best_hold, "decision": best_decision, "rationale": rationale}


# -------------------------------
# Interactive Trainer/Solver
# -------------------------------

def interactive_solver() -> None:
    try:
        target_score = int(input("Enter the total score required to win: "))
    except ValueError:
        print("Invalid target score.")
        return
    player_total = 0
    round_number = 1
    while player_total < target_score:
        print(f"\n--- Round {round_number} ---")
        try:
            opponent_score = int(input("Enter opponent's total score: "))
        except ValueError:
            print("Invalid input for opponent's score.")
            continue
        current_turn_points = 0
        remaining_dice = 6
        round_over = False
        while not round_over:
            dice_input = input(f"Enter dice roll for {remaining_dice} dice (space-separated or type 'bust'): ").strip()
            if dice_input.lower() == "bust":
                print("Input received: bust. Round ended; you lose all turn points.")
                current_turn_points = 0
                round_over = True
                continue
            try:
                current_roll = list(map(int, dice_input.split()))
            except ValueError:
                print("Invalid dice roll input.")
                continue
            devil_count_input = input("Enter number of Devil's Head dice (if any, else 0): ").strip()
            try:
                devil_count = int(devil_count_input) if devil_count_input else 0
            except ValueError:
                devil_count = 0
            rec = recommend_move(current_roll, remaining_dice, current_turn_points,
                                 player_total, opponent_score, devil_count)
            print("\nSolver Recommendation:")
            print(f"  Score dice: {rec['hold_dice']}")
            print(f"  Decision: {rec['decision']}")
            print(f"  Rationale: {rec['rationale']}")
            
            if rec["decision"] == "bust":
                print("Round busted! You lose all turn points.")
                current_turn_points = 0
                round_over = True
                continue
            
            # Find the hold score for the recommended hold.
            holds_list = generate_holds(current_roll)
            chosen_hold = rec["hold_dice"]
            hold_score: Optional[int] = None
            for hold, score in holds_list:
                if sorted(hold) == sorted(chosen_hold):
                    hold_score = score
                    break
            if hold_score is None:
                print("Error: recommended hold not found. Ending round.")
                round_over = True
                continue
            
            current_turn_points += hold_score
            remaining_dice -= len(chosen_hold)
            if remaining_dice <= 0:
                remaining_dice = 6
                print("Hot-dice! All dice scored. Re-rolling all 6 dice.")
            
            if rec["decision"] == "score and pass":
                print("Decision: Score and pass. Ending round.")
                round_over = True
            else:
                print("Decision: Score and continue. Continue rolling.")
                print(f"Current turn points: {current_turn_points}, Remaining dice: {remaining_dice}")
        
        player_total += current_turn_points
        print(f"Round {round_number} complete. Your total score: {player_total}")
        round_number += 1
    print("\nCongratulations! You've reached the target score and won!")

def main() -> None:
    interactive_solver()

if __name__ == "__main__":
    main()
