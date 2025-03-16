#!/usr/bin/env python3
import random
from collections import Counter
import itertools
import math
from typing import Any, Dict, List, Tuple, Optional

# -------------------------------
# Scoring Helper Functions
# -------------------------------
def score_trip(count: int, num: int) -> int:
    """Calculate score for three or more of a kind for a given number."""
    base = 1000 if num == 1 else num * 100
    return base * (2 ** (count - 3))

def is_straight(counts: Counter, dice_count: int) -> Optional[Tuple[int, List[int]]]:
    """
    Check for straights.
    Returns (score, dice used) if a straight is found.
      - 6-dice straight (1-2-3-4-5-6) for 1500 points,
      - 5-dice straights: 1-2-3-4-5 for 500, or 2-3-4-5-6 for 750.
    """
    if dice_count == 6 and all(counts.get(i, 0) >= 1 for i in range(1, 7)):
        return (1500, list(range(1, 7)))
    if dice_count >= 5:
        if all(counts.get(i, 0) >= 1 for i in range(1, 6)):
            return (500, [1, 2, 3, 4, 5])
        if all(counts.get(i, 0) >= 1 for i in range(2, 7)):
            return (750, [2, 3, 4, 5, 6])
    return None

def generate_holds(roll: List[int], devil_count: int = 0) -> List[Tuple[List[int], int]]:
    """
    Given a roll (list of dice) and an optional count for Devil's Head dice,
    generate all valid scoring holds (one scoring combination per roll).
    Returns a list of tuples: (dice held, score earned).
    """
    holds = []
    counts = Counter(roll)
    total_dice = len(roll) + devil_count

    # Check for straights.
    straight = is_straight(counts, len(roll))
    if straight:
        score, dice_used = straight
        holds.append((dice_used, score))
    
    # Three or more of a kind.
    for num in range(1, 7):
        cnt = counts.get(num, 0)
        effective = cnt + devil_count
        if effective >= 3:
            for used in range(3, min(effective, total_dice) + 1):
                actual_used = min(cnt, used)
                wild_used = used - actual_used
                if wild_used <= devil_count:
                    holds.append(([num] * used, score_trip(used, num)))
    
    # Single scoring dice: 1's and 5's.
    for num in [1, 5]:
        cnt = counts.get(num, 0)
        for i in range(1, cnt + 1):
            score = 100 * i if num == 1 else 50 * i
            holds.append(([num] * i, score))
    
    # Remove duplicates.
    unique = {}
    for dice, score in holds:
        key = (tuple(sorted(dice)), score)
        unique[key] = (list(dice), score)
    return list(unique.values())

# -------------------------------
# Risk & Probability Functions
# -------------------------------
def outcome_is_scoring(outcome: Tuple[int, ...]) -> bool:
    """
    Determine if a given outcome is scoring.
    An outcome scores if it contains any 1 or 5,
    three or more of a kind, or a valid straight.
    """
    if any(d in (1, 5) for d in outcome):
        return True
    counts = Counter(outcome)
    if any(cnt >= 3 for cnt in counts.values()):
        return True
    if len(outcome) >= 5:
        s = set(outcome)
        if set(range(1, 6)).issubset(s) or set(range(2, 7)).issubset(s):
            return True
    if len(outcome) == 6 and set(outcome) == set(range(1, 7)):
        return True
    return False

def compute_bust_probability(n: int) -> float:
    """
    Compute bust probability for n dice by enumerating outcomes.
    Returns a probability between 0 and 1.
    """
    total = 6 ** n
    bust = sum(1 for outcome in itertools.product(range(1, 7), repeat=n)
               if not outcome_is_scoring(outcome))
    return bust / total if total > 0 else 0.0

def average_score(n: int) -> float:
    """
    Estimate the average additional score available from rolling n dice.
    """
    total_score = 0
    count = 0
    for outcome in itertools.product(range(1, 7), repeat=n):
        holds = generate_holds(list(outcome))
        if holds:
            best = max(score for _, score in holds)
            total_score += best
        count += 1
    return total_score / count if count else 0

def expected_value(remaining_dice: int, current_turn_points: int,
                   player_score: int, opponent_score: int,
                   depth: int = 0, max_depth: int = 4) -> float:
    """
    Recursively estimate the expected value (EV) of continuing the turn.
    Weigh the chance of busting against potential gains.
    """
    if remaining_dice == 0:
        remaining_dice = 6  # Hot-dice rule: re-roll all if all dice scored.
    bust_prob = compute_bust_probability(remaining_dice)
    if depth >= max_depth:
        add_score = average_score(remaining_dice)
        return (1 - bust_prob) * (current_turn_points + add_score)
    add_score = average_score(remaining_dice)
    cont_ev = (1 - bust_prob) * expected_value(remaining_dice,
                                                current_turn_points + add_score,
                                                player_score, opponent_score,
                                                depth + 1, max_depth)
    ev_continue = (1 - bust_prob) * (current_turn_points + add_score + cont_ev)
    # Adjust EV based on score difference (more aggressive when behind).
    score_diff = player_score - opponent_score
    if score_diff < 0:
        ev_continue *= 1.1
    elif score_diff > 0:
        ev_continue *= 0.9
    return max(current_turn_points, ev_continue)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_move(current_roll: List[int],
                   remaining_dice: int,
                   current_turn_points: int,
                   player_score: int,
                   opponent_score: int,
                   devil_count: int = 0) -> Dict[str, Any]:
    """
    For the given roll and state, provide a recommendation:
      - 'hold_dice': the scoring combination to select,
      - 'decision': "score and continue" or "score and pass" (or "bust" if no scoring hold),
      - 'rationale': explanation with bust probability and EV estimates.
    """
    holds = generate_holds(current_roll, devil_count)
    if not holds:
        return {"hold_dice": [],
                "decision": "bust",
                "rationale": "No scoring combinations available; you bust and lose all turn points."}
    best_ev = -math.inf
    best_hold = None
    best_decision = "score and pass"
    rationale = ""
    for dice_hold, hold_score in holds:
        dice_used = len(dice_hold)
        new_turn_points = current_turn_points + hold_score
        new_remaining = remaining_dice - dice_used
        if new_remaining == 0:
            new_remaining = 6  # Hot-dice: re-roll all if all dice are scored.
        ev_continue = expected_value(new_remaining, new_turn_points, player_score, opponent_score)
        ev_pass = new_turn_points  # Banking the points.
        if ev_continue >= ev_pass:
            decision = "score and continue"
            ev_choice = ev_continue
        else:
            decision = "score and pass"
            ev_choice = ev_pass
        if ev_choice > best_ev:
            best_ev = ev_choice
            best_hold = dice_hold
            best_decision = decision
            rationale = (f"Hold {dice_hold} scoring {hold_score} points; "
                         f"with {new_remaining} dice remaining, bust chance is "
                         f"{compute_bust_probability(new_remaining)*100:.1f}%, "
                         f"EV if continuing: {ev_continue:.1f} vs banking: {ev_pass} points.")
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
    # Player always goes first; we assume opponent scores are input each round.
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
            dice_input = input(f"Enter dice roll for {remaining_dice} dice (space-separated): ")
            try:
                current_roll = list(map(int, dice_input.strip().split()))
            except ValueError:
                print("Invalid dice roll input.")
                continue
            devil_count_input = input("Enter number of Devil's Head dice (if any, else 0): ")
            try:
                devil_count = int(devil_count_input)
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
            holds = generate_holds(current_roll, devil_count)
            chosen_hold = rec["hold_dice"]
            hold_score: Optional[int] = None
            for hold, score in holds:
                if sorted(hold) == sorted(chosen_hold):
                    hold_score = score
                    break
            if hold_score is None:
                print("Error: recommended hold not found. Ending round.")
                round_over = True
                continue
            
            current_turn_points += hold_score
            remaining_dice -= len(chosen_hold)
            if remaining_dice == 0:
                remaining_dice = 6  # Hot-dice rule.
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
