"""my-team playbook — formations, playing styles and when to switch between them.

Five game plans, ordered from most defensive to most attacking:

    PARK_BUS   2-1-1  everyone behind the ball, protect the lead
    COUNTER    2-1-1  sit deep, win it, hit the striker fast
    TIKI_TAKA  1-2-1  keep the ball with short passes, win it back fast
    PRESS      1-1-2  press high in their half, two strikers
    ALL_OUT    1-0-3  three attackers, shoot on sight

The plan moves up or down this ladder with the score and the clock, and the coach
can pick one by name ("tiki taka", "2-1-1", "park the bus", ... and "auto" to hand
control back). Every tick each player is told its role, the team style, and what
the style means for it right now: with the ball, or without it.

"1-1-3" is not possible with 4 outfield players, so it is read as 1-0-3.
"""

import re

from parsing import FIELD_Y

START_PLAN = "TIKI_TAKA"                       # plan at kick-off when the score is level
LOCKED_PLAN = None                             # set to a plan name to play only that plan (no switching)
LADDER = ["PARK_BUS", "COUNTER", "TIKI_TAKA", "PRESS", "ALL_OUT"]

PLANS = {
    "PARK_BUS":  {"formation": "2-1-1", "name": "Park the bus", "shift": -18},
    "COUNTER":   {"formation": "2-1-1", "name": "Counter-attack", "shift": -10},
    "TIKI_TAKA": {"formation": "1-2-1", "name": "Tiki-taka", "shift": 0},
    "PRESS":     {"formation": "1-1-2", "name": "High press", "shift": 8},
    "ALL_OUT":   {"formation": "1-0-3", "name": "All-out attack", "shift": 12},
}

# Who plays where (player id → role) in each formation. The GK (0) is always GK.
FORMATIONS = {
    "2-1-1": {1: "LEFT_CB", 2: "RIGHT_CB", 3: "PLAYMAKER", 4: "STRIKER"},
    "1-2-1": {1: "ANCHOR", 2: "LEFT_MID", 3: "RIGHT_MID", 4: "STRIKER"},
    "1-1-2": {1: "ANCHOR", 2: "PLAYMAKER", 3: "LEFT_FWD", 4: "RIGHT_FWD"},
    "1-0-3": {1: "SWEEPER", 2: "LEFT_WING", 3: "STRIKER", 4: "RIGHT_WING"},
}

# Home spot per role in our frame (we attack +x): (base_x, base_y, x_min, x_max).
# Goal lines are at x=-50 / +50 (parsing.GOAL_X); spots stay in front of them.
# The spot slides 40% with the ball's x and 30% with its y, plus the plan's shift.
SPOTS = {
    "GK": (-48, 0, -49, -45),
    "LEFT_CB": (-35, -10, -46, 0), "RIGHT_CB": (-35, 10, -46, 0),
    "ANCHOR": (-30, 0, -44, 5), "SWEEPER": (-22, 0, -40, 15),
    "PLAYMAKER": (-5, 0, -30, 30), "LEFT_MID": (-5, -12, -30, 30), "RIGHT_MID": (-5, 12, -30, 30),
    "STRIKER": (20, 0, -5, 45), "LEFT_FWD": (18, -12, -10, 44), "RIGHT_FWD": (18, 12, -10, 44),
    "LEFT_WING": (12, -20, -15, 44), "RIGHT_WING": (12, 20, -15, 44),
}

# What the whole team does in each plan: (with the ball, without the ball).
STYLE = {
    "PARK_BUS": (
        "Keep it safe: clear it long (KICK/AERIAL) or keep it far from our goal. No risky passes near our goal.",
        "Everyone defends in our half: MARK TIGHT, block the middle. No SLIDE_TACKLE in our box. Striker drops to midfield.",
    ),
    "COUNTER": (
        "Go forward FAST: first look for a THROUGH or AERIAL pass to the STRIKER; carry the ball if the lane is open.",
        "Sit deep and compact, let them come. MARK the attackers near our goal; PRESSER presses only in our half.",
    ),
    "TIKI_TAKA": (
        "Keep the ball: short GROUND passes to the nearest OPEN mate, make triangles, be patient. Shoot only when goal < 20.",
        "Win it back fast: PRESSER presses hard; the next closest cuts the passing lane with INTERCEPT; others stay compact.",
    ),
    "PRESS": (
        "Attack quickly: forward passes, shoot as soon as goal < 25.",
        "Press high in their half: PRESSER uses PRESS_BALL intensity 0.85; others MARK TIGHT the nearest opponent to block passes.",
    ),
    "ALL_OUT": (
        "Everyone attacks: run into the box, pass only forward, SHOOT on sight when goal < 28.",
        "Press everywhere and sprint to win it back; SLIDE_TACKLE only in their half. Only the SWEEPER stays back.",
    ),
}

# What each role does: (with the ball, without the ball).
ROLES = {
    "GK": (
        "GK_DISTRIBUTE to the most OPEN mate: THROW if near, KICK if far (KICK long in COUNTER/PARK_BUS).",
        "Stay between the ball and the centre of our goal. Ball free within 10 of goal: INTERCEPT aggressive=true.",
    ),
    "LEFT_CB": (
        "Stay back on the left (y<0). Simple GROUND pass to the PLAYMAKER or an OPEN mate.",
        "Guard the left of our box: MARK TIGHT the nearest attacker on your side. Never leave the box with the other CB.",
    ),
    "RIGHT_CB": (
        "Stay back on the right (y>0). Simple GROUND pass to the PLAYMAKER or an OPEN mate.",
        "Guard the right of our box: MARK TIGHT the nearest attacker on your side. Never leave the box with the other CB.",
    ),
    "ANCHOR": (
        "Offer a safe pass behind the ball. Pass forward to an OPEN midfielder.",
        "Stay between the ball and our goal: MARK TIGHT their most advanced attacker.",
    ),
    "SWEEPER": (
        "Stay behind everyone around half-way; move the ball forward quickly.",
        "Last defender: stay between ball and goal, INTERCEPT through balls, do not dive in.",
    ),
    "PLAYMAKER": (
        "The link: receive, turn, THROUGH pass to the forwards. SHOOT if goal < 22.",
        "Screen the defence in the middle: INTERCEPT passes; PRESS_BALL if PRESSER.",
    ),
    "LEFT_MID": (
        "Support on the left ahead of the ball; GROUND or THROUGH pass to the STRIKER. SHOOT if goal < 22.",
        "Close down the left: cut passing lanes with INTERCEPT, track runners on your side.",
    ),
    "RIGHT_MID": (
        "Support on the right ahead of the ball; GROUND or THROUGH pass to the STRIKER. SHOOT if goal < 22.",
        "Close down the right: cut passing lanes with INTERCEPT, track runners on your side.",
    ),
    "STRIKER": (
        "Stay level with the OFFSIDE LINE and run in behind only as the pass is played. SHOOT when goal < 22, aim a corner (TL/TR/BL/BR).",
        "PRESS_BALL their defender if PRESSER; otherwise stay high, ready for the counter.",
    ),
    "LEFT_FWD": (
        "Attack the left side of the box, stay onside. SHOOT when goal < 22 (aim far corner TR/BR), else pass to the RIGHT_FWD if OPEN.",
        "PRESS_BALL if PRESSER; otherwise MARK LOOSE the nearest defender, stay high.",
    ),
    "RIGHT_FWD": (
        "Attack the right side of the box, stay onside. SHOOT when goal < 22 (aim far corner TL/BL), else pass to the LEFT_FWD if OPEN.",
        "PRESS_BALL if PRESSER; otherwise MARK LOOSE the nearest defender, stay high.",
    ),
    "LEFT_WING": (
        "Stay wide left, stay onside, run to the box. Pass inside to the STRIKER or SHOOT when goal < 22.",
        "Press the nearest defender on the left; sprint back only if the ball is in our half.",
    ),
    "RIGHT_WING": (
        "Stay wide right, stay onside, run to the box. Pass inside to the STRIKER or SHOOT when goal < 22.",
        "Press the nearest defender on the right; sprint back only if the ball is in our half.",
    ),
}


# --- Choosing the plan -----------------------------------------------------------

def auto_plan(score_diff: int, time_left: float, match_seconds: float) -> str:
    """Start from START_PLAN; defend a lead and chase a deficit, harder as time runs out."""
    late = time_left < match_seconds * 0.4
    final = time_left < match_seconds * 0.15
    step = 0
    if score_diff > 0:
        step = -2 if (score_diff >= 2 and late) or final else -1 if late or score_diff >= 2 else 0
    elif score_diff < 0:
        step = 2 if (score_diff <= -2 and late) or final else 1 if late or score_diff <= -2 else 0
    i = LADDER.index(START_PLAN) + step
    return LADDER[max(0, min(len(LADDER) - 1, i))]


_COACH_WORDS = [
    (r"\bauto\b|back to (the )?plan|back to normal|back to balanced", None),
    (r"park(ing)? the bus|all back|defend the lead", "PARK_BUS"),
    (r"\ball[- ]?out\b|all[- ]in|everyone attack|1-0-3|1-1-3", "ALL_OUT"),
    (r"counter|2-1-1", "COUNTER"),
    (r"tiki|possession|1-2-1", "TIKI_TAKA"),
    (r"high press|gegen|1-1-2", "PRESS"),
]


def coach_plan(notes: list[str]):
    """The plan the coach asked for most recently, or None (auto). Newest message wins."""
    for note in reversed(notes):
        text = note.lower()
        for pattern, plan in _COACH_WORDS:
            if re.search(pattern, text):
                return plan
    return None


def current_plan(notes: list[str], score_diff: int, time_left: float, match_seconds: float) -> tuple[str, str]:
    """(plan, who chose it): a locked plan wins, then the coach's pick, then the automatic one."""
    if LOCKED_PLAN:
        return LOCKED_PLAN, "locked"
    plan = coach_plan(notes)
    if plan:
        return plan, "coach"
    return auto_plan(score_diff, time_left, match_seconds), "auto"


# --- What one player is told ------------------------------------------------------

def role_of(plan: str, player_id: int) -> str:
    if player_id == 0:
        return "GK"
    return FORMATIONS[PLANS[plan]["formation"]].get(player_id, "PLAYMAKER")


def spot(plan: str, player_id: int, ball: tuple) -> tuple[float, float]:
    role = role_of(plan, player_id)
    base_x, base_y, x_min, x_max = SPOTS[role]
    shift = 0 if role == "GK" else PLANS[plan]["shift"]
    x = max(x_min, min(x_max, base_x + 0.4 * ball[0] + shift))
    y_range = 6 if role == "GK" else min(30, FIELD_Y - 5)
    y = max(-y_range, min(y_range, base_y + 0.3 * ball[1]))
    return round(x, 1), round(y, 1)


def orders(plan: str, player_id: int, we_have_ball: bool) -> list[str]:
    """The plan lines for this tick: team style + this player's role, for the current phase."""
    phase = 0 if we_have_ball else 1
    role = role_of(plan, player_id)
    roles = FORMATIONS[PLANS[plan]["formation"]]
    team = ", ".join(f"P{i} {r}" for i, r in sorted(roles.items()))
    return [
        f"PLAN {PLANS[plan]['name']} ({PLANS[plan]['formation']}) | {team}",
        f"TEAM ({'with ball' if we_have_ball else 'without ball'}): {STYLE[plan][phase]}",
        f"YOUR ROLE {role}: {ROLES[role][phase]}",
    ]
