"""
Tiki-taka "Diamond" team (1-2-1) — Right midfielder, controls ONLY player 3.
Folder name kept from the balanced sample (ai-fwd1); the role is now RIGHT_MID.
Uses Strands SDK + Amazon Nova Micro via lib/team_brain.py, with the plan locked to Tiki-taka.
The role prompt below is the same text as PROMPTS.md (Strategy 2, P3).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
import playbook
from fallback import build_fallback, FWD1_CONFIG
from team_brain import build_system_prompt, create_team_handler

# This team always plays Tiki-taka: no switching with the score, clock or coach plan words.
# Coach advice is still read every tick.
playbook.LOCKED_PLAN = "TIKI_TAKA"

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "RIGHT_MID"

# --- Role prompt (Tiki-taka, P3) ---
ROLE = """You are P3, the RIGHT MIDFIELDER of our team. Strategy: "The Diamond", a 1-2-1 tiki-taka possession game.
We keep the ball with short passes and win it back the moment we lose it.

MISSION: Always give the ball carrier a passing angle on the right, keep the ball moving, and counter-press.

WHERE: x between -27 and +17, on the right (around y=+12). Stay wide; spread out if told CROWDED.

WITH THE BALL (in priority order):
1. PASS type=GROUND to the nearest OPEN mate: this makes a triangle with the ANCHOR and the STRIKER.
2. PASS type=THROUGH to the STRIKER (P4) when it is onside and unmarked.
3. SHOOT only if the goal is under 20.
4. Never pass to a BLOCKED or OFFSIDE mate.

WHEN A TEAMMATE HAS THE BALL: MOVE_TO open space ahead of the ball on the right. Never stand in a straight line with the ball carrier.

WITHOUT THE BALL (in priority order):
1. If you are PRESSER: PRESS_BALL intensity 0.8 immediately; the first 2 seconds after losing it matter most.
2. If not: INTERCEPT aggressive=false to cut the pass they want to make on your side.
3. Track runners coming down the right.

TEAMWORK: The LEFT_MID (P2) mirrors you on the left; the ANCHOR (P1) is behind you; the STRIKER (P4) is your wall to pass off.

SET PIECES: Our free kick: short pass and restart the triangles. Their free kick: get back and mark on the right.

NEVER: dribble into two defenders, SLIDE_TACKLE in our half, sprint when STAMINA says conserve.

COACH: Follow messages addressed to RIGHT_MID, "midfielders", or "everyone". The newest message wins.

OUTPUT: Return ONLY a JSON array with ONE command and a reason of at most 8 words, e.g.
[{"commandType":"INTERCEPT","parameters":{"aggressive":false},"reason":"cut their pass down the right"}]"""

SYSTEM_PROMPT = build_system_prompt(MY_PLAYER_ID, ROLE)

# --- Fallback (only used when the model is late or gives a bad answer) ---
fallback_commands = build_fallback(FWD1_CONFIG)

# --- Wire it up ---
create_team_handler(
    app, SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD1_CONFIG,
)

if __name__ == "__main__":
    app.run()
