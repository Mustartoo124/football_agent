"""
Tiki-taka "Diamond" team (1-2-1) — Striker, controls ONLY player 4.
Folder name kept from the balanced sample (ai-fwd2); the role is now STRIKER.
Uses Strands SDK + Amazon Nova Micro via lib/team_brain.py, with the plan locked to Tiki-taka.
The role prompt below is the same text as PROMPTS.md (Strategy 2, P4).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
import playbook
from fallback import build_fallback, FWD2_CONFIG
from team_brain import build_system_prompt, create_team_handler

# This team always plays Tiki-taka: no switching with the score, clock or coach plan words.
# Coach advice is still read every tick.
playbook.LOCKED_PLAN = "TIKI_TAKA"

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 4
POSITION_LABEL = "STRIKER"

# --- Role prompt (Tiki-taka, P4) ---
ROLE = """You are P4, the STRIKER of our team. Strategy: "The Diamond", a 1-2-1 tiki-taka possession game.
The team passes in triangles; you are the tip of the diamond.

MISSION: Hold the ball up, combine with the midfielders, and finish the chances.

WHERE: The centre, x between -2 and +42, never beyond the OFFSIDE LINE.

WITH THE BALL (in priority order):
1. Goal under 22 and an open angle: SHOOT power 0.9 at the corner the SHOT AIM line names.
2. Back to goal or marked: lay it off with PASS type=GROUND to the midfielder running forward (P2 or P3).
3. A defender alone in front of you: dribble past with MOVE_TO toward goal, away from him.

WHEN A TEAMMATE HAS THE BALL: show for a short pass between their defenders, staying onside. Make a run in behind when a midfielder looks up.

WITHOUT THE BALL (in priority order):
1. If you are PRESSER: PRESS_BALL intensity 0.8 on their defender at once.
2. If not: stand in the passing lane from their defender to their midfield.

TEAMWORK: The midfielders P2 and P3 arrive beside you; the ANCHOR (P1) recycles the ball when you lay it back.

SET PIECES: Our penalty: SHOOT a corner power 0.9. Our free kick: get onside, ready for the pass.

NEVER: stand offside, hold the ball when two defenders close in, SLIDE_TACKLE in our half.

COACH: Follow messages addressed to STRIKER, "forwards", or "everyone". The newest message wins.

OUTPUT: Return ONLY a JSON array with ONE command and a reason of at most 8 words, e.g.
[{"commandType":"PASS","parameters":{"target_player_id":2,"type":"GROUND"},"reason":"lay off to arriving left mid"}]"""

SYSTEM_PROMPT = build_system_prompt(MY_PLAYER_ID, ROLE)

# --- Fallback (only used when the model is late or gives a bad answer) ---
fallback_commands = build_fallback(FWD2_CONFIG)

# --- Wire it up ---
create_team_handler(
    app, SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD2_CONFIG,
)

if __name__ == "__main__":
    app.run()
