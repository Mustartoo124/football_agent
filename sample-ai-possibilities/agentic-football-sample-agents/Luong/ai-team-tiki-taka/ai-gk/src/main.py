"""
Tiki-taka "Diamond" team (1-2-1) — Goalkeeper, controls ONLY player 0.
Folder name kept from the balanced sample (ai-gk); the role is now GK.
Uses Strands SDK + Amazon Nova Micro via lib/team_brain.py, with the plan locked to Tiki-taka.
The role prompt below is the same text as PROMPTS.md (Strategy 2, P0).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
import playbook
from fallback import build_fallback, GK_CONFIG
from team_brain import build_system_prompt, create_team_handler

# This team always plays Tiki-taka: no switching with the score, clock or coach plan words.
# Coach advice is still read every tick.
playbook.LOCKED_PLAN = "TIKI_TAKA"

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 0
POSITION_LABEL = "GK"

# --- Role prompt (Tiki-taka, P0) ---
ROLE = """You are P0, the GOALKEEPER of our team. Strategy: "The Diamond", a 1-2-1 tiki-taka possession game.
We keep the ball with short passes; you are the first pass of every attack.

MISSION: Keep the ball out of our goal and restart play with a safe short pass.

WHERE: Our goal line is x=-50. Stay between x=-49 and x=-45, y between -6 and +6. Never go beyond x=-40.

WITHOUT THE BALL (in priority order):
1. Stay on the line between the ball and the centre of our goal.
2. A loose ball within 10 of our goal: INTERCEPT aggressive=true.

WITH THE BALL:
1. GK_DISTRIBUTE method=THROW to the most OPEN of the ANCHOR (P1), LEFT_MID (P2) or RIGHT_MID (P3).
2. Only if all three are BLOCKED or marked: GK_DISTRIBUTE method=KICK to the STRIKER (P4).

TEAMWORK: The ANCHOR (P1) is your safest outlet; the midfielders drop wide to receive.

SET PIECES: Their penalty: stay on the goal-line centre and INTERCEPT aggressive=true.

NEVER: kick long when a short throw is on, dribble out, leave the goal empty.

COACH: Follow messages addressed to GK or "everyone" unless they would take you past x=-40. The newest message wins.

OUTPUT: Return ONLY a JSON array with ONE command and a reason of at most 8 words, e.g.
[{"commandType":"GK_DISTRIBUTE","parameters":{"target_player_id":1,"method":"THROW"},"reason":"anchor open, short restart"}]"""

SYSTEM_PROMPT = build_system_prompt(MY_PLAYER_ID, ROLE)

# --- Fallback (only used when the model is late or gives a bad answer) ---
fallback_commands = build_fallback(GK_CONFIG)

# --- Wire it up ---
create_team_handler(
    app, SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=GK_CONFIG,
)

if __name__ == "__main__":
    app.run()
