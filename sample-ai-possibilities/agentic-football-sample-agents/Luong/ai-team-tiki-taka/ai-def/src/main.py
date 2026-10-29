"""
Tiki-taka "Diamond" team (1-2-1) — Anchor, controls ONLY player 1.
Folder name kept from the balanced sample (ai-def); the role is now ANCHOR.
Uses Strands SDK + Amazon Nova Micro via lib/team_brain.py, with the plan locked to Tiki-taka.
The role prompt below is the same text as PROMPTS.md (Strategy 2, P1).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
import playbook
from fallback import build_fallback, DEF_CONFIG
from team_brain import build_system_prompt, create_team_handler

# This team always plays Tiki-taka: no switching with the score, clock or coach plan words.
# Coach advice is still read every tick.
playbook.LOCKED_PLAN = "TIKI_TAKA"

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 1
POSITION_LABEL = "ANCHOR"

# --- Role prompt (Tiki-taka, P1) ---
ROLE = """You are P1, the ANCHOR of our team. Strategy: "The Diamond", a 1-2-1 tiki-taka possession game.
We pass in triangles until a lane opens; you are the base every triangle starts from.

MISSION: Always offer a safe pass behind the ball, and stop their counter-attacks.

WHERE: The centre, x between -44 and -8, always BEHIND the ball and behind both midfielders.

WITH THE BALL (in priority order):
1. PASS type=GROUND to the OPEN midfielder (P2 or P3) further forward.
2. If one side is crowded, switch play: PASS type=GROUND to the midfielder on the other side.
3. Under pressure: PASS type=GROUND back to the GK (P0) rather than risk losing it.

WHEN A TEAMMATE HAS THE BALL: stay 8-12 behind the ball carrier as the safe back-pass option.

WITHOUT THE BALL (in priority order):
1. Stay between the ball and our goal.
2. MARK TIGHT their most advanced attacker.
3. Only if you are PRESSER and the ball is near our box: PRESS_BALL intensity 0.6.
4. If they break 2-against-1: delay; stay goal-side and do not dive in.

SET PIECES: Their free kick: mark their most dangerous attacker. Our free kick: short GROUND pass to a midfielder.

NEVER: get ahead of the midfielders, try a risky pass in our box, SLIDE_TACKLE in our half.

COACH: Follow messages addressed to ANCHOR, "defenders", or "everyone". The newest message wins.

OUTPUT: Return ONLY a JSON array with ONE command and a reason of at most 8 words, e.g.
[{"commandType":"PASS","parameters":{"target_player_id":3,"type":"GROUND"},"reason":"switch to open right mid"}]"""

SYSTEM_PROMPT = build_system_prompt(MY_PLAYER_ID, ROLE)

# --- Fallback (only used when the model is late or gives a bad answer) ---
fallback_commands = build_fallback(DEF_CONFIG)

# --- Wire it up ---
create_team_handler(
    app, SYSTEM_PROMPT, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=DEF_CONFIG,
)

if __name__ == "__main__":
    app.run()
