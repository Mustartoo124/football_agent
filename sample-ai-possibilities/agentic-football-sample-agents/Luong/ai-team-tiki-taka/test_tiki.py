"""Checks that this team is the Tiki-taka "Diamond" — no AWS needed.

    python ai-team-strands-balanced/test_tiki.py
"""

import asyncio
import copy
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

from test_helpers import mock_agentcore, GAME_STATE, _FakeApp
mock_agentcore()

import playbook
import team_brain

PLAYERS = [("ai-gk", 0, "GK"), ("ai-def", 1, "ANCHOR"), ("ai-mid", 2, "LEFT_MID"),
           ("ai-fwd1", 3, "RIGHT_MID"), ("ai-fwd2", 4, "STRIKER")]


def load(folder):
    spec = importlib.util.spec_from_file_location(f"tiki_{folder}", os.path.join(HERE, folder, "src", "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAgent:
    def __init__(self, **kwargs):
        self.system_prompt = kwargs.get("system_prompt", "")

    async def invoke_async(self, prompt):
        return '[{"commandType":"PASS","parameters":{"target_player_id":1,"type":"GROUND"},"reason":"short pass"}]'


def main():
    print("=== Tiki-taka team checks ===")
    # Losing late and the coach asks for another plan: the team must still play Tiki-taka.
    state = copy.deepcopy(GAME_STATE)
    state["gameTime"], state["score"] = 280.0, {"home": 0, "away": 2}
    state["teamChat"] = ["High press now", "All out attack"]
    for folder, pid, label in PLAYERS:
        m = load(folder)
        assert m.MY_PLAYER_ID == pid and m.POSITION_LABEL == label, folder
        assert f"You are P{pid}," in m.SYSTEM_PROMPT and '"The Diamond"' in m.SYSTEM_PROMPT, folder
        assert "## Commands" in m.SYSTEM_PROMPT, "shared rules and command list included"
        team_brain._coach_log.clear(); team_brain._last_game_time = -1
        prompt = team_brain.build_tick_prompt(state, 0, pid)
        assert "PLAN Tiki-taka (1-2-1)" in prompt and "plan by locked" in prompt, prompt
        assert f"YOUR ROLE {label}:" in prompt, (folder, prompt)
        assert 'COACH (oldest to newest): "High press now" | "All out attack"' in prompt, "advice still reaches players"

        team_brain.Agent = FakeAgent
        app = _FakeApp()
        cfg_name = {0: "GK_CONFIG", 1: "DEF_CONFIG", 2: "MID_CONFIG", 3: "FWD1_CONFIG", 4: "FWD2_CONFIG"}[pid]
        handler = team_brain.create_team_handler(app, m.SYSTEM_PROMPT, pid, label, m.fallback_commands, getattr(m, cfg_name))
        payload = {"prompt": json.dumps({"gameState": GAME_STATE, "teamId": 1, "myPlayers": [pid]})}

        async def collect():
            return [chunk async for chunk in handler(payload, None)]

        out = json.loads(asyncio.run(collect())[0])
        assert len(out) == 1 and out[0]["playerId"] == pid and out[0]["teamId"] == 1, out
        print(f"  [PASS] {folder:8} P{pid} {label:9} system prompt ~{len(m.SYSTEM_PROMPT) // 4} tokens, plan locked to Tiki-taka")
    playbook.LOCKED_PLAN = None
    print("All Tiki-taka team checks passed")


if __name__ == "__main__":
    main()
