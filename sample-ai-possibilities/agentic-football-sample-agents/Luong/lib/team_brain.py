"""Our team's brain — shared by all five players of my-team.

On top of the sample lib this adds:
  - a compact per-tick prompt in a normalised frame (we always attack toward +x)
  - facts worked out in code: who presses, open teammates, our formation spot
  - the coach's latest teamChat suggestions
  - a fresh Strands agent per tick (no ever-growing history) with a hard time limit
  - one log line per decision with latency, source and the model's short reason

The LLM still picks every action. Code only prepares facts and checks the answer;
the sample's rule-based fallback is used only when the model is late or invalid.
"""

import asyncio
import json
import math
import time
from typing import Callable

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel

import playbook
from agent_base import parse_prompt, team_id_from
from fallback import FallbackConfig, build_last_resort
from parsing import FIELD_X, FIELD_Y, GOAL_X, parse_commands
from state import _is_my_team, _player_idx, get_possession_info, holder_is_me

# --- Tunables (edit these, then redeploy) -------------------------------------

MODEL_ID = "us.amazon.nova-micro-v1:0"
LLM_TIMEOUT_S = 0.8      # hard limit for the model call; fallback answers after this
MAX_TOKENS = 120         # one command + a short reason fits easily
TEMPERATURE = 0.2
MATCH_SECONDS = 300      # workshop doc says 5 min; the Strands blog says 2 min
COACH_MEMORY = 4         # how many recent coach messages to keep in the prompt

MAINTAINED = {"PRESS_BALL", "MARK", "INTERCEPT", "FOLLOW_PLAYER"}
GK_MAX_X = -GOAL_X + 10  # hard rule: the GK never moves beyond this line (10 out from our goal)
PRESS_MAX = 0.85         # pressing harder risks fouls -> free kicks / penalties against us
PRESS_MAX_IN_BOX = 0.6   # even softer inside our own box (a foul there is a penalty)
BOX_DEPTH, BOX_HALF_WIDTH = 17, 20   # penalty box size, measured from the goal line
TIRED = 0.20             # below this stamina a player is never picked as PRESSER
LANE_WIDTH = 2.5         # an opponent this close to the pass line blocks it
SPACING = 8              # teammates closer than this are told they are CROWDED
AIM_RIGHT_IS_PLUS_Y = True   # shooter faces +x: is "R" in TR/BR the +y side? (sample code assumes yes;
                             # flip after watching a practice match if shots go the wrong way)


# --- Prompt pieces -------------------------------------------------------------

TEAM_RULES = """## Team rules (all players)
- The pitch is shown from OUR side: we ALWAYS attack toward +x. Centre (0,0). Our goal x=-GX, their goal x=+GX. The pitch runs x -FX..+FX (space behind the goals), y -FY..+FY.
- Follow the PLAN, TEAM and YOUR ROLE lines in every update: formation, style and your job change with the score and the coach.
- Only the player marked PRESSER chases the ball. Everyone else keeps shape: MARK the nearest dangerous opponent or MOVE_TO your spot.
- With the ball: SHOOT if goal is under ~22 away and the angle is open, aiming where the SHOT AIM line says; otherwise PASS to a mate marked OPEN further forward; otherwise dribble with MOVE_TO toward goal, away from defenders.
- Never pass to a mate marked BLOCKED (an opponent stands in the passing lane) or OFFSIDE.
- Keep about 8 apart from teammates: if told CROWDED, move away toward your spot.
- Commit: when the LAST ORDER line says the situation is unchanged, keep doing it. Do not zig-zag between targets.
- When we lose the ball, the PRESSER uses PRESS_BALL; a free ball near you → INTERCEPT.
- The ball never goes out: no throw-ins, no corners. FOULS give the other team a FREE KICK, or a PENALTY in our box. Pressing too hard is a foul: never PRESS_BALL above 0.85, never SLIDE_TACKLE in our half.
- OFFSIDE gives them a free kick: when a mate is about to pass forward, stay behind the OFFSIDE LINE. Never pass to a mate marked OFFSIDE.
- A SET PIECE line (kick-off, free kick, penalty) overrides the plan until play restarts.
- Stamina is limited: every action costs some, sprinting costs more. Sprint only when the STAMINA line says sprint OK.
- Use the full command set; do not just PASS.
- COACH suggestions come from our coach, who sees the whole match. Follow them when they apply to you, unless they break your hard rule. The newest suggestion wins.

## Commands (commandType: parameters)
MOVE_TO: target_x, target_y, sprint (bool)
PASS: target_player_id (teammate 0-4), type GROUND|AERIAL|THROUGH — only with the ball
SHOOT: aim_location TL|TR|BL|BR|CENTER, power 0.0-1.0 — only with the ball
GK_DISTRIBUTE: target_player_id, method THROW|KICK — GK with the ball only
PRESS_BALL: intensity 0.0-1.0
INTERCEPT: aggressive (bool)
MARK: target_player_id (opponent 0-4), tightness LOOSE|TIGHT
SLIDE_TACKLE: target_player_id (opponent), sprint (bool), distance (float)
FOLLOW_PLAYER: target_player_id, target_team HOME|AWAY, distance

## Response
Return ONLY a JSON array with exactly ONE command and a reason of at most 8 words. No other text.
Example: [{"commandType":"PASS","parameters":{"target_player_id":3,"type":"THROUGH"},"reason":"striker open, through ball"}]"""
TEAM_RULES = TEAM_RULES.replace("GX", str(GOAL_X)).replace("FX", str(FIELD_X)).replace("FY", str(FIELD_Y))


def build_system_prompt(player_id: int, role_text: str) -> str:
    """Role-specific text first, then the rules every player shares."""
    return (
        f"You are player {player_id} ({'goalkeeper' if player_id == 0 else 'outfield player'}) in a fast 5v5 AI football match. "
        f"You control ONLY yourself and decide one action every 2 seconds.\n\n"
        f"## Your role\n{role_text.strip()}\n\n{TEAM_RULES}"
    )


# --- Normalised frame ------------------------------------------------------------

def _flip(team_id: int) -> float:
    """HOME already attacks +x. AWAY is mirrored through the centre spot."""
    return 1.0 if team_id == 0 else -1.0


def _norm(pos: dict, team_id: int) -> tuple[float, float]:
    f = _flip(team_id)
    return f * float(pos.get("x", 0) or 0) + 0.0, f * float(pos.get("y", 0) or 0) + 0.0  # + 0.0 avoids "-0"


def _d(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --- Coach suggestions -------------------------------------------------------------

_coach_log: list[str] = []
_last_game_time = -1.0


def _chat_text(entry) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("message", "text", "content", "instruction", "msg"):
            if isinstance(entry.get(key), str):
                return entry[key].strip()
    return ""


def coach_notes(game_state: dict) -> list[str]:
    """Recent coach messages, oldest first, remembered across ticks of one match."""
    global _last_game_time
    game_time = float(game_state.get("gameTime", 0) or 0)
    if game_time < _last_game_time:   # clock went back → new match, forget old coaching
        _coach_log.clear()
    _last_game_time = game_time

    for entry in game_state.get("teamChat") or []:
        text = _chat_text(entry)[:200]
        if text and (not _coach_log or _coach_log[-1] != text) and text not in _coach_log[-COACH_MEMORY:]:
            _coach_log.append(text)
    del _coach_log[:-COACH_MEMORY]
    return list(_coach_log)


# --- Per-tick facts -----------------------------------------------------------------

def stamina_fraction(raw) -> float:
    """0.0-1.0 whatever the server sends: observed as a fraction, documented as 0-100."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, value / 100.0 if value > 1.5 else value))


def stamina_line(stamina: float, time_left: float) -> str:
    """Spend stamina evenly over the match; late on, it is fine to empty the tank."""
    reserve = 0.6 * time_left / MATCH_SECONDS
    if stamina < TIRED:
        return f"STAMINA {stamina:.0%} LOW: walk, no sprint, press only if the ball is right next to you."
    if stamina - reserve > 0.15:
        return f"STAMINA {stamina:.0%}: sprint OK when it matters."
    return f"STAMINA {stamina:.0%}: conserve, no sprint unless the ball is within 5."


def offside_line(opp_xy: dict, ball_xy: tuple) -> float:
    """Second-last opponent (GK included) or the ball, whichever is further forward; never in our half."""
    xs = sorted((xy[0] for xy in opp_xy.values()), reverse=True)
    second_last = xs[1] if len(xs) >= 2 else GOAL_X
    return max(second_last, ball_xy[0], 0.0)


def in_our_box(xy: tuple) -> bool:
    return xy[0] < -GOAL_X + BOX_DEPTH and abs(xy[1]) < BOX_HALF_WIDTH


def lane_blocked(a: tuple, b: tuple, opps, width: float = LANE_WIDTH) -> bool:
    """True when an opponent stands within `width` of the straight pass from a to b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    length2 = abx * abx + aby * aby
    if length2 < 1:
        return False
    for o in opps:
        t = ((o[0] - a[0]) * abx + (o[1] - a[1]) * aby) / length2
        if 0.1 < t < 0.95 and _d(o, (a[0] + t * abx, a[1] + t * aby)) < width:
            return True
    return False


def shot_aim(me_xy: tuple, keeper_xy: tuple | None) -> tuple[str, str]:
    """(side word, corners) for the side of the goal away from the keeper.

    Keeper clearly off-centre → the other side; keeper central → the far post from the shooter.
    """
    keeper_y = keeper_xy[1] if keeper_xy else 0.0
    side_y = -keeper_y if abs(keeper_y) > 0.8 else (-me_xy[1] if abs(me_xy[1]) > 1 else 1.0)
    right = (side_y > 0) == AIM_RIGHT_IS_PLUS_Y
    return ("right", "TR/BR") if right else ("left", "TL/BL")


# Per-player memory of the last decision, for the commitment rule. One agent process
# controls one player, but keying by id keeps tests and the local demo honest.
_last_tick: dict[int, dict] = {}   # written by build_tick_prompt: situation this tick
_memory: dict[int, dict] = {}      # written by remember(): situation + what we decided


def _situation(ball_status: str, presser_is_me: bool) -> str:
    owner = "free" if ball_status == "FREE" else "us" if ball_status.startswith(("YOU", "WE")) else "them"
    return f"{owner}/{'presser' if presser_is_me else 'shape'}"


def summarize_command(cmd: dict, team_id: int) -> str:
    """Short text of a command in our frame, e.g. 'MOVE_TO (12,-5)' or 'MARK opp 3'."""
    kind = cmd.get("commandType", "?")
    p = cmd.get("parameters") or {}
    if kind == "MOVE_TO":
        f = _flip(team_id)
        try:
            return f"MOVE_TO ({f * float(p.get('target_x', 0)):.0f},{f * float(p.get('target_y', 0)):.0f})"
        except (TypeError, ValueError):
            return "MOVE_TO"
    if kind in ("MARK", "SLIDE_TACKLE"):
        return f"{kind} opp {p.get('target_player_id', '?')}"
    if kind in ("PASS", "GK_DISTRIBUTE"):
        return f"{kind} to P{p.get('target_player_id', '?')}"
    return kind


def remember(my_id: int, cmd: dict, team_id: int) -> None:
    """Store what this player just decided, with the situation it decided in."""
    tick = _last_tick.get(my_id)
    if tick:
        _memory[my_id] = {**tick, "order": summarize_command(cmd, team_id)}


def commitment_line(my_id: int, game_time: float, situation: str, ball_xy: tuple) -> str | None:
    last = _memory.get(my_id)
    if not last or not (0 < game_time - last["time"] <= 6):
        return None
    ago = game_time - last["time"]
    if last["situation"] == situation and _d(last["ball"], ball_xy) < 8:
        return f"LAST ORDER ({ago:.0f}s ago): {last['order']}. Situation unchanged: keep it unless something clearly better appears."
    return f"LAST ORDER ({ago:.0f}s ago): {last['order']}. Situation changed: decide again."


def recommended_aim(my_id: int) -> str:
    """Corner to use when the model shoots without a valid aim."""
    return (_last_tick.get(my_id) or {}).get("aim", "BR")


def _ours(mode_team, team_id: int):
    """True/False when the set piece is ours/theirs, None when the server does not say."""
    if mode_team is None:
        return None
    if isinstance(mode_team, str):
        return mode_team.strip().lower() == ("home" if team_id == 0 else "away")
    return mode_team == team_id


def set_piece_line(play_mode, mode_team, team_id: int):
    mode = str(play_mode or "").upper()
    ours = _ours(mode_team, team_id)
    if "PENALTY" in mode:
        if ours is True:
            return "SET PIECE our penalty: the PRESSER SHOOTs a corner (TL/TR/BL/BR) power 0.9; others wait outside the box."
        if ours is False:
            return "SET PIECE their penalty: GK stays on the goal line centre and INTERCEPT aggressive=true; others wait outside our box ready to clear."
        return "SET PIECE penalty: if ours, the PRESSER shoots a corner; if theirs, GK guards the centre and others wait outside the box."
    if "FREE_KICK" in mode:
        if ours is True:
            return "SET PIECE our free kick: the PRESSER takes it: SHOOT if goal < 25, else PASS to an OPEN onside mate; others move into space."
        if ours is False:
            return "SET PIECE their free kick: get back between the ball and our goal and MARK their attackers; do not press the taker."
        return "SET PIECE free kick: if ours, the PRESSER shoots (goal < 25) or passes; if theirs, get back and MARK."
    if "KICK_OFF" in mode:
        return "SET PIECE kick-off: everyone to your spot fast; if it is our kick-off the PRESSER passes to a mate."
    return None


def build_tick_prompt(game_state: dict, team_id: int, my_id: int) -> str:
    """Compact, normalised description of this tick for one player."""
    players = game_state.get("players", [])
    ball = game_state.get("ball", {})
    ball_xy = _norm(ball.get("position", {}), team_id)
    mine = {_player_idx(p): p for p in players if _is_my_team(p, team_id)}
    theirs = {_player_idx(p): p for p in players if not _is_my_team(p, team_id)}
    my_xy = {i: _norm(p.get("position", {}), team_id) for i, p in mine.items()}
    opp_xy = {i: _norm(p.get("position", {}), team_id) for i, p in theirs.items()}

    score = game_state.get("score", {})
    us, them = (score.get("home", 0), score.get("away", 0)) if team_id == 0 else (score.get("away", 0), score.get("home", 0))
    game_time = float(game_state.get("gameTime", 0) or 0)
    time_left = max(0.0, MATCH_SECONDS - game_time)
    me = mine.get(my_id, {})
    me_xy = my_xy.get(my_id, (0.0, 0.0))
    stamina = stamina_fraction(me.get("stamina", 1.0))
    has_ball = holder_is_me(ball, players, team_id, my_id)
    holder_idx, _, we_have = get_possession_info(ball, players, team_id)

    if ball.get("isFree") is True or holder_idx is None:
        ball_status = "FREE"
    elif we_have:
        ball_status = "YOU have it" if has_ball else f"WE have it (P{holder_idx})"
    else:
        ball_status = f"THEY have it (opp {holder_idx})"

    # Presser = nearest outfield teammate to the ball (tie → lower id). The GK only
    # presses when the ball is deep in our box.
    # Tired players are skipped unless everyone is tired.
    fresh = {i for i, p in mine.items() if stamina_fraction(p.get("stamina", 1.0)) >= TIRED}
    outfield = sorted((round(_d(xy, ball_xy), 1), i) for i, xy in my_xy.items() if i != 0 and i in fresh) \
        or sorted((round(_d(xy, ball_xy), 1), i) for i, xy in my_xy.items() if i != 0)
    presser = outfield[0][1] if outfield else my_id
    gk_closer = outfield and 0 in my_xy and _d(my_xy[0], ball_xy) < outfield[0][0]
    if gk_closer and ball_xy[0] < -GOAL_X + 8 and abs(ball_xy[1]) < 12:
        presser = 0
    my_team_code, opp_team_code = ("HOME", "AWAY") if team_id == 0 else ("AWAY", "HOME")

    notes = coach_notes(game_state)
    plan, chosen_by = playbook.current_plan(notes, us - them, time_left, MATCH_SECONDS)
    spot = playbook.spot(plan, my_id, ball_xy)
    line_x = offside_line(opp_xy, ball_xy)
    if my_id != 0 and spot[0] > line_x - 1:
        spot = (round(line_x - 1, 1), spot[1])   # never send a player to an offside spot
    goal = (GOAL_X, 0)
    lines = [
        f"Time left {time_left:.0f}s | Score us {us} - {them} them | Mode {game_state.get('playMode', '?')} | plan by {chosen_by}",
        *playbook.orders(plan, my_id, bool(we_have)),
        f"Ball ({ball_xy[0]:.0f},{ball_xy[1]:.0f}): {ball_status}",
        f"YOU P{my_id}: pos ({me_xy[0]:.0f},{me_xy[1]:.0f}) "
        f"distBall {_d(me_xy, ball_xy):.0f} distGoal {_d(me_xy, goal):.0f} spot ({spot[0]:.0f},{spot[1]:.0f})",
        stamina_line(stamina, time_left),
        "PRESSER: YOU" if presser == my_id else f"PRESSER: P{presser} (not you, keep shape)",
    ]
    if my_id != 0:
        offside = " (you are OFFSIDE: step back now)" if me_xy[0] > line_x and not has_ball else ""
        lines.append(f"OFFSIDE LINE x={line_x:.0f}{offside}")
    if in_our_box(me_xy):
        lines.append("You are in OUR BOX: a foul here is a penalty. No SLIDE_TACKLE, press gently.")
    set_piece = set_piece_line(game_state.get("playMode"), game_state.get("modeTeamId"), team_id)
    if set_piece:
        lines.insert(1, set_piece)

    situation = _situation(ball_status, presser == my_id)
    commit = commitment_line(my_id, game_time, situation, ball_xy)
    if commit:
        lines.append(commit)
    keeper = opp_xy.get(0)
    side, corners = shot_aim(me_xy, keeper)
    if has_ball and _d(me_xy, goal) < 35:
        ky = f"y={keeper[1]:.0f}" if keeper else "unknown"
        lines.append(f"SHOT AIM: keeper at {ky}, aim {side} corners ({corners}), away from the keeper.")
    _last_tick[my_id] = {"time": game_time, "situation": situation, "ball": ball_xy,
                         "aim": corners.split("/")[1]}
    if my_id != 0 and not has_ball and presser != my_id:
        crowd = min(((round(_d(me_xy, xy)), i) for i, xy in my_xy.items() if i not in (my_id, 0)), default=None)
        if crowd and crowd[0] < SPACING:
            lines.append(f"CROWDED: P{crowd[1]} is {crowd[0]} away, move apart toward your spot.")

    def nearest_opp(xy):
        return min((_d(xy, o) for o in opp_xy.values()), default=99.0)

    mates = []
    for i, xy in sorted(my_xy.items()):
        if i == my_id:
            continue
        if i != 0 and xy[0] > line_x:
            tag = "OFFSIDE"
        elif lane_blocked(ball_xy, xy, opp_xy.values()):
            tag = "BLOCKED"
        else:
            tag = "OPEN" if nearest_opp(xy) > 7 else "marked"
        mates.append(f"P{i}({xy[0]:.0f},{xy[1]:.0f}) {tag} goal{_d(xy, goal):.0f}")
    lines.append("Mates: " + "; ".join(mates))

    opps = []
    for i, xy in sorted(opp_xy.items()):
        opps.append(f"{i}({xy[0]:.0f},{xy[1]:.0f}) me{_d(xy, me_xy):.0f}")
    lines.append(f"Opponents ({opp_team_code}): " + "; ".join(opps))
    lines.append(f"Our team code {my_team_code}.")

    if notes:
        lines.append("COACH (oldest to newest): " + " | ".join(f'"{n}"' for n in notes))
    return "\n".join(lines)


# --- Answer checking -----------------------------------------------------------------

def finalise_command(cmd: dict, team_id: int, my_id: int, me_xy: tuple | None = None) -> tuple[dict, str]:
    """Strip the reason, enforce hard rules, convert back to real coordinates.

    me_xy is our player's position in the normalised frame; it drives the foul guards.
    """
    reason = str(cmd.pop("reason", "") or "")[:80]
    params = cmd.get("parameters")
    if not isinstance(params, dict):
        params = {}
        cmd["parameters"] = params
    if me_xy is not None and cmd.get("commandType") == "SLIDE_TACKLE" and me_xy[0] < 0:
        # A slide tackle in our half risks a free kick or penalty: press instead.
        cmd["commandType"], params = "PRESS_BALL", {"intensity": PRESS_MAX_IN_BOX}
        cmd["parameters"] = params
        reason = f"[guard: no slide in our half] {reason}"
    if cmd.get("commandType") == "PRESS_BALL":
        cap = PRESS_MAX_IN_BOX if me_xy is not None and in_our_box(me_xy) else PRESS_MAX
        try:
            params["intensity"] = min(float(params.get("intensity", 0.7)), cap)
        except (TypeError, ValueError):
            params["intensity"] = cap
    if cmd.get("commandType") == "MOVE_TO":
        x = float(params.get("target_x", 0) or 0)
        y = float(params.get("target_y", 0) or 0)
        if my_id == 0:
            x = min(x, GK_MAX_X)
        f = _flip(team_id)
        params["target_x"], params["target_y"] = round(f * x, 2), round(f * y, 2)
        params["sprint"] = bool(params.get("sprint", False))
    cmd.setdefault("duration", 3 if cmd.get("commandType") in MAINTAINED else 0)
    return cmd, reason


# --- Invoke handler ------------------------------------------------------------------

def create_team_handler(
    app,
    system_prompt: str,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    model_id: str = MODEL_ID,
):
    """Register the @app.entrypoint for one player. Same 3 layers as the sample:
    LLM → rule-based fallback → last-resort command."""
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)
    model = BedrockModel(
        model_id=model_id,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        streaming=False,
        boto_client_config=Config(connect_timeout=1, read_timeout=2, retries={"max_attempts": 1}),
    )

    async def ask_model(prompt: str) -> str:
        # A fresh agent each tick: no history to grow, and no clash if a slow
        # call from the previous tick is still running.
        agent = Agent(model=model, system_prompt=system_prompt, callback_handler=None)
        result = await asyncio.wait_for(agent.invoke_async(prompt), timeout=LLM_TIMEOUT_S)
        return str(result)

    @app.entrypoint
    async def invoke(payload, context):
        started = time.perf_counter()
        prompt_data = parse_prompt(payload)
        team_id = team_id_from(prompt_data, log, position_label)
        game_state = prompt_data.get("gameState", {}) or {}
        my_players = prompt_data.get("myPlayers") or [my_player_id]
        pid = my_players[0]
        source, reason = "llm", ""
        me = next((p for p in game_state.get("players", [])
                   if _is_my_team(p, team_id) and _player_idx(p) == pid), {})
        me_xy = _norm(me.get("position", {}), team_id) if me else None
        try:
            prompt = build_tick_prompt(game_state, team_id, pid)
            try:
                text = await ask_model(prompt)
                commands = parse_commands(text, team_id, pid)
            except asyncio.TimeoutError:
                source, commands, text = "timeout", [], ""
            if commands:
                cmd, reason = finalise_command(commands[0], team_id, pid, me_xy)
                if cmd.get("commandType") == "SHOOT" and \
                        cmd["parameters"].get("aim_location") not in ("TL", "TR", "BL", "BR"):
                    cmd["parameters"]["aim_location"] = recommended_aim(pid)
                commands = [cmd]
            else:
                if source == "llm":
                    source = "parse_fail"
                    log.warn(f"{position_label} unparseable reply: {text[:200]}")
                commands = fallback_fn(game_state, team_id, pid)[:1]
        except Exception as e:
            log.error(f"{position_label} error: {e}")
            source = "error"
            try:
                commands = fallback_fn(game_state, team_id, pid)[:1]
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                commands = []
        if not commands:
            cmd = dict(last_resort)
            cmd["teamId"], cmd["playerId"] = team_id, pid
            commands = [cmd]
        remember(pid, commands[0], team_id)

        log.info(json.dumps({
            "decision": position_label,
            "tick": game_state.get("tick"),
            "ms": round((time.perf_counter() - started) * 1000),
            "source": source,
            "cmd": commands[0].get("commandType"),
            "params": commands[0].get("parameters"),
            "reason": reason,
            "stamina": round(stamina_fraction(me.get("stamina", 1.0)), 3) if me else None,
            "mode": game_state.get("playMode"),
        }))
        yield json.dumps(commands)

    return invoke
