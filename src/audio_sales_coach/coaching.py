"""Draft coaching notes from observations (rule-based, deterministic, no language model).

Each note: timestamp + playable reference to the ORIGINAL recording, speaker attribution (or unknown),
exact words when transcribed, the observable cue with its detector/measurement and score or uncertainty,
the interaction worth reviewing, and a constructive alternative question or response.

Notes never state what anyone felt, intended or will do, and never suggest pressure tactics. A language
guard enforces this on every generated note.
"""

from __future__ import annotations

import re
import shlex
import urllib.parse
from dataclasses import asdict, dataclass, field

from .observations import Observation, Turn

ROLE_LABEL = {
    "salesperson": "the salesperson",
    "client_in_this_call": "the client in this call",
    "speaker": "the speaker",
    "unknown": "an unconfirmed speaker",
}

TOPIC_WORDS = {"price": "price/budget", "contract_terms": "contract or cancellation terms", "timeline": "timing",
               "decision_process": "who decides", "competitor": "other providers"}

NOT_INFERRED = ("This note describes what can be heard and measured. It does not say how anyone felt, whether they "
                "were truthful, or whether they will buy.")

BANNED = [
    r"\bnervous", r"\banxious", r"\banxiety", r"\blying\b", r"\bliar\b", r"\blied\b", r"\bdeceptive", r"\bdishonest",
    r"\bcomfortable\b", r"\buncomfortable", r"\bready to (buy|sign|close)", r"\bbuying signal", r"\bexcited",
    r"\bangry", r"\bupset", r"\bfrustrat", r"\bannoyed", r"\bhesitant", r"\bskeptical", r"\bsceptical", r"\bstressed",
    r"\bbored", r"\bemotional", r"\bfelt\b", r"\bfeels\b", r"\bwas (happy|sad|confident|insecure|interested)",
    r"\b(is|was|seems|seemed|sounded) (interested|uninterested|keen|reluctant|worried|scared|afraid)",
    # pressure / manipulation tactics
    r"\bpressure\b", r"\burgency\b", r"\bscarcity", r"\blimited[- ]time", r"\bact now", r"\bfear of missing", r"\bfomo\b",
    r"\bmanipulat", r"\bpush (them|the client|harder)", r"\bclose (them|the deal) (now|immediately)", r"\bdon'?t let them leave",
]
_BANNED_RE = re.compile("|".join(BANNED), re.IGNORECASE)


class GuardError(ValueError):
    pass


def guard_text(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _BANNED_RE.finditer(text or "")})


@dataclass
class Note:
    id: str
    timestamp: str
    start: float
    end: float
    play: dict
    speaker: dict
    words: str | None
    words_status: str
    cue: dict
    interaction: str
    suggestion: str
    alternative_explanations: list[str]
    observation_ids: list[str]
    uncertain: bool
    priority: int
    not_inferred: str = NOT_INFERRED
    council: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def hms(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


def label(spk: str, roles: dict) -> str:
    if spk in ("unknown", None):
        return "an unattributed speaker"
    role = roles.get(spk, "unknown")
    return f"{ROLE_LABEL.get(role, ROLE_LABEL['unknown'])} ({spk})"


def render(text: str, roles: dict) -> str:
    out = re.sub(r"\{(S\d+)\}", lambda m: label(m.group(1), roles), text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)


def play_ref(recording_path: str, start: float, end: float, report_dir: str | None, note_id: str) -> dict:
    a, b = max(0.0, start - 2.0), end + 2.0
    uri = "file://" + urllib.parse.quote(recording_path) + f"#t={a:.1f},{b:.1f}"
    cmd = f"ffplay -nodisp -autoexit -loglevel error -ss {a:.1f} -t {b - a:.1f} {shlex.quote(recording_path)}"
    out = {"recording": "original", "uri": uri, "from_s": round(a, 1), "to_s": round(b, 1), "ffplay": cmd}
    if report_dir:
        out["command"] = f"audio-sales-coach play {shlex.quote(report_dir)} --note {note_id}"
    return out


class _NoteBuilder:
    def __init__(self, turns: list[Turn], roles: dict, recording_path: str, report_dir: str | None, has_transcript: bool):
        self.turns = {t.id: t for t in turns}
        self.order = sorted(turns, key=lambda t: t.start)
        self.roles = roles
        self.rec = recording_path
        self.rdir = report_dir
        self.has_tx = has_transcript
        self.notes: list[Note] = []

    def role(self, spk):
        return self.roles.get(spk, "unknown")

    def next_turn(self, after: float, speaker: str | None = None) -> Turn | None:
        for t in self.order:
            if t.start >= after - 0.05 and (speaker is None or t.speaker == speaker):
                return t
        return None

    def words(self, *turns: Turn | None) -> tuple[str | None, str]:
        if not self.has_tx:
            return None, "no transcript (audio-only analysis)"
        parts = [f"{label(t.speaker, self.roles)}: \"{t.words}\"" for t in turns if t and t.words]
        unreliable = any(t and t.words and t.words_reliable is False for t in turns)
        if not parts:
            return None, "no words transcribed in this span"
        return " | ".join(parts), "transcript marked unreliable here - check by replaying" if unreliable else "transcribed"

    def add(self, o: Observation, interaction: str, suggestion: str, speaker: str | None, priority: int,
            turns: tuple = (), extra_obs: tuple = (), cue_desc: str | None = None, start=None, end=None):
        s = o.start if start is None else start
        e = o.end if end is None else end
        nid = f"note-{len(self.notes) + 1}"
        w, ws = self.words(*turns)
        conf = None
        if speaker and speaker != "unknown":
            confs = [t.attribution_confidence for t in turns if t and t.speaker == speaker]
            conf = round(min(confs), 2) if confs else None
        unc = o.uncertain or not o.attribution_reliable
        n = Note(
            id=nid, timestamp=hms(s), start=round(s, 2), end=round(e, 2),
            play=play_ref(self.rec, s, e, self.rdir, nid),
            speaker={"id": speaker or "unknown", "role": self.role(speaker) if speaker else "unknown",
                     "label": label(speaker, self.roles) if speaker else "unknown",
                     "attribution_confidence": conf, "attribution_reliable": o.attribution_reliable},
            words=w, words_status=ws,
            cue={"type": o.type, "description": render(cue_desc or o.statement, self.roles), "detector": o.detector,
                 "method": o.method, "measurement": o.measurement, "score": o.score,
                 "uncertainty": o.uncertainty or ("none noted" if not unc else "see attribution")},
            interaction=render(interaction, self.roles), suggestion=suggestion,
            alternative_explanations=[render(a, self.roles) for a in o.alternatives],
            observation_ids=[o.id, *[x.id for x in extra_obs]], uncertain=unc, priority=priority)
        problems = guard_text(" ".join([n.interaction, n.suggestion, n.cue["description"], *n.alternative_explanations]))
        if problems:
            raise GuardError(f"{nid} would contain disallowed wording: {problems}")
        self.notes.append(n)


def _topic_clause(t: Turn | None, prefix=" about ") -> str:
    if not t or not t.topics:
        return ""
    return prefix + " and ".join(TOPIC_WORDS[x] for x in t.topics[:2])


def _suggest_for_topic(topic_turn: Turn | None) -> str:
    tp = topic_turn.topics if topic_turn else []
    if "price" in tp:
        return "\"Before I answer - what budget range were you planning for this?\""
    if "contract_terms" in tp:
        return "\"Which terms matter most to you - the length of the commitment or how cancellation works?\""
    if "decision_process" in tp:
        return "\"Who else will be part of the decision, and what will they want to see?\""
    if "timeline" in tp:
        return "\"What timing are you working towards?\""
    return "\"Sorry, I cut in - please go on. What were you about to say?\""


def build_notes(observations: list[Observation], turns: list[Turn], roles: dict, recording_path: str,
                report_dir: str | None, has_transcript: bool, layout: str, max_notes: int = 20) -> list[Note]:
    b = _NoteBuilder(turns, roles, recording_path, report_dir, has_transcript)
    SP, CL = "salesperson", "client_in_this_call"
    roles_known = SP in roles.values()
    obs_by_type: dict[str, list[Observation]] = {}
    for o in observations:
        obs_by_type.setdefault(o.type, []).append(o)

    for o in observations:
        T = [b.turns[i] for i in o.turn_ids if i in b.turns]
        if o.type in ("interruption", "possible_interruption"):
            interrupter, interrupted = o.speakers[0], o.speakers[1]
            prev_t, cur_t = (T + [None, None])[:2]
            if b.role(interrupter) == SP or not roles_known:
                nxt = b.next_turn((cur_t.end if cur_t else o.end), speaker=interrupted)
                follow = ""
                if nxt:
                    if nxt.question_text:
                        follow = f" Their next turn was a question{_topic_clause(nxt)}."
                    elif nxt.topics:
                        follow = f" Their next turn was{_topic_clause(nxt)}."
                    elif nxt.rising_end_st_per_s and nxt.rising_end_st_per_s >= 8:
                        follow = " Their next turn ended with rising pitch (possibly a question)."
                verb = "started speaking while" if o.type == "interruption" else "appears to have started speaking right as"
                inter = (f"{{{interrupter}}} {verb} {{{interrupted}}} was talking{_topic_clause(prev_t, ' about ')}.{follow}")
                sug = ("Let them finish, summarise what you heard in one sentence, then ask: " + _suggest_for_topic(prev_t))
                b.add(o, inter, sug, interrupter, 1, turns=(prev_t, cur_t, nxt))
            elif b.role(interrupter) == CL:
                inter = f"{{{interrupter}}} cut in while {{{interrupted}}} was speaking{_topic_clause(prev_t, ' about ')}."
                sug = "Stop and hand over the floor: \"Go ahead - what would you like to know?\" Then answer that first."
                b.add(o, inter, sug, interrupter, 2, turns=(prev_t, cur_t))
        elif o.type == "response_gap":
            replier, before = o.speakers[0], o.speakers[1]
            prev_t, nxt_t = (T + [None, None])[:2]
            if b.role(before) == SP or (not roles_known and len(set(roles)) > 1):
                inter = (f"After {{{before}}} finished{_topic_clause(prev_t)}, there were {o.measurement['gap_s']:.1f} s of "
                         f"silence before {{{replier}}} replied.")
                sug = ("Silence can be thinking time - let it run a little. If you add something, make it a short open question "
                       "rather than more information, e.g. \"What questions do you have about that?\"")
                b.add(o, inter, sug, replier, 2, turns=(prev_t, nxt_t))
            elif b.role(before) == CL:
                q = " after a question" if prev_t and prev_t.question_text else ""
                inter = f"{{{replier}}} took {o.measurement['gap_s']:.1f} s to respond{q}{_topic_clause(prev_t, ' about ')}."
                sug = ("If you need a moment, say so - \"Good question, let me check that for you\" - so the silence "
                       "is explained.")
                b.add(o, inter, sug, replier, 3, turns=(prev_t, nxt_t))
        elif o.type == "monologue" and (b.role(o.speakers[0]) == SP or not roles_known):
            t = T[0] if T else None
            inter = f"{{{o.speakers[0]}}} spoke for {o.measurement['turn_s']:.0f} s without a turn from the other person{_topic_clause(t)}."
            sug = ("Break long explanations with a check-in about every 30-45 s, e.g. \"Does this match what you need so far?\" "
                   "or \"Which part is most relevant for you?\"")
            b.add(o, inter, sug, o.speakers[0], 2, turns=(t,), start=o.start, end=min(o.end, o.start + 30))
        elif o.type == "talk_share" and (b.role(o.speakers[0]) == SP or not roles_known):
            inter = (f"{{{o.speakers[0]}}} had {o.measurement['talk_share']:.0%} of the speaking time; the window shown is "
                     f"the stretch where they spoke most.")
            sug = ("Plan two or three discovery questions and let the answers run, e.g. \"Can you walk me through how you "
                   "handle this today?\"")
            b.add(o, inter, sug, o.speakers[0], 3)
        elif o.type == "pace_change" and o.measurement["relative_change"] > 0 and (b.role(o.speakers[0]) in (SP, "speaker") or not roles_known):
            t = T[0] if T else None
            inter = (f"{{{o.speakers[0]}}}'s estimated speaking rate rose by {o.measurement['relative_change']:.0%} compared with "
                     f"their usual rate in this recording{_topic_clause(t, ' while discussing ')}.")
            sug = ("Slow down on key numbers and terms and pause after each one so the other person can respond or ask."
                   if t and t.topics else "Slow down and pause after key points so the listener can respond.")
            b.add(o, inter, sug, o.speakers[0], 3, turns=(t,))
        elif o.type in ("laughter",) and o.measurement.get("attributed_to") not in (None,) and \
                (b.role(o.measurement["attributed_to"]) == CL or o.measurement["attributed_to"] in ("unknown", "both")):
            prev = None
            for t in b.order:
                if t.end <= o.start + 0.3 and t.speaker != o.measurement.get("attributed_to"):
                    prev = t
            inter = ("Laughter was detected" + (f" right after {{{prev.speaker}}}'s turn{_topic_clause(prev)}" if prev else "")
                     + ". Replay the 10 s before it to hear what prompted it.")
            sug = ("If it was shared humour you can build on it; if you are unsure how a remark landed, check neutrally: "
                   "\"Did that come across okay?\"")
            spk = o.measurement["attributed_to"] if o.measurement["attributed_to"] not in ("unknown", "both") else None
            b.add(o, inter, sug, spk, 3, turns=(prev,), start=max(0.0, o.start - 4.0))
        elif o.type == "breathing" and o.measurement.get("attributed_to") and b.role(o.measurement["attributed_to"]) == CL:
            nxt = b.next_turn(o.start, speaker=o.measurement["attributed_to"])
            nxt = nxt if nxt and nxt.start - o.end <= 3.0 else None
            inter = ("An audible breath or sigh was detected from {" + o.measurement["attributed_to"] + "}"
                     + (f" before their turn{_topic_clause(nxt)}" if nxt else "") + ". It is a sound, not evidence of a feeling.")
            sug = "Leave room for them to speak; a neutral check-in works either way: \"How does that sound so far?\""
            b.add(o, inter, sug, o.measurement["attributed_to"], 4, turns=(nxt,))
        elif o.type in ("question", "rising_ending") and (b.role(o.speakers[0]) == CL or (not roles_known and layout != "single_speaker")):
            t = T[0] if T else None
            q_start = t.start if t else o.start
            reply = next((x for x in b.order if x.speaker != o.speakers[0] and x.start > q_start + 0.5
                          and x.end > o.end), None)
            gap = (reply.start - o.end) if reply else None
            if o.type == "question":
                inter = f"{{{o.speakers[0]}}} asked a question{_topic_clause(t)}."
            else:
                inter = f"{{{o.speakers[0]}}}'s turn ended with rising pitch{_topic_clause(t)} - possibly a question."
            if reply and gap < 0:
                inter += (f" {{{reply.speaker}}} started replying {-gap:.1f} s before the question ended (overlap) and "
                          f"spoke for {reply.end - reply.start:.0f} s.")
            elif reply:
                inter += f" The reply from {{{reply.speaker}}} started {gap:.1f} s later and lasted {reply.end - reply.start:.0f} s."
                if reply.end - reply.start > 45:
                    inter += " Check whether the question was answered early in that reply."
            else:
                inter += " No reply from the other speaker was detected afterwards."
            sug = ("Answer it directly first, then confirm: \"Does that answer your question?\""
                   + (" For price questions, give the number, then pause." if t and "price" in t.topics else ""))
            b.add(o, inter, sug, o.speakers[0], 1 if o.type == "question" else 3, turns=(t, reply))
        elif o.type == "long_pause" and layout == "single_speaker":
            inter = f"{{{o.speakers[0]}}} paused {o.measurement['pause_s']:.1f} s in the middle of the message."
            sug = "For voice messages, note the main points first so the message flows; lead with the key point and end with one clear question."
            b.add(o, inter, sug, o.speakers[0], 4)
        elif o.type in ("pitch_change", "energy_change") and b.role(o.speakers[0]) == CL:
            t = T[0] if T else None
            inter = (f"{render(o.statement, roles)} This is a measurable change in the voice{_topic_clause(t, ' while discussing ')}; "
                     "replay it to hear the context.")
            sug = "If something shifted here, ask rather than assume: \"What's your take on that?\""
            b.add(o, inter, sug, o.speakers[0], 4, turns=(t,))

    # call-level: voice message length
    if layout == "single_speaker" and b.order:
        dur = b.order[-1].end - b.order[0].start
        if dur > 90:
            o = Observation("obs-length", "message_length", b.order[0].start, b.order[-1].end, [b.order[0].speaker],
                            f"The message is {dur:.0f} s long.", {"duration_s": round(dur, 1)}, "speech span from VAD")
            b.add(o, f"This voice message runs {dur:.0f} s.",
                  "Long voice notes are easy to skip; lead with the key point and end with one clear question.",
                  b.order[0].speaker, 4, start=b.order[0].start, end=b.order[0].start + 15)

    notes = sorted(b.notes, key=lambda n: (n.priority, n.start))[:max_notes]
    notes.sort(key=lambda n: n.start)
    for i, n in enumerate(notes, 1):
        old = n.id
        n.id = f"note-{i}"
        if "command" in n.play:
            n.play["command"] = n.play["command"].replace(f"--note {old}", f"--note {n.id}")
    return notes


def roles_from(speakers: list[str], layout: str, match_summary: dict | None, channel_map: dict, user_roles: dict | None):
    """Decide roles. Order: explicit user assignment > channel map > confident voice-profile match > unknown."""
    roles = {s: "unknown" for s in speakers}
    source = "none"
    if layout == "single_speaker":
        roles = {s: "speaker" for s in speakers}
    if user_roles:
        roles.update(user_roles)
        return roles, "user"
    if layout == "separate_channels" and channel_map:
        for s in speakers:
            ch = str(int(s[1:]) - 1)
            if ch in channel_map:
                roles[s] = channel_map[ch]
        if any(v == "salesperson" for v in roles.values()):
            other = [s for s in speakers if roles[s] == "unknown"]
            for s in other:
                roles[s] = "client_in_this_call"
            return roles, "channel_map"
    if match_summary and match_summary.get("status") == "confident":
        for s, row in match_summary["scores"].items():
            if row["decision"] == "match":
                roles[s] = "salesperson"
        for s in speakers:
            if roles[s] in ("unknown", "speaker") and layout != "single_speaker":
                roles[s] = "client_in_this_call"
        source = "voice_profile"
    return roles, source
