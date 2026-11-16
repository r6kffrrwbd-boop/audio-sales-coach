"""Observable cues from turns, measurements and detected events.

Observations describe what can be heard and measured (who spoke when, overlaps, silences, pace, pitch and
level relative to the same speaker, detector-scored laughter/breathing, words when transcribed). They never
describe feelings or intentions. Statements use {S1}/{S2} placeholders that are rendered with the current
role labels, so a role correction re-renders every note without re-analysing audio.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .events import Event
from .features import METHODS, ChannelFeatures
from .speakers import Segment, SpeakerResult

TOPICS = {
    "price": [r"\bprice", r"\bpricing", r"\bcost", r"\bbudget", r"\bexpensive", r"\bcheap", r"\bdiscount", r"\bfee",
              r"\bpay\b", r"\bpayment", r"\$\s?\d", r"\bdollars?\b", r"سعر", r"أسعار", r"مصاري", r"كلفة", r"تكلفة", r"ميزانية",
              r"غالي", r"حسم", r"خصم", r"دفع", r"\bsi3r", r"\bmasari", r"\bghali", r"\bkelfe"],
    "contract_terms": [r"\bcancel", r"\bcontract", r"\bterms?\b", r"\bcommit", r"\brenew", r"\brefund", r"\btrial",
                       r"\bnotice period", r"إلغاء", r"الغاء", r"عقد", r"شروط", r"تجديد", r"استرجاع", r"\bfaskh"],
    "timeline": [r"\bnext week", r"\bdeadline", r"\bwhen can", r"\bstart date", r"\bquarter", r"\btimeline",
                 r"إيمتى", r"امتى", r"متى", r"الأسبوع الجاي", r"\bemta\b"],
    "decision_process": [r"\bmanager", r"\bboss", r"\bpartner", r"\bfinance", r"\bapprov", r"\bsign[- ]?off",
                         r"\bdecide", r"\bdecision", r"مدير", r"شريك", r"موافقة", r"قرار"],
    "competitor": [r"\bcompetitor", r"\banother (company|provider|vendor)", r"\bother (company|provider|vendor)",
                   r"\balternative", r"شركة تانية", r"شركة ثانية", r"منافس"],
}
QUESTION_START = re.compile(
    r"^\s*(what|how|when|why|where|who|which|can|could|would|will|do|does|did|is|are|should|may|"
    r"شو|كيف|كيف|ليش|ليه|قديش|قدي|وين|إيمتى|امتى|هل|ما|ماذا|متى|أين|كم|مين|بتقدر|فيك|"
    r"shu|kif|keef|lesh|leh|adesh|addesh|wein|wen|emta|min)\b", re.IGNORECASE)


def topics_in(text: str | None) -> list[str]:
    if not text:
        return []
    t = text.lower()
    return [k for k, pats in TOPICS.items() if any(re.search(p, t) for p in pats)]


# Levantine/MSA interrogatives often come mid-sentence ("بس بدي أعرف قديش ..."), so they count anywhere.
QUESTION_ANYWHERE = re.compile(r"(?:^|\s)(شو|كيف|ليش|ليه|قديش|قدي|وين|إيمتى|امتى|متى|أين|كم|مين|هل|ماذا|"
                               r"shu|kif|keef|lesh|adesh|addesh|wein|emta)(?=\s|$|[؟?،,.])", re.IGNORECASE)


def looks_like_question(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip()
    return (t.endswith("?") or t.endswith("؟") or bool(QUESTION_START.match(t))
            or bool(QUESTION_ANYWHERE.search(t)))


@dataclass
class Turn:
    id: str
    speaker: str
    start: float
    end: float
    segment_count: int
    attribution_confidence: float
    uncertain: bool
    overlap: bool
    possible_overlap: bool
    stats: dict
    pauses: list[list[float]]
    words: str | None = None
    words_reliable: bool | None = None
    topics: list[str] = field(default_factory=list)
    question_text: bool = False
    rising_end_st_per_s: float | None = None


@dataclass
class Observation:
    id: str
    type: str
    start: float
    end: float
    speakers: list[str]
    statement: str
    measurement: dict
    method: str
    detector: str | None = None
    score: float | None = None
    uncertain: bool = False
    uncertainty: str = ""
    alternatives: list[str] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    attribution_reliable: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = round(self.start, 2), round(self.end, 2)
        return d


def build_turns(sr_res: SpeakerResult, feats: dict[str, ChannelFeatures], transcript=None, merge_gap: float = 1.2) -> list[Turn]:
    segs = sorted([s for s in sr_res.segments if s.speaker != "unknown"], key=lambda s: s.start)
    groups: list[list[Segment]] = []
    for s in segs:
        if groups and groups[-1][-1].speaker == s.speaker and s.start - groups[-1][-1].end <= merge_gap:
            groups[-1].append(s)
        elif groups and groups[-1][-1].speaker == s.speaker and not _other_between(segs, groups[-1][-1], s):
            groups[-1].append(s)
        else:
            groups.append([s])
    turns = []
    for i, g in enumerate(groups, 1):
        spk = g[0].speaker
        f = feats.get(spk) or next(iter(feats.values()))
        start, end = g[0].start, max(x.end for x in g)
        pauses = [[round(a.end, 2), round(b.start, 2)] for a, b in zip(g, g[1:]) if b.start - a.end > 0]
        conf = float(np.average([x.confidence for x in g], weights=[x.duration for x in g]))
        t = Turn(f"turn-{i}", spk, start, end, len(g), round(conf, 3), any(x.uncertain for x in g),
                 any(x.overlap for x in g), any(x.possible_overlap for x in g), f.segment_stats(start, end), pauses)
        slope = f.final_pitch_slope(start, end)
        t.rising_end_st_per_s = round(slope, 1) if slope is not None else None
        tr = transcript.for_speaker(spk) if hasattr(transcript, "for_speaker") else transcript
        if tr is not None:
            t.words = tr.words_between(start, end) or None
            t.words_reliable = tr.reliable_between(start, end)
            t.topics = topics_in(t.words)
            t.question_text = looks_like_question(t.words) or bool(t.words and re.search(r"[?؟]", t.words))
        turns.append(t)
    return turns


def _other_between(segs, a: Segment, b: Segment) -> bool:
    return any(s.speaker != a.speaker and s.start < b.start and s.end > a.end for s in segs)


class _Ids:
    def __init__(self):
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"obs-{self.n}"


def derive(sr_res: SpeakerResult, turns: list[Turn], events: list[Event], feats: dict[str, ChannelFeatures],
           cfg: dict, duration: float) -> list[Observation]:
    a = cfg["analysis"]
    nid = _Ids()
    obs: list[Observation] = []
    two = len(sr_res.speakers) == 2
    sep = sr_res.layout == "separate_channels"

    # ---- talk share (two speakers)
    if two:
        talk = {s: sum(t.end - t.start for t in turns if t.speaker == s) for s in sr_res.speakers}
        tot = sum(talk.values()) or 1.0
        for s, v in talk.items():
            share = v / tot
            if share >= a["talk_share_flag"]:
                win = _busiest_window(turns, s, 120.0, duration)
                obs.append(Observation(nid(), "talk_share", win[0], win[1], [s],
                                       f"{{{s}}} spoke {share:.0%} of the speaking time in this recording "
                                       f"(busiest 2-minute stretch shown).",
                                       {"talk_share": round(share, 3), "talk_s": round(v, 1), "total_talk_s": round(tot, 1)},
                                       "sum of turn durations per speaker from voice activity / speaker turns",
                                       uncertain=not sep, uncertainty="" if sep else
                                       "speaker attribution on a mixed recording can shift talk time between speakers",
                                       alternatives=["the call may have been a planned presentation or demo"],
                                       attribution_reliable=sep))

    # ---- interruptions / overlaps
    by_start = sorted(turns, key=lambda t: t.start)
    if sep:
        for i, t in enumerate(by_start):
            prev = [p for p in by_start[:i] if p.speaker != t.speaker and p.end > t.start + a["interruption_min_overlap_s"]
                    and t.start - p.start >= 1.0]
            if not prev:
                continue
            p = prev[-1]
            ov = min(p.end, t.end) - t.start
            # the earlier speaker's last segment overlapping this onset
            p_seg_end = max((s.end for s in sr_res.segments if s.speaker == p.speaker and s.start < t.start < s.end + 0.05),
                            default=p.end)
            yielded = p_seg_end - t.start <= 2.0
            short = (t.end - t.start) < 1.0
            if short:
                kind, stmt = "overlap_short", (f"{{{t.speaker}}} made a short sound or remark ({t.end - t.start:.1f} s) while "
                                               f"{{{p.speaker}}} was speaking.")
                alts = ["a backchannel such as 'mm-hm' or 'yes'", "agreement or encouragement to continue"]
            elif yielded:
                kind, stmt = "interruption", (f"{{{t.speaker}}} started speaking while {{{p.speaker}}} was speaking "
                                              f"({ov:.1f} s of overlap); {{{p.speaker}}} stopped {max(0.0, p_seg_end - t.start):.1f} s later.")
                alts = [f"{{{p.speaker}}} may have been finishing anyway", "line delay can make turn-taking look like overlap"]
            else:
                kind, stmt = "overlap", (f"{{{t.speaker}}} and {{{p.speaker}}} spoke at the same time for {ov:.1f} s; "
                                         f"{{{p.speaker}}} kept speaking.")
                alts = ["both speakers continuing in parallel", "a backchannel that ran long"]
            obs.append(Observation(nid(), kind, t.start, min(p.end, t.end), [t.speaker, p.speaker], stmt,
                                   {"overlap_s": round(ov, 2), "interrupter": t.speaker, "interrupted": p.speaker,
                                    "earlier_speaker_continued_s": round(max(0.0, p_seg_end - t.start), 2)},
                                   "simultaneous voice activity on the two call channels",
                                   detector="per-channel VAD", score=round(ov, 2), alternatives=alts,
                                   turn_ids=[p.id, t.id]))
    else:
        for p, t in zip(by_start, by_start[1:]):
            if p.speaker != t.speaker and t.start - p.end < 0.08 and p.end - p.start >= 2.0:
                obs.append(Observation(nid(), "possible_interruption", t.start, min(t.end, t.start + 2), [t.speaker, p.speaker],
                                       f"{{{t.speaker}}} started speaking with no audible pause after {{{p.speaker}}} "
                                       f"({max(0.0, t.start - p.end):.2f} s gap).",
                                       {"gap_s": round(t.start - p.end, 3)},
                                       "speaker change without a silence on a single mixed channel",
                                       detector="VAD + speaker clustering", uncertain=True,
                                       uncertainty="on a mixed recording, overlap and the exact change point cannot be "
                                                   "confirmed; replay before acting on this",
                                       alternatives=["a natural quick hand-over", "the first speaker had finished"],
                                       turn_ids=[p.id, t.id], attribution_reliable=False))

    # ---- response gaps and long pauses
    for p, t in zip(by_start, by_start[1:]):
        gap = t.start - p.end
        if p.speaker != t.speaker and gap >= a["long_response_gap_s"]:
            obs.append(Observation(nid(), "response_gap", p.end, t.start, [t.speaker, p.speaker],
                                   f"{gap:.1f} s of silence after {{{p.speaker}}} finished before {{{t.speaker}}} spoke.",
                                   {"gap_s": round(gap, 2)}, METHODS["response_gap"], detector="VAD", score=round(gap, 2),
                                   alternatives=["thinking or reading", "checking something off-call", "a connection delay",
                                                 "muted microphone"], turn_ids=[p.id, t.id],
                                   attribution_reliable=sep or (not p.uncertain and not t.uncertain)))
    for t in turns:
        for s, e in t.pauses:
            if e - s >= 2.0:
                obs.append(Observation(nid(), "long_pause", s, e, [t.speaker],
                                       f"{{{t.speaker}}} paused {e - s:.1f} s mid-turn.", {"pause_s": round(e - s, 2)},
                                       METHODS["pauses"], detector="VAD", score=round(e - s, 2),
                                       alternatives=["looking something up", "letting a point sink in", "a dropout in the line"],
                                       turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))

    # ---- monologues
    if two:
        for t in turns:
            if t.end - t.start >= a["monologue_s"]:
                obs.append(Observation(nid(), "monologue", t.start, t.end, [t.speaker],
                                       f"{{{t.speaker}}} spoke for {t.end - t.start:.0f} s without the other person taking a turn.",
                                       {"turn_s": round(t.end - t.start, 1)}, "turn duration from speaker turns",
                                       alternatives=["the other person asked for a full explanation", "a requested demo"],
                                       turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))

    # ---- pace / pitch / level changes relative to the same speaker
    for spk in sr_res.speakers:
        ts = [t for t in turns if t.speaker == spk and t.end - t.start >= 4.0]
        rates = [t.stats["syllable_rate_per_s"] for t in ts if t.stats.get("syllable_rate_per_s")]
        f0s = [t.stats["median_f0_hz"] for t in ts if t.stats.get("median_f0_hz")]
        dbs = [t.stats["median_db"] for t in ts if t.stats.get("median_db") is not None]
        med_rate = float(np.median(rates)) if len(rates) >= 3 else None
        med_f0 = float(np.median(f0s)) if len(f0s) >= 3 else None
        med_db = float(np.median(dbs)) if len(dbs) >= 3 else None
        for t in ts:
            r = t.stats.get("syllable_rate_per_s")
            if med_rate and r and abs(r - med_rate) / med_rate >= a["pace_change_ratio"]:
                d = "faster" if r > med_rate else "slower"
                obs.append(Observation(nid(), "pace_change", t.start, t.end, [spk],
                                       f"{{{spk}}} spoke {d} than usual in this turn: about {r:.1f} vs a median of "
                                       f"{med_rate:.1f} estimated syllables/s.",
                                       {"syllable_rate": r, "speaker_median_rate": round(med_rate, 2),
                                        "relative_change": round((r - med_rate) / med_rate, 2)},
                                       METHODS["pace"], detector="intensity-peak syllable estimate",
                                       score=round(abs(r - med_rate) / med_rate, 2), uncertain=True,
                                       uncertainty="syllable estimates are approximate (+/- ~15%) and depend on audio quality",
                                       alternatives=["reading a list or numbers", "summarising familiar material"],
                                       turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))
            f = t.stats.get("median_f0_hz")
            if med_f0 and f:
                st = 12 * np.log2(f / med_f0)
                if abs(st) >= a["pitch_change_semitones"]:
                    obs.append(Observation(nid(), "pitch_change", t.start, t.end, [spk],
                                           f"{{{spk}}}'s voice was {abs(st):.1f} semitones {'higher' if st > 0 else 'lower'} "
                                           f"than their median in this recording.",
                                           {"semitones_vs_speaker_median": round(float(st), 2), "median_f0_hz": f,
                                            "speaker_median_f0_hz": round(med_f0, 1)},
                                           METHODS["pitch_change"], detector="autocorrelation pitch tracker",
                                           score=round(abs(float(st)), 2), uncertain=True,
                                           uncertainty="pitch shifts have many causes; this is a measurement, not an emotion",
                                           alternatives=["emphasis", "reading aloud", "speaking to someone else in the room",
                                                         "pitch-tracking error on compressed audio"],
                                           turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))
            d_ = t.stats.get("median_db")
            if med_db is not None and d_ is not None and abs(d_ - med_db) >= a["energy_change_db"]:
                obs.append(Observation(nid(), "energy_change", t.start, t.end, [spk],
                                       f"{{{spk}}} was {abs(d_ - med_db):.0f} dB {'louder' if d_ > med_db else 'quieter'} "
                                       f"than their median level in this recording.",
                                       {"db_vs_speaker_median": round(d_ - med_db, 1)}, METHODS["energy_change"],
                                       detector="RMS level", score=round(abs(d_ - med_db), 1), uncertain=True,
                                       uncertainty="level depends on distance to the microphone and line gain",
                                       alternatives=["moved closer to / away from the phone", "speakerphone", "emphasis"],
                                       turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))

    # ---- detector events (laughter, breathing, cough)
    for ev in events:
        spk, reliable, how = attribute_event(ev, sr_res, turns)
        label = {"laughter": "Laughter", "breathing": "Audible breathing/sigh", "cough_throat": "Cough or throat clearing"}[ev.type]
        who = f"{{{spk}}}" if spk not in ("unknown", "both") else ("both speakers" if spk == "both" else "an unattributed source")
        alts = {"laughter": ["shared humour", "politeness", "a reaction to something off-call", "detector false alarm on speech"],
                "breathing": ["microphone close to the mouth", "movement or walking", "detector false alarm on noise"],
                "cough_throat": ["an ordinary cough", "detector false alarm"]}[ev.type]
        thr_unc = ev.type == "breathing" or ev.score < 0.5
        obs.append(Observation(nid(), ev.type, ev.start, ev.end, [] if spk in ("unknown", "both") else [spk],
                               f"{label} detected from {who} (score {ev.score:.2f}).",
                               {"detector_score": ev.score, "classes": ev.classes, "attributed_to": spk, "attribution": how},
                               f"{ev.detector} audio-event classifier on the original audio, 0.96 s windows, 0.48 s hop"
                               if ev.detector == "yamnet" else f"{ev.detector} audio tagging, 1 s windows, 0.5 s hop",
                               detector=ev.detector, score=ev.score, uncertain=thr_unc or not reliable,
                               uncertainty=("breathing is a weak detector class; replay to confirm" if ev.type == "breathing"
                                            else "low detector score" if ev.score < 0.5 else "")
                               + ("" if reliable else ("; " if thr_unc else "") + "speaker attribution uncertain"),
                               alternatives=alts, attribution_reliable=reliable))

    # ---- questions (words) and rising endings (acoustic)
    for t in turns:
        if t.question_text:
            obs.append(Observation(nid(), "question", t.start, t.end, [t.speaker],
                                   f"{{{t.speaker}}} asked a question (from the transcript).",
                                   {"words_reliable": bool(t.words_reliable), "topics": t.topics},
                                   "question mark or interrogative word (English/Arabic/Arabizi list) in the transcript",
                                   detector="transcript", uncertain=not t.words_reliable,
                                   uncertainty="" if t.words_reliable else "transcript confidence is low for this turn",
                                   turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))
        elif t.rising_end_st_per_s is not None and t.rising_end_st_per_s >= 8.0 and t.end - t.start >= 1.0:
            obs.append(Observation(nid(), "rising_ending", max(t.start, t.end - 1.0), t.end, [t.speaker],
                                   f"{{{t.speaker}}}'s turn ended with rising pitch (+{t.rising_end_st_per_s:.0f} semitones/s) - "
                                   f"possibly a question.",
                                   {"final_slope_st_per_s": t.rising_end_st_per_s}, "slope of F0 over the last 0.5 s of the turn",
                                   detector="pitch tracker", score=t.rising_end_st_per_s, uncertain=True,
                                   uncertainty="rising endings are also used for lists and statements, especially in Levantine Arabic",
                                   turn_ids=[t.id], attribution_reliable=sep or not t.uncertain))
    obs.sort(key=lambda o: (o.start, o.id))
    for i, o in enumerate(obs, 1):  # stable, time-ordered ids
        o.id = f"obs-{i}"
    return obs


def attribute_event(ev: Event, sr_res: SpeakerResult, turns: list[Turn]) -> tuple[str, bool, str]:
    if sr_res.layout == "separate_channels" and ev.channel is not None:
        spk = f"S{ev.channel + 1}"
        return spk, True, f"heard on channel {ev.channel + 1}"
    if len(sr_res.speakers) == 1:
        return sr_res.speakers[0], True, "single-speaker recording"
    cover = {}
    for t in turns:
        ov = min(t.end, ev.end) - max(t.start, ev.start)
        if ov > 0:
            cover[t.speaker] = cover.get(t.speaker, 0) + ov
    if not cover:
        return "unknown", False, "outside any speaker turn on a mixed recording"
    best = max(cover, key=cover.get)
    if len(cover) > 1 and sorted(cover.values())[-2] > 0.3 * cover[best]:
        return "unknown", False, "overlaps both speakers' turns on a mixed recording"
    turn_uncertain = any(t.uncertain for t in turns if t.speaker == best and t.start < ev.end and t.end > ev.start)
    return best, not turn_uncertain, "inside this speaker's turn on a mixed recording"


def merge_channel_events(evs: list[Event]) -> list[Event]:
    """Mark the same event type at the same time on both channels (e.g. shared laughter or crosstalk)."""
    return sorted(evs, key=lambda e: e.start)


def _busiest_window(turns, spk, win, duration):
    if duration <= win:
        return 0.0, duration
    best, best_t = -1.0, 0.0
    for t0 in np.arange(0, max(duration - win, 0) + 1, 10.0):
        s = sum(max(0.0, min(t.end, t0 + win) - max(t.start, t0)) for t in turns if t.speaker == spk)
        if s > best:
            best, best_t = s, float(t0)
    return best_t, best_t + win
