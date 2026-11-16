"""Human-readable renderings of report.json (Markdown and a self-contained HTML timeline)."""

from __future__ import annotations

import html
import json
import urllib.parse

from .coaching import hms


def render_markdown(r: dict) -> str:
    rec = r["recording"]
    L = [f"# Coaching report {r['report_id']}", "",
         f"- Recording: `{rec['path']}` ({rec['codec']}, {rec['channels']} ch, {rec['duration_s']:.1f} s)",
         f"- Layout: **{r['layout']['result']}**; speakers: " +
         ", ".join(f"{s['id']} = {s['role']} ({s['talk_share']:.0%} of talk)" for s in r["speakers"]),
         f"- Events detector: {r['methods']['events'].get('detector', 'unavailable')}; transcription: "
         f"{r['methods']['transcription'].get('status')}",
         f"- Council review: **{r['council'].get('status')}**" +
         (f" - {r['council'].get('reason')}" if r["council"].get("reason") else
          f" (verdict {r['council'].get('verdict_status')}, providers used: {r['council'].get('providers_used') or 'none - local checks only'})"),
         ""]
    if r["role_assignment"]["status"] == "needs_user":
        L += ["## Roles need your confirmation", "",
              "Listen to these excerpts from the original recording, then run:", "",
              f"    {r['role_assignment']['how_to_assign']}", ""]
        for e in r["role_assignment"]["excerpts"]:
            L.append(f"- {e['speaker']} at {hms(e['start'])}: `{e['ffplay']}`")
        L.append("")
    if r["warnings"]:
        L += ["## Reliability warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    L += ["## Coaching notes", ""]
    if not r["coaching_notes"]:
        L += ["No notes were drafted for this recording.", ""]
    for n in r["coaching_notes"]:
        c = n["cue"]
        L += [f"### {n['id']} - {n['timestamp']} - {n['speaker']['label']}" + (" (uncertain)" if n["uncertain"] else ""), "",
              f"- **Replay (original recording):** `{n['play']['ffplay']}`",
              f"- **Cue:** {c['description']}",
              f"- **Detector / measurement:** {c['detector'] or 'measurement'} - {c['method']}"
              + (f"; score {c['score']}" if c["score"] is not None else ""),
              f"- **Uncertainty:** {c['uncertainty']}",
              f"- **Words:** {n['words'] or '(' + n['words_status'] + ')'}",
              f"- **Interaction to review:** {n['interaction']}",
              f"- **Try instead:** {n['suggestion']}",
              f"- **Other explanations:** {'; '.join(n['alternative_explanations']) or '-'}",
              f"- **Council:** {n['council'].get('status')}"
              + (f" - {'; '.join(n['council'].get('issues', []))}" if n["council"].get("issues") else ""),
              f"- Evidence: {', '.join(n['observation_ids'])}", ""]
    for c in r["council"].get("provider_comments") or []:
        L += [f"## Council comment from {c['provider']} (unverified)", "", "> " + " ".join(c["text"].split()), "", f"_{c['status']}_", ""]
    L += ["## All observations", "", "| id | time | type | statement | uncertain |", "|---|---|---|---|---|"]
    from .coaching import render
    roles = {s["id"]: s["role"] for s in r["speakers"]}
    for o in r["observations"]:
        L.append(f"| {o['id']} | {hms(o['start'])} | {o['type']} | {render(o['statement'], roles).replace('|', '/')} | "
                 f"{'yes' if o['uncertain'] else 'no'} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in r["limitations"]] + [""]
    L += ["_Every note describes audible, measurable cues. None states how anyone felt, whether they were truthful, "
          "or whether they will buy._", ""]
    return "\n".join(L)


_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e0d8;--sp:#2f6db5;--cl:#b5622f;--ev:#7a4fb5;--warn:#9a6a00;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--sp:#6fa4e8;--cl:#e89a6f;--ev:#b598e8;--warn:#e0b44a;--card:#1f1f1d}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
h1{font-size:20px}h2{font-size:17px;margin-top:28px}.muted{color:var(--muted)}
.lane{display:flex;align-items:center;gap:8px;margin:4px 0}.lane .name{width:150px;font-size:13px;color:var(--muted);flex:none}
.track{position:relative;height:18px;flex:1;background:var(--card);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.seg{position:absolute;top:2px;height:14px;border-radius:2px;cursor:pointer;opacity:.85}
.mark{position:absolute;top:0;width:3px;height:18px;background:var(--ev);cursor:pointer}
.note{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
.note h3{margin:0 0 6px;font-size:15px}.note dl{display:grid;grid-template-columns:160px 1fr;gap:4px 10px;margin:0}
.note dt{color:var(--muted)}.note dd{margin:0}.unc{color:var(--warn);font-size:12px;margin-left:6px}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:2px 10px;cursor:pointer}
audio{width:100%;margin:8px 0}@media (max-width:640px){.note dl{grid-template-columns:1fr}.lane .name{width:90px}}
"""


def render_html(r: dict) -> str:
    e = html.escape
    dur = max(r["recording"]["duration_s"], 1e-6)
    src = "file://" + urllib.parse.quote(r["recording"]["path"])
    roles = {s["id"]: s["role"] for s in r["speakers"]}
    color = {"salesperson": "var(--sp)", "client_in_this_call": "var(--cl)"}

    def lane(name, turns, marks):
        segs = "".join(
            f'<div class="seg" title="{e(t["speaker"])} {t["start"]:.1f}-{t["end"]:.1f}s" data-t="{t["start"]}" '
            f'style="left:{100 * t["start"] / dur:.3f}%;width:{max(0.2, 100 * (t["end"] - t["start"]) / dur):.3f}%;'
            f'background:{color.get(roles.get(t["speaker"]), "var(--muted)")}"></div>' for t in turns)
        mk = "".join(f'<div class="mark" title="{e(m[1])}" data-t="{m[0]}" style="left:{100 * m[0] / dur:.3f}%"></div>'
                     for m in marks)
        return f'<div class="lane"><div class="name">{e(name)}</div><div class="track">{segs}{mk}</div></div>'

    lanes = []
    views = r.get("views", {})
    turns_by = {t["id"]: t for t in r["turns"]}
    notes_by = {n["id"]: n for n in r["coaching_notes"]}
    for key, v in views.items():
        if not isinstance(v, dict) or "turn_ids" not in v:
            continue
        marks = [(notes_by[i]["start"], f'{i}: {notes_by[i]["cue"]["type"]}') for i in v["note_ids"] if i in notes_by]
        lanes.append(lane(key.replace("_", " "), [turns_by[i] for i in v["turn_ids"] if i in turns_by], marks))
    cards = []
    for n in r["coaching_notes"]:
        c = n["cue"]
        rows = [("Replay", f'<button data-t="{n["play"]["from_s"]}" data-end="{n["play"]["to_s"]}">play {e(hms(n["play"]["from_s"]))}'
                           f'-{e(hms(n["play"]["to_s"]))}</button> <span class="muted">original recording</span>'),
                ("Speaker", e(n["speaker"]["label"]) + ("" if n["speaker"]["attribution_reliable"] else ' <span class="unc">attribution uncertain</span>')),
                ("Cue", e(c["description"])),
                ("Detector / method", e(f'{c["detector"] or "measurement"}: {c["method"]}') + (f' (score {c["score"]})' if c["score"] is not None else "")),
                ("Uncertainty", e(c["uncertainty"])),
                ("Words", e(n["words"]) if n["words"] else f'<span class="muted">{e(n["words_status"])}</span>'),
                ("Interaction", e(n["interaction"])),
                ("Try instead", e(n["suggestion"])),
                ("Other explanations", e("; ".join(n["alternative_explanations"]) or "-")),
                ("Council", e(n["council"].get("status", "")) + (": " + e("; ".join(n["council"].get("issues", []))) if n["council"].get("issues") else ""))]
        dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
        cards.append(f'<div class="note" id="{e(n["id"])}"><h3>{e(n["id"])} &middot; {e(n["timestamp"])}'
                     f'{" <span class=unc>uncertain</span>" if n["uncertain"] else ""}</h3><dl>{dl}</dl></div>')
    warn = "".join(f"<li>{e(w)}</li>" for w in r["warnings"])
    roles_block = ""
    if r["role_assignment"]["status"] == "needs_user":
        ex = "".join(f'<li>{e(x["speaker"])}: <button data-t="{x["start"]}" data-end="{x["end"]}">play {e(hms(x["start"]))}</button></li>'
                     for x in r["role_assignment"]["excerpts"])
        roles_block = (f'<h2>Who is the salesperson?</h2><p>Listen, then run <code>{e(r["role_assignment"]["how_to_assign"])}</code></p><ul>{ex}</ul>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coaching report</title><style>{_CSS}</style></head><body>
<h1>Coaching report <span class="muted">{e(r['report_id'])}</span></h1>
<p class="muted">{e(r['recording']['path'])} &middot; {r['recording']['duration_s']:.0f} s &middot; layout {e(r['layout']['result'])}
&middot; council {e(r['council'].get('status', ''))}</p>
<audio id="a" controls preload="metadata" src="{e(src)}"></audio>
<p class="muted">Timeline (one timebase for every view). Click a bar or mark to play the original recording there.</p>
{''.join(lanes)}
{roles_block}
{'<h2>Reliability warnings</h2><ul>' + warn + '</ul>' if warn else ''}
<h2>Coaching notes</h2>{''.join(cards) or '<p>No notes.</p>'}
<p class="muted">Notes describe audible, measurable cues only. They do not say how anyone felt, whether they were truthful, or whether they will buy.</p>
<script>
const a=document.getElementById('a');let stopAt=null;
document.querySelectorAll('[data-t]').forEach(el=>el.addEventListener('click',()=>{{a.currentTime=parseFloat(el.dataset.t);stopAt=el.dataset.end?parseFloat(el.dataset.end):null;a.play();}}));
a.addEventListener('timeupdate',()=>{{if(stopAt!==null&&a.currentTime>=stopAt){{a.pause();stopAt=null;}}}});
</script>
<script type="application/json" id="report-meta">{e(json.dumps({"report_id": r["report_id"], "schema_version": r["schema_version"]}))}</script>
</body></html>"""
