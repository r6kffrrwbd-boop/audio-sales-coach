import pytest

from audio_sales_coach import coaching
from helpers import call_fixture

ROLES = {"S1": "salesperson", "S2": "client_in_this_call"}
REQUIRED = ("timestamp", "play", "speaker", "words", "words_status", "cue", "interaction", "suggestion",
            "alternative_explanations", "observation_ids", "not_inferred")


def build(roles=ROLES, tx=True):
    turns, obs = call_fixture()
    return coaching.build_notes(obs, turns, roles, "/rec/call.m4a", "/tmp/r", tx, "separate_channels")


def test_every_note_has_required_fields_and_original_reference():
    notes = build()
    assert len(notes) >= 3
    for n in notes:
        d = n.to_dict()
        for k in REQUIRED:
            assert k in d
        assert d["play"]["uri"].startswith("file:///rec/call.m4a#t=")
        assert d["cue"]["detector"] or d["cue"]["method"]
        assert d["speaker"]["role"] in ("salesperson", "client_in_this_call", "unknown")


def test_interruption_note_describes_interaction_and_follow_up_question():
    n = [n for n in build() if n.cue["type"] == "interruption"][0]
    assert "salesperson (S1)" in n.interaction and "client in this call (S2)" in n.interaction
    assert "price/budget" in n.interaction
    assert "question" in n.interaction and "contract or cancellation terms" in n.interaction
    assert "?" in n.suggestion
    assert n.words and "twelve hundred" in n.words


def test_no_transcript_means_words_status_not_invented():
    for n in build(tx=False):
        assert n.words is None
        assert "no transcript" in n.words_status


def test_unknown_roles_are_labelled_unknown():
    notes = build(roles={"S1": "unknown", "S2": "unknown"})
    assert notes
    assert all("unconfirmed speaker" in n.interaction or "unattributed" in n.interaction or "S" in n.interaction for n in notes)
    assert not any("salesperson" in n.interaction for n in notes)


@pytest.mark.parametrize("text", ["The client was nervous.", "she is lying", "he seemed interested",
                                  "create urgency with a limited-time offer", "they are ready to buy",
                                  "the client felt uncomfortable", "pressure them to sign"])
def test_guard_flags_emotion_and_manipulation(text):
    assert coaching.guard_text(text)


def test_generated_notes_pass_the_guard():
    for n in build():
        assert not coaching.guard_text(" ".join([n.interaction, n.suggestion, n.cue["description"],
                                                 *n.alternative_explanations]))


def test_roles_precedence_user_over_profile():
    match = {"status": "confident", "scores": {"S2": {"decision": "match"}, "S1": {"decision": "no_match"}}}
    roles, src = coaching.roles_from(["S1", "S2"], "separate_channels", match, {}, {"S1": "salesperson", "S2": "client_in_this_call"})
    assert src == "user" and roles["S1"] == "salesperson"
    roles, src = coaching.roles_from(["S1", "S2"], "mixed_mono", match, {}, None)
    assert src == "voice_profile" and roles == {"S1": "client_in_this_call", "S2": "salesperson"}
    roles, src = coaching.roles_from(["S1"], "single_speaker", None, {}, None)
    assert roles == {"S1": "speaker"}


def test_question_detection_english_arabic_arabizi():
    from audio_sales_coach.observations import looks_like_question, topics_in
    assert looks_like_question("What are the cancellation terms")
    assert looks_like_question("بس بدي أعرف قديش الـ price تبع الـ annual plan")
    assert looks_like_question("ok bas adesh el si3r")
    assert not looks_like_question("I will need to check with my manager.")
    assert not looks_like_question("تمام، رح إحكي مع المدير وبرجعلك")
    assert "price" in topics_in("قديش السعر؟") and "contract_terms" in topics_in("بدنا نعمل إلغاء للعقد")


def test_overlapping_reply_to_a_question_is_not_reported_as_missing():
    from audio_sales_coach.observations import Observation
    from helpers import turn
    turns = [turn(1, "S2", 10.0, 16.0, "How does the pricing work?", topics=["price"], q=True),
             turn(2, "S1", 15.6, 20.0, "The annual plan is best.", topics=["price"])]
    obs = [Observation("obs-1", "question", 10.0, 16.0, ["S2"], "{S2} asked a question (from the transcript).",
                       {"words_reliable": True, "topics": ["price"]}, "transcript", detector="transcript", turn_ids=["turn-1"])]
    n = coaching.build_notes(obs, turns, ROLES, "/r.wav", None, True, "separate_channels")[0]
    assert "No reply" not in n.interaction and "before the question ended" in n.interaction


def test_noise_marks_sensitive_cues_uncertain():
    from audio_sales_coach.observations import mark_noise
    from helpers import call_fixture
    _, obs = call_fixture()
    assert mark_noise(obs, 20.0) == 0
    n = mark_noise(obs, 7.0)
    gap = [o for o in obs if o.type == "response_gap"][0]
    assert n == 1 and gap.uncertain and "background noise" in gap.uncertainty
    assert not [o for o in obs if o.type == "question"][0].uncertain
