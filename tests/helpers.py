from audio_sales_coach.observations import Observation, Turn


def turn(i, spk, s, e, words=None, topics=(), q=False, conf=0.95):
    return Turn(f"turn-{i}", spk, s, e, 1, conf, False, False, False,
                {"syllable_rate_per_s": 4.5, "median_f0_hz": 150.0, "median_db": -25.0}, [], words, True if words else None,
                list(topics), q, None)


def call_fixture():
    turns = [turn(1, "S1", 0.5, 6.0, "Thanks for your time today."),
             turn(2, "S2", 6.6, 14.0, "Our budget is limited so I need the monthly cost and", topics=["price"]),
             turn(3, "S1", 13.2, 19.0, "The annual plan is twelve hundred dollars.", topics=["price"]),
             turn(4, "S2", 22.5, 25.0, "What are the cancellation terms?", topics=["contract_terms"], q=True)]
    obs = [
        Observation("obs-1", "interruption", 13.2, 14.0, ["S1", "S2"],
                    "{S1} started speaking while {S2} was speaking (0.8 s of overlap); {S2} stopped 0.8 s later.",
                    {"overlap_s": 0.8, "interrupter": "S1", "interrupted": "S2", "earlier_speaker_continued_s": 0.8},
                    "simultaneous voice activity", detector="per-channel VAD", score=0.8,
                    alternatives=["line delay"], turn_ids=["turn-2", "turn-3"]),
        Observation("obs-2", "response_gap", 19.0, 22.5, ["S2", "S1"], "3.5 s of silence after {S1} finished before {S2} spoke.",
                    {"gap_s": 3.5}, "gap", detector="VAD", score=3.5, turn_ids=["turn-3", "turn-4"]),
        Observation("obs-3", "question", 22.5, 25.0, ["S2"], "{S2} asked a question (from the transcript).",
                    {"words_reliable": True, "topics": ["contract_terms"]}, "transcript", detector="transcript",
                    turn_ids=["turn-4"]),
        Observation("obs-4", "laughter", 25.5, 26.5, ["S2"], "Laughter detected from {S2} (score 0.81).",
                    {"detector_score": 0.81, "classes": ["Laughter"], "attributed_to": "S2", "attribution": "channel 2"},
                    "yamnet", detector="yamnet", score=0.81, alternatives=["politeness"]),
    ]
    return turns, obs
