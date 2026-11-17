from engine.scheduler import Request, Scheduler


def test_admits_until_capacity_then_frees() -> None:
    scheduler = Scheduler(max_running=2, max_tokens=100)
    for seq_id in range(5):
        scheduler.admit(Request(seq_id, prompt_len=10, max_new_tokens=5))

    assert len(scheduler.step()) == 0
    assert [sequence.seq_id for sequence in scheduler.running] == [0, 1]
    assert len(scheduler.waiting) == 3

    retired = []
    for _ in range(5):
        retired.extend(scheduler.step())
    assert sorted(sequence.seq_id for sequence in retired) == [0, 1]
    assert 0 not in scheduler.tokens_in_cache and 1 not in scheduler.tokens_in_cache
    assert sorted(sequence.seq_id for sequence in scheduler.running) == [2, 3]


def test_no_sequence_loses_tokens() -> None:
    scheduler = Scheduler(max_running=3, max_tokens=40)
    requests = [Request(i, 4, 3) for i in range(4)]
    for request in requests:
        scheduler.admit(request)

    retired = []
    for _ in range(20):
        retired.extend(scheduler.step())

    assert sorted(sequence.seq_id for sequence in retired) == [0, 1, 2, 3]
    assert all(sequence.generated == 3 for sequence in requests)
    assert scheduler.tokens_in_cache == {}
    assert not scheduler.running and not scheduler.waiting
