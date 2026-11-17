"""CPU-testable continuous batching core: admit, step, retire, free."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    seq_id: int
    prompt_len: int
    max_new_tokens: int
    generated: int = 0
    finished: bool = False


@dataclass
class Scheduler:
    max_running: int
    max_tokens: int
    waiting: list[Request] = field(default_factory=list)
    running: list[Request] = field(default_factory=list)
    tokens_in_cache: dict[int, int] = field(default_factory=dict)

    def admit(self, request: Request) -> None:
        self.waiting.append(request)

    def step(self) -> list[Request]:
        for sequence in self.running:
            sequence.generated += 1
            if sequence.generated >= sequence.max_new_tokens:
                sequence.finished = True

        retired = [sequence for sequence in self.running if sequence.finished]
        for sequence in retired:
            self.tokens_in_cache.pop(sequence.seq_id, None)
        self.running = [sequence for sequence in self.running if not sequence.finished]

        used = sum(self.tokens_in_cache.values())
        while self.waiting and len(self.running) < self.max_running:
            candidate = self.waiting[0]
            needed = candidate.prompt_len + candidate.max_new_tokens
            if used + needed > self.max_tokens:
                break
            self.waiting.pop(0)
            self.tokens_in_cache[candidate.seq_id] = candidate.prompt_len
            used += needed
            self.running.append(candidate)

        return retired
