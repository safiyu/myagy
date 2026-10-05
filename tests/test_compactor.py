import asyncio

from myagy.agents.compactor import ContextCurator
from myagy.core.coordinator import CudaCoordinator


def _history(n):
    return [{"role": "user", "target": "9000", "content": f"turn {n_}"} for n_ in range(n)]


def test_background_synthesis_runs_and_releases_slot(session, monkeypatch):
    calls = []
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: calls.append(len(t)) or "SUMMARY"))
    session.history = _history(12)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        await asyncio.sleep(0.2)

    asyncio.run(go())
    assert calls == [6]
    assert CudaCoordinator.state() == "idle"
    assert len(session.history) == 8  # 2 summary entries + 6 recent


def test_subagent_preempts_compaction_and_slot_recovers(session, monkeypatch):
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: "SUMMARY"))
    session.history = _history(12)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        CudaCoordinator.reserve_for_subagent()
        await asyncio.sleep(0.2)
        assert CudaCoordinator.state() == "subagent"
        await CudaCoordinator.release_from_subagent()

    asyncio.run(go())
    assert CudaCoordinator.can_compact()


def test_no_synthesis_when_history_short(session, monkeypatch):
    calls = []
    monkeypatch.setattr(ContextCurator, "synthesize_context_cuda_sync", staticmethod(lambda s, t: calls.append(1)))
    session.history = _history(4)

    async def go():
        ContextCurator.trigger_background_synthesis(session)
        await asyncio.sleep(0.1)

    asyncio.run(go())
    assert calls == [] and CudaCoordinator.state() == "idle"


def test_context_estimate_does_not_summarize(session, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("estimate must not run inference")
    monkeypatch.setattr(session, "summarize_history_sync", boom)
    session.history = _history(30)
    metrics = ContextCurator.get_context_metrics(session, "cloud")
    assert metrics["tokens"] > 0
