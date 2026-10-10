"""Synthetic worlds for the graph wiring tests (Task 11). No conftest yet (Task 13 builds it): everything is explicit.

A ``World`` is one MemoryRepo (frozen clock) + one FakeSurePassClient logging into that repo's sink + one FakeRankModel,
bound into ``GraphDeps``. Lenders are seeded with the db/schema.sql criteria spellings (``min_score_floor`` etc.) ONLY.
All PANs, names, mobiles, pincodes and amounts are invented.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Literal, cast

from langchain_core.runnables import RunnableConfig

from eligibility_agent.db.repo import (
    CheckPatch,
    CheckRow,
    ConsentRow,
    LenderRow,
    MemoryRawStore,
    MemoryRepo,
    PanLockRow,
    ReviewRow,
    SurePassCallRow,
)
from eligibility_agent.graph.build import CompiledGraph, GraphDeps, build_graph, build_recompute_graph, new_check_state
from eligibility_agent.graph.prompts import local_prompts
from eligibility_agent.graph.state import CheckState, ConsentInfo, Kind
from eligibility_agent.policy.guards import hash_mobile, hash_pan
from eligibility_agent.surepass.client import CallLogSink, FakeSurePassClient
from eligibility_agent.testing.clock import FrozenClock
from eligibility_agent.testing.fake_llm import FakeRankModel
from eligibility_agent.testing.scenarios import PAN_OK

UTC = timezone.utc
T0: Final[datetime] = datetime(2026, 10, 10, 4, 30, tzinfo=UTC)  # 10:00 IST on 2026-10-10
HMAC_KEY: Final[str] = "test-hmac-key"
MOBILE: Final[str] = "9876543210"
LC_TYPED_NAME: Final[str] = "Typed Learnername"
LC_ID: Final[str] = "lc-graph-1"
SERVED_PIN: Final[str] = "560001"  # the scenarios' PAN zip
OTHER_PIN: Final[str] = "110001"


def criteria(**values: Any) -> dict[str, Any]:
    """The jsonb contract: every field ``{value, status: 'known'}``, schema spellings."""
    return {key: {"value": value, "status": "known"} for key, value in values.items()}


L1_CRITERIA: Final[dict[str, Any]] = criteria(
    min_score_floor=650, ntc_accepted=False, foir_limit=0.55, rate_min_pct=11.0, rate_max_pct=14.0,
    tenure_min_months=12, tenure_max_months=48, loan_min=50000, loan_max=1500000,
)
L2_CRITERIA: Final[dict[str, Any]] = criteria(
    min_score_floor=650, ntc_accepted=True, foir_limit=0.45, rate_min_pct=14.0, rate_max_pct=18.0,
    tenure_min_months=12, tenure_max_months=36, loan_min=25000, loan_max=500000,
)
L3_CRITERIA: Final[dict[str, Any]] = criteria(
    min_score_floor=720, ntc_accepted=False, foir_limit=0.60, rate_min_pct=10.5, rate_max_pct=12.5,
    tenure_min_months=24, tenure_max_months=60, loan_min=100000, loan_max=2500000,
)
L1_NOTES: Final[str] = "Prefers salaried applicants; collateral-free up to the cap."


def seed_lenders(repo: MemoryRepo, *, pincodes: Collection[str] = (SERVED_PIN,)) -> None:
    repo.seed_lender("L1", 1, "Alpha Finance", True, L1_CRITERIA, L1_NOTES, pincodes)
    repo.seed_lender("L2", 1, "Bharat Credit", True, L2_CRITERIA, "", pincodes)
    repo.seed_lender("L3", 1, "Crest Lending", True, L3_CRITERIA, "", pincodes)


class SpyRepo(MemoryRepo):
    """MemoryRepo that records the order of the protocol calls the graph makes (names only), and can be told to fail."""

    def __init__(self, clock: Callable[[], datetime], raw_store: MemoryRawStore | None = None) -> None:
        super().__init__(clock, raw_store)
        self.log: list[str] = []
        self.fail_update_check: Exception | None = None
        self.fail_lenders: Exception | None = None
        self.fail_review_insert: Exception | None = None
        self.fail_create_lock: Exception | None = None

    def insert_consent(self, row: ConsentRow) -> None:
        self.log.append("insert_consent")
        super().insert_consent(row)

    def insert_check(self, row: CheckRow) -> None:
        self.log.append("insert_check")
        super().insert_check(row)

    def update_check(self, check_id: str, patch: CheckPatch) -> None:
        self.log.append("update_check")
        if self.fail_update_check is not None:
            raise self.fail_update_check
        super().update_check(check_id, patch)

    def get_active_locks(self, pan_hash: str) -> list[PanLockRow]:
        self.log.append("get_active_locks")
        return super().get_active_locks(pan_hash)

    def try_acquire_pan_lock(self, pan_hash: str, check_id: str, until: datetime) -> bool:
        self.log.append("try_acquire_pan_lock")
        return super().try_acquire_pan_lock(pan_hash, check_id, until)

    def release_pan_lock(self, pan_hash: str, check_id: str) -> None:
        self.log.append("release_pan_lock")
        super().release_pan_lock(pan_hash, check_id)

    def fetch_raw(self, storage_key: str) -> dict[str, Any]:
        self.log.append("fetch_raw")
        return super().fetch_raw(storage_key)

    def insert_review_item(self, row: ReviewRow) -> None:
        self.log.append("insert_review_item")
        if self.fail_review_insert is not None:
            raise self.fail_review_insert
        super().insert_review_item(row)

    def create_pan_lock(
        self, pan_hash: str, kind: Literal["timeout", "daily_cap"], check_id: str | None, until: datetime
    ) -> None:
        self.log.append("create_pan_lock")
        if self.fail_create_lock is not None:
            raise self.fail_create_lock
        super().create_pan_lock(pan_hash, kind, check_id, until)

    def lenders_for_pincodes(self, pincodes: Collection[str]) -> list[LenderRow]:
        self.log.append("lenders_for_pincodes")
        if self.fail_lenders is not None:
            raise self.fail_lenders
        return super().lenders_for_pincodes(pincodes)


SurePassFactory = Callable[[CallLogSink], FakeSurePassClient]


@dataclass
class World:
    clock: FrozenClock
    repo: SpyRepo
    surepass: FakeSurePassClient
    rank: FakeRankModel
    deps: GraphDeps = field(init=False)

    def __post_init__(self) -> None:
        self.deps = GraphDeps(
            repo=self.repo, surepass=self.surepass, rank_model=self.rank, prompts=local_prompts(), now=self.clock.now
        )

    def with_surepass(self, factory: SurePassFactory) -> World:
        """Same repo, clock and model; a fresh SurePass fake (e.g. one with nothing configured, to prove a cache hit)."""
        return World(clock=self.clock, repo=self.repo, surepass=factory(self.repo), rank=self.rank)

    def with_rank(self, rank: FakeRankModel) -> World:
        return World(clock=self.clock, repo=self.repo, surepass=self.surepass, rank=rank)

    def calls(self, check_id: str) -> list[SurePassCallRow]:
        # Insertion order: under a frozen clock every row shares called_at, so the protocol's (called_at, id) order is
        # uuid order, not call order.
        return self.repo.list_calls(check_id)

    def in_flight_held(self, pan_hash: str) -> bool:
        return any(lock.kind == "in_flight" for lock in self.repo.get_active_locks(pan_hash))

    def row(self, check_id: str) -> CheckRow:
        got = self.repo.get_check(check_id)
        assert got is not None, check_id
        return got

    def reviews(self) -> list[ReviewRow]:
        return self.repo.list_reviews()


def make_world(
    factory: SurePassFactory,
    *,
    rank: FakeRankModel | None = None,
    lenders: bool = True,
    lender_pincodes: Sequence[str] = (SERVED_PIN,),
    start: datetime = T0,
) -> World:
    clock = FrozenClock(start)
    repo = SpyRepo(clock.now)
    if lenders:
        seed_lenders(repo, pincodes=lender_pincodes)
    return World(clock=clock, repo=repo, surepass=factory(repo), rank=rank if rank is not None else FakeRankModel())


def nothing_configured(sink: CallLogSink) -> FakeSurePassClient:
    """Every endpoint unconfigured: ANY SurePass call raises AssertionError ('unexpected SurePass call')."""
    return FakeSurePassClient(sink=sink)


def consent() -> ConsentInfo:
    return ConsentInfo(given=True, wording_version="v2-quick", call_ref="sc-call-1", ip=None)


def pan_hash_of(pan: str = PAN_OK) -> str:
    return hash_pan(pan, key=HMAC_KEY)


def check_state(
    kind: Kind,
    check_id: str,
    *,
    pan: str = PAN_OK,
    income: float | None = 60000.0,
    emi: float | None = None,
    with_consent: bool = True,
    opportunity_id: str | None = None,
    stage: str | None = None,
    refresh: bool = False,
    acknowledge_ntc_warning: bool = False,
    langsmith_run_id: str | None = None,
) -> CheckState:
    return new_check_state(
        kind=kind,
        check_id=check_id,
        lc_id=LC_ID,
        lead_id="lead-graph-1",
        opportunity_id=opportunity_id if opportunity_id is not None else ("opp-graph-1" if kind == "full" else None),
        opportunity_stage=stage if stage is not None else ("offer" if kind == "full" else None),
        pan=pan,
        name=LC_TYPED_NAME,
        mobile=MOBILE,
        pan_hash=hash_pan(pan, key=HMAC_KEY),
        mobile_hash=hash_mobile(MOBILE, key=HMAC_KEY),
        monthly_income=income,
        existing_emi_declared=emi,
        refresh=refresh,
        acknowledge_ntc_warning=acknowledge_ntc_warning,
        consent=consent() if with_consent else None,
        langsmith_run_id=langsmith_run_id,
    )


def invoke(graph: CompiledGraph, state: CheckState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """``graph.invoke`` with the final state typed (langgraph's overload set carries an Unknown ``Command`` arm)."""
    final: object = graph.invoke(state, config)  # pyright: ignore[reportUnknownMemberType]
    return cast("dict[str, Any]", final)


def run(world: World, state: CheckState) -> dict[str, Any]:
    kind = state.get("kind")
    assert kind is not None
    return invoke(build_graph(kind, world.deps), state)


def run_recompute(world: World, state: CheckState) -> dict[str, Any]:
    kind = state.get("kind")
    assert kind is not None
    return invoke(build_recompute_graph(kind, world.deps), state)
