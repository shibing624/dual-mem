"""Public exception types raised by the write, digest and review paths."""


class DecisionContextError(RuntimeError):
    """The decision provider rejected a request or returned an unusable decision."""


class DecisionPlanError(DecisionContextError):
    """A complete plan violates execution dependencies; the batch is re-decided."""


class CoverageError(DecisionContextError):
    """Candidates do not faithfully cover their source; extraction is regenerated."""


class StaleDecisionError(DecisionContextError):
    """Targets or evidence changed concurrently; retrying the same call is safe."""


class IdempotencyConflictError(ValueError):
    """A request_id was reused with different input."""


__all__ = [
    "CoverageError",
    "DecisionContextError",
    "DecisionPlanError",
    "IdempotencyConflictError",
    "StaleDecisionError",
]
