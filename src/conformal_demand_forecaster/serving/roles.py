"""Role checks on top of API-key authentication.

A key authenticates through :func:`require_api_key`; the roles it holds come from the settings
list it was configured under. Endpoints demand exactly one role, so a reader key cannot ingest
and an ingestor key cannot approve a recalibration. Missing roles get a uniform 403.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request

from conformal_demand_forecaster.core.config import Role
from conformal_demand_forecaster.core.errors import AppError
from conformal_demand_forecaster.core.security import Principal, require_api_key

__all__ = ["IngestorPrincipal", "OperatorPrincipal", "ReaderPrincipal", "require_role"]


def require_role(role: Role) -> Callable[[Request, Principal], Principal]:
    def dependency(
        request: Request, principal: Annotated[Principal, Depends(require_api_key)]
    ) -> Principal:
        roles: dict[str, frozenset[Role]] = request.app.state.key_roles
        if role not in roles.get(principal.key_id, frozenset()):
            raise AppError(403, "forbidden", "This API key does not hold the required role.")
        return principal

    return dependency


ReaderPrincipal = Annotated[Principal, Depends(require_role("reader"))]
IngestorPrincipal = Annotated[Principal, Depends(require_role("ingestor"))]
OperatorPrincipal = Annotated[Principal, Depends(require_role("operator"))]
