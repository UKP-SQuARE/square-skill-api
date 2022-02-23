from typing import Dict
from fastapi import APIRouter, Depends

from square_skill_api.api import auth
from square_skill_api.models.heartbeat import HeartbeatResult

router = APIRouter()


@router.get(
    "/heartbeat",
    response_model=HeartbeatResult,
    name="heartbeat",
    dependencies=[Depends(auth)],
)
def get_hearbeat() -> HeartbeatResult:
    """Checks if a Skill is still up and running."""
    heartbeat = HeartbeatResult(is_alive=True)
    return heartbeat
