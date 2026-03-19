import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum

import asyncpg
from scenario_server.entities import ScenarioGrade, SubmissionResult
from scenario_server.grading.grading import grade_responses

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


class DeferredGradingStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"


@dataclass
class DeferredGradingResult:
    result: SubmissionResult | None
    status: DeferredGradingStatus
    error: str | None


@dataclass
class DeferredGradingState:
    grading_id: str
    status: DeferredGradingStatus


class DeferredGradingStorage(ABC):
    @abstractmethod
    async def close(self) -> None:
        """Close any open connections"""

    @abstractmethod
    async def fetch(self, grading_id: str) -> DeferredGradingResult:
        """Given a grading id, return a result

        Args:
            grading_id (str) : grading id

        Returns:
            DefferedGradingResult

        Raises:
            KeyError: if grading_id is not present
        """

    @abstractmethod
    async def prune(self, grading_id: str | list[str]) -> int:
        """Delete data related to grading ids

        If grading id is not present, does nothing

        Args:
            grading_id (str|list[str]) : one or more grading ids to remove

        Returns:
            int: count of grading ids deleted
        """

    @abstractmethod
    async def state(self, grading_id: str) -> DeferredGradingState:
        """Return state of deferred grading task

        Args:
            grading_id (str) : grading id

        Returns:
            DeferredGradingState

        Raises:
            KeyError: if grading_id is not present
        """

    @abstractmethod
    async def store(self, grading_id: str, data: DeferredGradingResult) -> None:
        """Create or update a deferred grading result for the given grading id

        Args:
            grading_id (str) : grading id
            data (DeferedGradingResult) : grading result, state, and error
        """

    @abstractmethod
    async def valid(self, grading_id: str) -> bool:
        """Verifies a grading id exists

        Args:
            grading_id (str) : grading id

        Returns:
            bool: grading id exists or not
        """


class InMemGradingStorage(DeferredGradingStorage):
    def __init__(self) -> None:
        self.storage: dict[str, DeferredGradingResult] = dict()

    async def close(self) -> None:
        pass

    async def fetch(self, grading_id) -> DeferredGradingResult:
        return self.storage[grading_id]

    async def prune(self, grading_id: str | list[str]) -> int:
        if isinstance(grading_id, str):
            grading_ids = [grading_id]
        else:
            grading_ids = grading_id

        count = 0
        for gi in grading_ids:
            if gi in self.storage:
                count += 1
                del self.storage[gi]

        return count

    async def state(self, grading_id) -> DeferredGradingState:
        data: DeferredGradingResult = self.storage[grading_id]
        return DeferredGradingState(grading_id=grading_id, status=data.status)

    async def store(self, grading_id, data: DeferredGradingResult) -> None:
        self.storage[grading_id] = data

    async def valid(self, grading_id) -> bool:
        return grading_id in self.storage


class PostGresGradingStorage(DeferredGradingStorage):
    def __init__(self, database_url: str) -> None:
        self.database_url: str = database_url
        self.pool: asyncpg.Pool | None = None

    def _result_pack(self, result: SubmissionResult | None) -> str:
        if result:
            result_str: str = json.dumps({"result": asdict(result)})
        else:
            result_str = json.dumps({"result": None})

        return result_str

    def _result_unpack(self, result_str: str) -> SubmissionResult:
        result_obj = json.loads(result_str)
        result = result_obj["result"]
        return SubmissionResult(**result)

    async def _connect(self):
        self.pool = await asyncpg.create_pool(self.database_url)

    async def connect(self):
        await self._connect()

        # Create table if not exists
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deferred_grading (
                    grading_id VARCHAR(36) PRIMARY KEY,
                    status     VARCHAR(24) NOT NULL,
                    result     JSONB,
                    error      TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def fetch(self, grading_id) -> DeferredGradingResult:
        if not self.pool:
            await self._connect()

        query = """
            SELECT status, result, error FROM deferred_grading WHERE grading_id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, grading_id)
            logger.debug(f"{grading_id=}: {row=}")

            if row is None:
                raise KeyError(f"{grading_id} not found")

            result: SubmissionResult = self._result_unpack(row["result"])

            return DeferredGradingResult(
                result=result,
                status=DeferredGradingStatus(row["status"]),
                error=row["error"],
            )

    async def prune(self, grading_id: str | list[str]) -> int:
        if isinstance(grading_id, str):
            grading_ids = [grading_id]
        else:
            grading_ids = grading_id

        if not grading_ids:
            return 0

        query = """
            DELETE FROM deferred_grading WHERE grading_id = ANY($1)
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, grading_ids)

        rows_deleted = int(result.split()[-1])
        return rows_deleted

    async def state(self, grading_id) -> DeferredGradingState:
        if not self.pool:
            await self._connect()

        query = """
            SELECT status FROM deferred_grading WHERE grading_id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, grading_id)
            logger.debug(f"{grading_id=}: {row=}")

            if not row:
                raise KeyError(f"{grading_id} not found")

            status = DeferredGradingStatus(row["status"])
            return DeferredGradingState(grading_id=grading_id, status=status)

    async def store(self, grading_id, data: DeferredGradingResult) -> None:
        if not self.pool:
            await self._connect()

        result_str: str = self._result_pack(data.result)

        logger.debug(
            f"storing: {grading_id=}, {data.status.value=}, {data.error=}, {result_str=}"
        )

        query = """
            INSERT INTO deferred_grading (grading_id, status, result, error, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (grading_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                result = EXCLUDED.result,
                error = EXCLUDED.error,
                updated_at = NOW()
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                grading_id,
                data.status.value,
                result_str,
                data.error,
            )

    async def valid(self, grading_id) -> bool:
        query = """
            SELECT EXISTS(SELECT 1 FROM deferred_grading WHERE grading_id = $1)
        """

        async with self.pool.acquire() as conn:
            exists = await conn.fetchval(query, grading_id)

        return exists


async def process_deferred_grading(
    grade_id, grader, data, storage: DeferredGradingStorage
) -> None:
    try:
        logger.debug(f"deferred grading for {grade_id=}")
        result: SubmissionResult = await grade_responses(grader=grader, data=data)
        logger.debug(f"deferred grading for {grade_id=}: {result=}")

        await storage.store(
            grading_id=grade_id,
            data=DeferredGradingResult(
                result=result,
                status=DeferredGradingStatus.COMPLETED,
                error=None,
            ),
        )

    except Exception as e:
        logger.exception(f"deferred grading failed {grade_id=}: {e=}")
        await storage.store(
            grading_id=grade_id,
            data=DeferredGradingResult(
                result=None,
                status=DeferredGradingStatus.FAILED,
                error=str(e),
            ),
        )
