import asyncio
import logging
import os
import uuid

import mlflow
from litestar import Response, get, post
from litestar.background_tasks import BackgroundTask
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.handlers.http_handlers.base import HTTPRouteHandler
from litestar.openapi.config import OpenAPIConfig
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_202_ACCEPTED,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from scenario_server.entities import (
    ScenarioGrade,
    ScenarioSet,
    ScenarioType,
    SubmissionResult,
)
from scenario_server.grading import (
    DeferredGradingResult,
    DeferredGradingState,
    DeferredGradingStatus,
    DeferredGradingStorage,
    grade_responses,
    process_deferred_grading,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


REGISTERED_SCENARIO_HANDLERS = dict()


def register_scenario_handlers(handlers: list):
    global REGISTERED_SCENARIO_HANDLERS

    for handler in handlers:
        try:
            REGISTERED_SCENARIO_HANDLERS[handler.id] = handler()
        except Exception as e:
            logger.exception(f"failed to load {handler.title=}: {e=}")


TRACKING_URI: str = ""


def set_tracking_uri(tracking_uri: str):
    global TRACKING_URI

    TRACKING_URI = tracking_uri
    mlflow.set_tracking_uri(uri=tracking_uri)


class Answer(BaseModel):
    scenario_id: str = Field(
        description="Unique identifier for the scenario being answered"
    )
    answer: str = Field(
        description=(
            "Serialized answer content as a string. The answer must be encoded "
            "using appropriate serialization methods (e.g., json.dumps() for JSON objects, "
            "base64.b64encode() for binary data) depending on the handler implementation. "
            "Refer to the specific scenario handler documentation for the expected format."
        )
    )


class BuildDate(BaseModel):
    build_date: str


class TrackingContext(BaseModel):
    uri: str = TRACKING_URI
    experiment_id: str
    run_id: str


class Submission(BaseModel):
    submission: list[Answer] = Field(
        description="List of answers for one or more scenarios in this submission"
    )
    tracking_context: TrackingContext | None = None


@get("/build-date")
async def build_date() -> BuildDate:
    return BuildDate(build_date=os.getenv("BUILD_DATE", "unknown"))


@post("/scenario-set/{scenario_set_id: str}/deferred-grading")
async def deferred_grading(
    scenario_set_id: str, data: Submission, state: State
) -> Response[DeferredGradingState]:
    if scenario_set_id not in REGISTERED_SCENARIO_HANDLERS.keys():
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"no scenario set {scenario_set_id}",
        )

    try:
        grading_id = str(uuid.uuid4())
        storage: DeferredGradingStorage = state.storage

        await storage.store(
            grading_id=grading_id,
            data=DeferredGradingResult(
                result=None,
                status=DeferredGradingStatus.PROCESSING,
                error=None,
            ),
        )
    except Exception as e:
        logger.exception(f"deferred grading storage failed: {e=}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"deferred storage failed",
        )

    try:
        grading_fn = REGISTERED_SCENARIO_HANDLERS[scenario_set_id].grade_responses
        submission = data.model_dump()

        return Response(
            content=DeferredGradingState(
                grading_id=grading_id,
                status=DeferredGradingStatus.PROCESSING,
            ),
            background=BackgroundTask(
                process_deferred_grading,
                grading_id,
                grading_fn,
                submission,
                storage,
            ),
        )
    except Exception as e:
        logger.exception(f"grading failed: {e}")
        await storage.store(
            grading_id=grading_id,
            data=DeferredGradingResult(
                result=None,
                status=DeferredGradingStatus.FAILED,
                error=f"{e}",
            ),
        )

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"grading failed {scenario_set_id}",
        )


@get("/deferred-grading/{grading_id: str}/status")
async def deferred_grading_status(
    grading_id: str, state: State
) -> DeferredGradingState:
    try:
        storage: DeferredGradingStorage = state.storage
        return await storage.state(grading_id=grading_id)
    except KeyError as ke:
        logger.error(f"invalid {grading_id=}: {ke=}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"grading id not found: {grading_id}",
        )
    except Exception as ex:
        logger.exception(f"failed to fetch status {grading_id=}: {ex=}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to determine status of {grading_id=}",
        )


@get("deferred-grading/{grading_id: str}/result")
async def deferred_grading_result(grading_id: str, state: State) -> list[ScenarioGrade]:
    try:
        storage: DeferredGradingStorage = state.storage
        grading_state: DeferredGradingState = await storage.state(grading_id=grading_id)
        if grading_state.status == DeferredGradingStatus.PROCESSING:
            raise HTTPException(
                status_code=HTTP_202_ACCEPTED,
                detail="grading still progressing",
            )

        if grading_state.status == DeferredGradingStatus.FAILED:
            e: DeferredGradingResult = await storage.fetch(grading_id=grading_id)
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"grading failed: {e.error}",
            )

        result: DeferredGradingResult = await storage.fetch(grading_id=grading_id)
        return result.result
    except HTTPException as he:
        logger.exception(f"{he=}")
        raise

    except KeyError as ke:
        logger.exception(f"invalid {grading_id=}: {ke=}")
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"grading id not found: {grading_id}",
        )

    except Exception as ex:
        logger.exception(f"failed to fetch status/result {grading_id=}: {ex=}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to fetch result {grading_id=}",
        )


@get("/scenario-set/{scenario_set_id: str}")
async def fetch_scenario(scenario_set_id: str, tracking: bool = False) -> dict:
    if scenario_set_id not in REGISTERED_SCENARIO_HANDLERS.keys():
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"no scenario set {scenario_set_id}",
        )

    title: str = REGISTERED_SCENARIO_HANDLERS[scenario_set_id].title
    scenario_set: ScenarioSet = REGISTERED_SCENARIO_HANDLERS[
        scenario_set_id
    ].fetch_scenarios()

    if tracking and TRACKING_URI:
        logger.info(f"{tracking=} and {TRACKING_URI=}")

        def mlflow_start_run(scenario_set_title):
            client = MlflowClient()

            experiment = client.get_experiment_by_name(scenario_set_title)
            if experiment is None:
                experiment_id = client.create_experiment(scenario_set_title)
            else:
                experiment_id = experiment.experiment_id

            run = client.create_run(
                experiment_id=experiment_id,
                run_name=f"{uuid.uuid4()}",
            )
            run_id = run.info.run_id

            client.set_terminated(run_id=run_id)
            logger.debug(f"{experiment_id=}, {run_id=}")

            return experiment_id, run_id

        eid, rid = await asyncio.to_thread(mlflow_start_run, title)

        return {
            "title": title,
            "scenarios": scenario_set,
            "tracking_context": {
                "uri": TRACKING_URI,
                "experiment_id": eid,
                "run_id": rid,
            },
        }

    return {
        "title": title,
        "scenarios": scenario_set,
    }


@post("/scenario-set/{scenario_set_id: str}/grade")
async def grade_submission(
    scenario_set_id: str, data: Submission
) -> list[ScenarioGrade]:
    if scenario_set_id not in REGISTERED_SCENARIO_HANDLERS.keys():
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"no scenario set {scenario_set_id}",
        )

    try:
        grading_fn = REGISTERED_SCENARIO_HANDLERS[scenario_set_id].grade_responses
        results: SubmissionResult = await grade_responses(
            grader=grading_fn,
            data=data.model_dump(),
        )

        return results
    except Exception as e:
        logger.exception(f"grading failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"grading failed {scenario_set_id}",
        )


@get("/health")
async def health() -> dict[str, int]:
    return {"status": HTTP_200_OK}


@get("/scenario-types")
async def scenario_types() -> list[ScenarioType]:
    """Get all scenario types"""
    return [rsh.scenario_type() for rsh in REGISTERED_SCENARIO_HANDLERS.values()]


OPENAPI_CONFIG = OpenAPIConfig(
    title="Asset Operations Bench",
    description="",
    version="0.0.1",
)

ROUTE_HANDLERS: list[HTTPRouteHandler] = [
    build_date,
    health,
    scenario_types,
    fetch_scenario,
    grade_submission,
    deferred_grading,
    deferred_grading_status,
    deferred_grading_result,
]
