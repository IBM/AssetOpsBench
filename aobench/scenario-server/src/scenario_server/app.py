import logging
import os
import random
import string
import time

from litestar import Litestar, Request, get
from litestar.logging import LoggingConfig
from litestar.middleware import DefineMiddleware
from litestar.openapi.config import OpenAPIConfig
from litestar.response import Redirect
from litestar.types import ASGIApp, Receive, Scope, Send
from scenario_server.endpoints import (
    OPENAPI_CONFIG,
    ROUTE_HANDLERS,
    register_scenario_handlers,
    set_tracking_uri,
)
from scenario_server.grading import InMemGradingStorage, PostGresGradingStorage
from scenario_server.handlers.aob.aob import AOBScenarios
from scenario_server.handlers.aob_iot.aob_iot import AOBIoTScenarios
from scenario_server.handlers.aob_tsfm.aob_tsfm import AOBTSFMScenarios
from scenario_server.handlers.aob_workorders.aob_workorders import AOBWorkOrderScenarios

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")


class RequestTimingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if "http" == scope["type"]:
            request = Request(scope)

            bag: str = string.ascii_lowercase + string.digits
            rid: str = "".join(random.choices(bag, k=12))
            request.state["rid"] = rid

            rc = request.client
            rp = request.url.path

            logger.info(f"[{rid}] > request: {rp} {rc}")

            t1: float = time.perf_counter()

            async def send_wrapper(message):
                if message["type"] == "http.response.body" and not message.get(
                    "more_body", False
                ):
                    t3: float = time.perf_counter() - t1
                    logger.info(f"[{rid}] < response-bg: {rp} {rc} ~ {t3:0.5f}")

                await send(message)

            await self.app(scope, receive, send_wrapper)
            t2: float = time.perf_counter() - t1

            logger.info(f"[{rid}] < response+bg: {rp} {rc} ~ {t2:0.5f}")
        else:
            await self.app(scope, receive, send)


@get("/")
async def redirect_to_swagger() -> Redirect:
    return Redirect(path="/schema/swagger")


async def startup(app: Litestar) -> None:
    try:
        pg_user: str = os.environ["POSTGRES_USERNAME"]
        pg_pass: str = os.environ["POSTGRES_PASSWORD"]
        pg_db: str = os.environ["POSTGRES_DATABASE"]
        pg_host: str = os.environ["POSTGRES_HOST"]
        pg_port: str = os.environ["POSTGRES_PORT"]

        pg_url: str = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

        deferred_grading_storage = PostGresGradingStorage(database_url=pg_url)
        await deferred_grading_storage.connect()
        logger.info(f"deferred grading storage: {pg_host}:{pg_db}")
    except Exception as e:
        logger.exception(
            f"failed to init deferred grading storage, using default: {e=}"
        )
        deferred_grading_storage = InMemGradingStorage()
        logger.info("deferred grading storage: inmemory")

    app.state.storage = deferred_grading_storage


async def shutdown(app: Litestar) -> None:
    await app.state.storage.close()


def get_app(
    handlers: list = [],
    include_default_handlers: bool = True,
    tracking_uri: str = "",
    openapi_config: OpenAPIConfig | None = None,
    debug: bool = False,
) -> Litestar:
    if tracking_uri != "":
        logger.info(f"{tracking_uri=}")
        set_tracking_uri(tracking_uri=tracking_uri)

    if len(handlers) > 0:
        register_scenario_handlers(handlers=handlers)

    if include_default_handlers:
        register_scenario_handlers(
            handlers=[
                AOBScenarios,
                AOBIoTScenarios,
                AOBTSFMScenarios,
                AOBWorkOrderScenarios,
            ]
        )

    openapi_cfg: OpenAPIConfig = openapi_config or OPENAPI_CONFIG

    logging_config = LoggingConfig(
        root={"level": "INFO", "handlers": ["queue_listener"]},
        formatters={
            "standard": {
                "format": "[%(levelname)1.1s %(asctime)s %(filename)-24s:%(lineno)5d] :: %(message)s"
            }
        },
        log_exceptions="always",
    )

    app = Litestar(
        debug=debug,
        middleware=[DefineMiddleware(RequestTimingMiddleware)],
        route_handlers=[redirect_to_swagger] + ROUTE_HANDLERS,
        openapi_config=openapi_cfg,
        on_startup=[startup],
        on_shutdown=[shutdown],
        logging_config=logging_config,
    )

    return app
