"""Scenario client for interacting with scenario servers and MLflow tracking."""

import logging
import ssl
from dataclasses import dataclass
from enum import Enum
from os import environ
from typing import Any, Callable, Optional

import httpx
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

# Version of the client library
__version__ = "1.0.0"


def get_timeout() -> float:
    """
    Get the timeout value from the SCENARIO_CLIENT_TIMEOUT environment variable.
    Maximum value is 180 seconds. Defaults to 30 seconds if not set or invalid.
    """
    try:
        timeout = float(environ.get("SCENARIO_CLIENT_TIMEOUT", "30"))
        # Ensure timeout doesn't exceed 180 seconds
        return min(timeout, 180.0)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid SCENARIO_CLIENT_TIMEOUT value, using default of 30 seconds"
        )
        return 30.0


@dataclass
class TrackingContext:
    """MLflow tracking context for experiment tracking.

    This class encapsulates the MLflow tracking information needed to
    associate scenario runs with experiments and runs.

    Attributes:
        uri (str): MLflow tracking server URI
        experiment_id (str): MLflow experiment ID
        run_id (str): MLflow run ID

    Examples:
        >>> context = TrackingContext(
        ...     uri="https://mlflow.example.com",
        ...     experiment_id="exp-123",
        ...     run_id="run-456"
        ... )
        >>> print(context.uri)
        'https://mlflow.example.com'
    """

    uri: str
    experiment_id: str
    run_id: str


@dataclass
class SSLConfig:
    """SSL configuration for the scenario client.

    This class manages SSL verification settings and can be initialized from
    environment variables or provided directly.

    Attributes:
        ssl_verify (bool | ssl.SSLContext | str): SSL verification setting.
            - True: Use default SSL verification (default)
            - False: Disable SSL verification (not recommended for production)
            - ssl.SSLContext: Custom SSL context
            - str: Path to CA certificate file

    Examples:
        >>> # Default configuration (SSL verification enabled)
        >>> config = SSLConfig()

        >>> # Disable SSL verification
        >>> config = SSLConfig(ssl_verify=False)

        >>> # Use custom CA file
        >>> config = SSLConfig(ssl_verify="/path/to/ca-bundle.crt")

        >>> # Load from environment variables
        >>> config = SSLConfig.from_env()
    """

    ssl_verify: bool | ssl.SSLContext | str = True

    @classmethod
    def from_env(cls) -> "SSLConfig":
        """Create configuration from environment variables.

        Reads the SSL_CERT_FILE environment variable to configure SSL verification.

        Environment Variables:
            SSL_CERT_FILE: Controls SSL verification behavior
                - None/unset: Use default verification (True)
                - "false", "f", "no", "n": Disable verification (False)
                - "true", "t", "yes", "y": Enable verification (True)
                - Path to file: Use custom CA bundle

        Returns:
            SSLConfig: Configuration instance with settings from environment.

        Examples:
            >>> import os
            >>> os.environ["SSL_CERT_FILE"] = "false"
            >>> config = SSLConfig.from_env()
            >>> config.ssl_verify
            False
        """
        ssl_cert_file = environ.get("SSL_CERT_FILE", None)

        if ssl_cert_file is None:
            ssl_verify = True
        elif str(ssl_cert_file).lower() in ["f", "false", "n", "no"]:
            ssl_verify = False
        elif str(ssl_cert_file).lower() in ["t", "true", "y", "yes"]:
            ssl_verify = True
        else:
            # It's a path to a CA file
            ssl_verify = ssl_cert_file

        return cls(ssl_verify=ssl_verify)

    def get_ssl_context(self) -> bool | ssl.SSLContext:
        """Get the SSL context for HTTP requests.

        Converts the ssl_verify setting into a format suitable for httpx.Client.

        Returns:
            bool | ssl.SSLContext: SSL verification setting for httpx.
                - True: Use default SSL verification
                - False: Disable SSL verification
                - ssl.SSLContext: Custom SSL context

        Raises:
            Exception: If SSL context creation fails, returns True (default verification).

        Examples:
            >>> config = SSLConfig(ssl_verify=True)
            >>> config.get_ssl_context()
            True

            >>> config = SSLConfig(ssl_verify="/path/to/ca.crt")
            >>> ctx = config.get_ssl_context()
            >>> isinstance(ctx, (bool, ssl.SSLContext))
            True
        """
        try:
            if isinstance(self.ssl_verify, bool):
                logger.debug(f"setting verify ssl to {self.ssl_verify}")
                return self.ssl_verify
            elif isinstance(self.ssl_verify, ssl.SSLContext):
                logger.debug("using provided SSL context")
                return self.ssl_verify
            else:
                # It's a string path to CA file
                logger.debug(f"creating SSL context with CA file: {self.ssl_verify}")
                return ssl.create_default_context(cafile=self.ssl_verify)
        except Exception as e:
            logger.exception(f"failed to create SSL context: {e}, defaulting to True")
            return True


# Default SSL configuration from environment variables
_default_config: SSLConfig = SSLConfig.from_env()


class MLflowAutoLogger(Enum):
    langchain = "langchain"
    default = "default"


class AOBench:
    """Client for interacting with scenario servers and running benchmarks.

    This class provides methods to load scenario sets, execute scenarios with
    optional MLflow tracking, and submit results for grading. It supports both
    synchronous and asynchronous execution patterns.

    Attributes:
        scenario_uri (str): Base URL of the scenario server
        tracking_uri (str): Optional MLflow tracking server URL
        config (SSLConfig): SSL configuration for the client

    Examples:
        >>> client = AOBench(scenario_uri="https://scenarios.example.com")
        >>> types = client.scenario_types()
        >>> scenario_set, tracking = client.scenario_set("my-set", tracking=True)
        >>> answer = client.run(my_function, "scenario-1", **scenario["inputs"])
        >>> results = client.grade("my-set", [answer], tracking)
    """

    def __init__(
        self,
        scenario_uri: str,
        tracking_uri: str = "",
        config: Optional[SSLConfig] = None,
    ):
        """Initialize the AOBench client.

        Args:
            scenario_uri (str): Base URL of the scenario server
            tracking_uri (str, optional): MLflow tracking server URL. If provided,
                overrides the tracking URI from the scenario server. Defaults to "".
            config (SSLConfig, optional): SSL configuration. If not provided,
                uses default configuration from environment variables. Defaults to None.

        Examples:
            >>> # Use default configuration from environment
            >>> client = AOBench(scenario_uri="https://scenarios.example.com")

            >>> # Provide custom configuration
            >>> config = SSLConfig(ssl_verify=False)
            >>> client = AOBench(
            ...     scenario_uri="https://scenarios.example.com",
            ...     tracking_uri="https://mlflow.example.com",
            ...     config=config
            ... )
        """
        self.scenario_uri: str = scenario_uri
        self.tracking_uri: str = tracking_uri
        self.config: SSLConfig = config if config is not None else _default_config
        self._headers: dict[str, str] = {
            "User-Agent": f"scenario-client/{__version__}",
        }

    async def arun(
        self,
        afunc,
        scenario_id,
        run_name: str = "",
        post_process: Optional[Callable[[Any], str]] = None,
        tracking_context: Optional[TrackingContext] = None,
        **kwargs,
    ):
        """Execute an asynchronous function for a scenario with optional tracking.

        Args:
            afunc: Async function to execute. Will be called with **kwargs.
            scenario_id: Unique identifier for the scenario being executed.
            run_name (str, optional): Custom name for the MLflow run. Defaults to "".
            post_process (Callable[[Any], str] | None, optional): Function to transform
                the result before returning. Takes any type and returns a string. Defaults to None.
            tracking_context (TrackingContext | None, optional): MLflow tracking context from
                scenario_set(). If provided, execution is tracked. Defaults to None.
            **kwargs: Arguments to pass to afunc.

        Returns:
            dict: Dictionary containing:
                - scenario_id (str): The scenario identifier
                - answer: The result from afunc (possibly post-processed)

        Raises:
            Exception: Re-raises any exception from afunc execution.

        Examples:
            >>> async def my_async_func(prompt: str) -> str:
            ...     return f"Response to: {prompt}"
            >>>
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> answer = await client.arun(
            ...     afunc=my_async_func,
            ...     scenario_id="scenario-1",
            ...     prompt="What is 2+2?"
            ... )
            >>> print(answer)
            {'scenario_id': 'scenario-1', 'answer': 'Response to: What is 2+2?'}
        """
        try:
            if tracking_context:
                rid: str = tracking_context.run_id

                if not mlflow.active_run():
                    mlflow.start_run(run_id=rid)

                if run_name != "":
                    mlflow.set_tag("mlflow.runName", run_name)

                trace_id: str | None = None
                with mlflow.start_span(name=scenario_id) as span:
                    result = await afunc(**kwargs)

                    trace_id = span.trace_id

                try:
                    if trace_id is None:
                        raise ValueError("failed to identify trace id")

                    mlflow.set_trace_tag(
                        trace_id=trace_id, key="mlflow.runId", value=rid
                    )
                    mlflow.set_trace_tag(
                        trace_id=trace_id, key="scenario_id", value=scenario_id
                    )
                except Exception as ex:
                    logger.error(f"failed to associate trace to run: {ex}")

                mlflow.end_run()

            else:
                result = await afunc(**kwargs)

            if post_process:
                result: str = post_process(result)

            logger.debug(f"{result=}")

            answer = {
                "scenario_id": str(scenario_id),
                "answer": result,
            }

        except Exception as e:
            logger.exception(f"aobench.run failed: {e=}")
            raise

        return answer

    def run(
        self,
        func,
        scenario_id,
        run_name: str = "",
        post_process: Optional[Callable[[Any], str]] = None,
        tracking_context: Optional[TrackingContext] = None,
        **kwargs,
    ):
        """Execute a synchronous function for a scenario with optional tracking.

        Args:
            func: Synchronous function to execute. Will be called with **kwargs.
            scenario_id: Unique identifier for the scenario being executed.
            run_name (str, optional): Custom name for the MLflow run. Defaults to "".
            post_process (Callable[[Any], str] | None, optional): Function to transform
                the result before returning. Takes any type and returns a string. Defaults to None.
            tracking_context (TrackingContext | None, optional): MLflow tracking context from
                scenario_set(). If provided, execution is tracked. Defaults to None.
            **kwargs: Arguments to pass to func.

        Returns:
            dict: Dictionary containing:
                - scenario_id (str): The scenario identifier
                - answer: The result from func (possibly post-processed)

        Raises:
            Exception: Re-raises any exception from func execution.

        Examples:
            >>> def my_func(prompt: str) -> str:
            ...     return f"Response to: {prompt}"
            >>>
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> answer = client.run(
            ...     func=my_func,
            ...     scenario_id="scenario-1",
            ...     prompt="What is 2+2?"
            ... )
            >>> print(answer)
            {'scenario_id': 'scenario-1', 'answer': 'Response to: What is 2+2?'}
        """
        try:
            if tracking_context:
                rid: str = tracking_context.run_id

                if not mlflow.active_run():
                    mlflow.start_run(run_id=rid)

                if run_name != "":
                    mlflow.set_tag("mlflow.runName", run_name)

                trace_id: str | None = None
                with mlflow.start_span(name=scenario_id) as span:
                    result = func(**kwargs)

                    trace_id = span.trace_id

                try:
                    if trace_id is None:
                        raise ValueError("failed to identify trace id")

                    mlflow.set_trace_tag(
                        trace_id=trace_id, key="mlflow.runId", value=rid
                    )
                    mlflow.set_trace_tag(
                        trace_id=trace_id, key="scenario_id", value=scenario_id
                    )
                except Exception as ex:
                    logger.error(f"failed to associate trace to run: {ex}")

                mlflow.end_run()

            else:
                result = func(**kwargs)

            if post_process:
                result: str = post_process(result)

            logger.debug(f"{result=}")

            answer = {
                "scenario_id": str(scenario_id),
                "answer": result,
            }

        except Exception as e:
            logger.exception(f"aobench.run failed: {e=}")
            raise

        return answer

    def scenario_types(self) -> dict:
        """Retrieve available scenario types from the server.

        Returns:
            dict: Dictionary containing available scenario types and their metadata.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> types = client.scenario_types()
            >>> print(types)
            {'types': [{'id': 'qa', 'name': 'Question Answering'}, ...]}
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(), timeout=timeout
        ) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-types"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(f"{endpoint}", headers=self._headers)
            r.raise_for_status()

            return r.json()

    def scenario_set(
        self,
        scenario_set_id: str,
        tracking: bool,
        mlflow_logger: MLflowAutoLogger = MLflowAutoLogger.langchain,
    ) -> tuple[dict, Optional[TrackingContext]]:
        """Load a scenario set with optional MLflow tracking setup.

        Args:
            scenario_set_id (str): Unique identifier for the scenario set to load.
            tracking (bool): Whether to enable MLflow tracking. If True, initializes
                MLflow tracking and returns tracking context.
            mlflow_logger (MLflowAutoLogger): Set mlflow autologging method, defaults
                to `MLflowAutoLogger.langchain` which enables `mlflow.langchain.autolog()`.
                The alternative is `MLflowAutoLogger.default` which enables
                `mlflow.autolog()`

        Returns:
            tuple[dict, TrackingContext | None]: A tuple containing:
                - scenario_set (dict): Dictionary with 'title' and 'scenarios' keys
                - tracking_context (TrackingContext | None): MLflow tracking context if tracking
                  is enabled, None otherwise.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> scenario_set, tracking = client.scenario_set("my-set", tracking=True)
            >>> print(scenario_set['title'])
            'My Scenario Set'
            >>> print(len(scenario_set['scenarios']))
            10
            >>> if tracking:
            ...     print(tracking.experiment_id)
            'exp-123'
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(), timeout=timeout
        ) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-set/{scenario_set_id}"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(
                f"{endpoint}",
                params={"tracking": tracking},
                headers=self._headers,
            )
            r.raise_for_status()

            rsp = r.json()

            scenario_set = {
                "title": rsp["title"],
                "scenarios": rsp["scenarios"],
            }

            if tracking:
                try:
                    tracking_data = rsp["tracking_context"]

                    tracking_uri = tracking_data["uri"]
                    if self.tracking_uri != "" and self.tracking_uri != tracking_uri:
                        logger.info(f"tracking uri differs: {tracking_uri}")
                        tracking_uri = self.tracking_uri

                    experiment_id = tracking_data["experiment_id"]
                    run_id = tracking_data["run_id"]

                    logger.info(f"{tracking_uri=} / {experiment_id=} / {run_id=}")

                    mlflow.set_tracking_uri(uri=tracking_uri)
                    if mlflow_logger == MLflowAutoLogger.langchain:
                        mlflow.langchain.autolog()
                    else:
                        mlflow.autolog()

                    mlflow.set_experiment(experiment_id=experiment_id)

                    tracking_context = TrackingContext(
                        uri=tracking_uri, experiment_id=experiment_id, run_id=run_id
                    )
                    return scenario_set, tracking_context
                except Exception as e:
                    logger.exception(f"failed to init tracking: {e=}")

            return scenario_set, None

    def grade(
        self,
        scenario_set_id: str,
        answers,
        tracking_context: Optional[TrackingContext],
    ) -> dict:
        """Submit answers for immediate grading.

        Args:
            scenario_set_id (str): Identifier of the scenario set being graded.
            answers: List of answer dictionaries, each containing 'scenario_id'
                and 'answer' keys.
            tracking_context (TrackingContext | None): MLflow tracking context from scenario_set().
                If provided, ends the MLflow run and includes tracking info in submission.

        Returns:
            dict: Grading results containing score, details, and other metrics.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
            httpx.TimeoutException: If the grading request times out (90s read timeout).
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> answers = [
            ...     {'scenario_id': 'scenario-1', 'answer': '4'},
            ...     {'scenario_id': 'scenario-2', 'answer': 'Paris'}
            ... ]
            >>> results = client.grade("my-set", answers, None)
            >>> print(results['score'])
            0.95
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(),
            timeout=timeout,
        ) as client:
            endpoint: str = f"{self.scenario_uri}/scenario-set/{scenario_set_id}/grade"
            logger.debug(f"{endpoint=}")

            jsn = {
                "scenario_set_id": scenario_set_id,
                "submission": answers,
            }

            if tracking_context is not None:
                mlflow.end_run()
                jsn["tracking_context"] = {
                    "experiment_id": tracking_context.experiment_id,
                    "run_id": tracking_context.run_id,
                }

            r: httpx.Response = client.post(
                f"{endpoint}", json=jsn, headers=self._headers
            )
            r.raise_for_status()

            return r.json()

    def deferred_grading(
        self,
        scenario_set_id: str,
        answers,
        tracking_context: Optional[TrackingContext],
    ) -> dict:
        """Submit answers for deferred (asynchronous) grading.

        Use this method for long-running grading tasks. The server will process
        the grading as a background task and you can check status and retrieve results
        using the returned grading_id.

        Args:
            scenario_set_id (str): Identifier of the scenario set being graded.
            answers: List of answer dictionaries, each containing 'scenario_id'
                and 'answer' keys.
            tracking_context (TrackingContext | None): MLflow tracking context from scenario_set().
                If provided, ends the MLflow run and includes tracking info in submission.

        Returns:
            dict: Response containing:
                - grading_id (str): Unique identifier for this grading job
                - status (str): Initial status (typically "pending")

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> answers = [{'scenario_id': 'scenario-1', 'answer': '4'}]
            >>> response = client.deferred_grading("my-set", answers, None)
            >>> grading_id = response['grading_id']
            >>> print(grading_id)
            'grading-abc123'
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(),
            timeout=timeout,
        ) as client:
            endpoint: str = (
                f"{self.scenario_uri}/scenario-set/{scenario_set_id}/deferred-grading"
            )
            logger.debug(f"{endpoint=}")

            jsn = {
                "scenario_set_id": scenario_set_id,
                "submission": answers,
            }

            if tracking_context is not None:
                mlflow.end_run()
                jsn["tracking_context"] = {
                    "experiment_id": tracking_context.experiment_id,
                    "run_id": tracking_context.run_id,
                }

            r: httpx.Response = client.post(
                f"{endpoint}", json=jsn, headers=self._headers
            )
            r.raise_for_status()

            return r.json()

    def deferred_grading_status(self, grading_id) -> dict:
        """Check the status of a deferred grading job.

        Args:
            grading_id: Unique identifier returned by deferred_grading().

        Returns:
            dict: Status information containing:
                - status (str): Current status ("pending", "processing", "completed", "failed")
                - progress (float, optional): Progress percentage if available
                - Other status-specific fields

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code
                (e.g., 404 if grading_id not found).
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> status = client.deferred_grading_status("grading-abc123")
            >>> print(status['status'])
            'completed'
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(),
            timeout=timeout,
        ) as client:
            endpoint: str = f"{self.scenario_uri}/deferred-grading/{grading_id}/status"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(endpoint, headers=self._headers)
            r.raise_for_status()

            return r.json()

    def deferred_grading_result(self, grading_id) -> dict:
        """Retrieve the results of a completed deferred grading job.

        Args:
            grading_id: Unique identifier returned by deferred_grading().

        Returns:
            dict: Grading results containing score, details, and other metrics.
                Same format as returned by grade().

        Raises:
            httpx.HTTPStatusError: If the server returns an error status code.
                Returns 425 (Too Early) if grading is not yet complete.
                Returns 404 if grading_id not found.
            httpx.RequestError: If the request fails due to network issues.

        Examples:
            >>> client = AOBench(scenario_uri="https://example.com")
            >>> # Wait for grading to complete first
            >>> results = client.deferred_grading_result("grading-abc123")
            >>> print(results['score'])
            0.95
        """
        timeout: float = get_timeout()
        with httpx.Client(
            verify=self.config.get_ssl_context(),
            timeout=timeout,
        ) as client:
            endpoint: str = f"{self.scenario_uri}/deferred-grading/{grading_id}/result"
            logger.debug(f"{endpoint=}")

            r: httpx.Response = client.get(endpoint, headers=self._headers)
            r.raise_for_status()

            return r.json()
