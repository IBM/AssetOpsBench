"""Tests for package initialization and exports."""


def test_aobench_import():
    """Test that AOBench can be imported from package root."""
    from scenario_client import AOBench

    assert AOBench is not None


def test_ssl_config_import():
    """Test that SSLConfig can be imported from package root."""
    from scenario_client import SSLConfig

    assert SSLConfig is not None


def test_tracking_context_import():
    """Test that TrackingContext can be imported from package root."""
    from scenario_client import TrackingContext

    assert TrackingContext is not None


def test_version_attribute():
    """Test that version is available."""
    import scenario_client

    assert hasattr(scenario_client, "__version__")
    assert scenario_client.__version__ == "1.0.0"


def test_version_in_client_module():
    """Test that version is available in client module."""
    from scenario_client.client import __version__

    assert __version__ == "1.0.0"


def test_all_exports():
    """Test __all__ exports."""
    import scenario_client

    assert hasattr(scenario_client, "__all__")
    assert "AOBench" in scenario_client.__all__
    assert "SSLConfig" in scenario_client.__all__
    assert "TrackingContext" in scenario_client.__all__


def test_aobench_instantiation():
    """Test that AOBench can be instantiated."""
    from scenario_client import AOBench

    client = AOBench(scenario_uri="https://test.com")
    assert client.scenario_uri == "https://test.com"
    assert client.tracking_uri == ""


def test_aobench_with_custom_config():
    """Test that AOBench can be instantiated with custom config."""
    from scenario_client import AOBench, SSLConfig

    config = SSLConfig(ssl_verify=False)
    client = AOBench(scenario_uri="https://test.com", config=config)
    assert client.scenario_uri == "https://test.com"
    assert client.config.ssl_verify is False


def test_ssl_config_instantiation():
    """Test that SSLConfig can be instantiated."""
    from scenario_client import SSLConfig

    config = SSLConfig()
    assert config.ssl_verify is True

    config_false = SSLConfig(ssl_verify=False)
    assert config_false.ssl_verify is False


def test_tracking_context_instantiation():
    """Test that TrackingContext can be instantiated."""
    from scenario_client import TrackingContext

    context = TrackingContext(
        uri="https://mlflow.example.com", experiment_id="exp-123", run_id="run-456"
    )
    assert context.uri == "https://mlflow.example.com"
    assert context.experiment_id == "exp-123"
    assert context.run_id == "run-456"
