"""Unit tests for SSL configuration."""

import ssl
from unittest.mock import patch, MagicMock
from scenario_client.client import SSLConfig


class TestSSLConfig:
    """Test SSLConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SSLConfig()
        assert config.ssl_verify is True

    def test_config_with_false(self):
        """Test configuration with False."""
        config = SSLConfig(ssl_verify=False)
        assert config.ssl_verify is False

    def test_config_with_true(self):
        """Test configuration with True."""
        config = SSLConfig(ssl_verify=True)
        assert config.ssl_verify is True

    def test_config_with_ca_file_path(self):
        """Test configuration with CA file path."""
        ca_file = "/path/to/ca-bundle.crt"
        config = SSLConfig(ssl_verify=ca_file)
        assert config.ssl_verify == ca_file

    @patch("ssl.create_default_context")
    def test_config_with_ssl_context(self, mock_create_context):
        """Test configuration with SSL context."""
        mock_context = MagicMock(spec=ssl.SSLContext)
        config = SSLConfig(ssl_verify=mock_context)
        assert config.ssl_verify == mock_context


class TestSSLConfigFromEnv:
    """Test SSLConfig.from_env() method."""

    def test_from_env_default_none(self, monkeypatch):
        """Test from_env when SSL_CERT_FILE is not set."""
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_false_lowercase(self, monkeypatch):
        """Test from_env with 'false' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "false")
        config = SSLConfig.from_env()
        assert config.ssl_verify is False

    def test_from_env_false_uppercase(self, monkeypatch):
        """Test from_env with 'False' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "False")
        config = SSLConfig.from_env()
        assert config.ssl_verify is False

    def test_from_env_f(self, monkeypatch):
        """Test from_env with 'f' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "f")
        config = SSLConfig.from_env()
        assert config.ssl_verify is False

    def test_from_env_no(self, monkeypatch):
        """Test from_env with 'no' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "no")
        config = SSLConfig.from_env()
        assert config.ssl_verify is False

    def test_from_env_n(self, monkeypatch):
        """Test from_env with 'n' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "n")
        config = SSLConfig.from_env()
        assert config.ssl_verify is False

    def test_from_env_true_lowercase(self, monkeypatch):
        """Test from_env with 'true' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "true")
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_true_uppercase(self, monkeypatch):
        """Test from_env with 'True' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "True")
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_t(self, monkeypatch):
        """Test from_env with 't' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "t")
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_yes(self, monkeypatch):
        """Test from_env with 'yes' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "yes")
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_y(self, monkeypatch):
        """Test from_env with 'y' value."""
        monkeypatch.setenv("SSL_CERT_FILE", "y")
        config = SSLConfig.from_env()
        assert config.ssl_verify is True

    def test_from_env_with_ca_file(self, monkeypatch):
        """Test from_env with custom CA file path."""
        ca_file = "/path/to/ca-bundle.crt"
        monkeypatch.setenv("SSL_CERT_FILE", ca_file)
        config = SSLConfig.from_env()
        assert config.ssl_verify == ca_file

    def test_from_env_empty_string(self, monkeypatch):
        """Test from_env with empty string."""
        monkeypatch.setenv("SSL_CERT_FILE", "")
        config = SSLConfig.from_env()
        # Empty string is not a boolean keyword, so treated as path
        assert config.ssl_verify == ""


class TestGetSSLContext:
    """Test SSLConfig.get_ssl_context() method."""

    def test_get_ssl_context_true(self):
        """Test get_ssl_context with True."""
        config = SSLConfig(ssl_verify=True)
        result = config.get_ssl_context()
        assert result is True

    def test_get_ssl_context_false(self):
        """Test get_ssl_context with False."""
        config = SSLConfig(ssl_verify=False)
        result = config.get_ssl_context()
        assert result is False

    @patch("ssl.create_default_context")
    def test_get_ssl_context_with_ca_file(self, mock_create_context):
        """Test get_ssl_context with custom CA file path."""
        ca_file = "/path/to/ca-bundle.crt"
        mock_context = MagicMock(spec=ssl.SSLContext)
        mock_create_context.return_value = mock_context

        config = SSLConfig(ssl_verify=ca_file)
        result = config.get_ssl_context()

        mock_create_context.assert_called_once_with(cafile=ca_file)
        assert result == mock_context

    def test_get_ssl_context_with_ssl_context(self):
        """Test get_ssl_context with existing SSL context."""
        mock_context = MagicMock(spec=ssl.SSLContext)
        config = SSLConfig(ssl_verify=mock_context)
        result = config.get_ssl_context()
        assert result == mock_context

    @patch("ssl.create_default_context")
    def test_get_ssl_context_exception_handling(self, mock_create_context):
        """Test get_ssl_context falls back to True on exception."""
        ca_file = "/invalid/path/to/ca-bundle.crt"
        mock_create_context.side_effect = Exception("File not found")

        config = SSLConfig(ssl_verify=ca_file)
        result = config.get_ssl_context()

        assert result is True


class TestSSLConfigIntegration:
    """Test SSLConfig integration scenarios."""

    def test_config_in_aobench_default(self):
        """Test that AOBench uses default config from environment."""
        from scenario_client.client import AOBench

        client = AOBench(scenario_uri="https://test.com")
        assert client.config is not None
        assert isinstance(client.config, SSLConfig)

    def test_config_in_aobench_custom(self):
        """Test that AOBench accepts custom config."""
        from scenario_client.client import AOBench

        custom_config = SSLConfig(ssl_verify=False)
        client = AOBench(scenario_uri="https://test.com", config=custom_config)
        assert client.config == custom_config
        assert client.config.ssl_verify is False

    def test_config_ssl_context_used_in_requests(self):
        """Test that config SSL context is used in HTTP requests."""
        from scenario_client.client import AOBench

        config = SSLConfig(ssl_verify=False)
        client = AOBench(scenario_uri="https://test.com", config=config)

        ssl_context = client.config.get_ssl_context()
        assert ssl_context is False
