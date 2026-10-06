from pathlib import Path

import pytest

from neuralguard.config import ConfigError, Settings


def test_defaults_are_valid():
    settings = Settings()
    assert settings.kafka_topic == "network-traffic"
    assert settings.kafka_ports == (9092,)
    assert settings.es_ports == (9200,)


def test_from_env_parses_values():
    settings = Settings.from_env(
        {
            "NEURALGUARD_KAFKA_BOOTSTRAP_SERVERS": "kafka1:29092, kafka2:29093",
            "NEURALGUARD_ES_HOSTS": "https://es.example:9243",
            "NEURALGUARD_ES_VERIFY_CERTS": "false",
            "NEURALGUARD_THREAT_THRESHOLD": "0.7",
            "NEURALGUARD_MODEL_PATH": "/tmp/m.joblib",
            "NEURALGUARD_LOG_FORMAT": "JSON",
            "NEURALGUARD_ES_USERNAME": "elastic",
            "NEURALGUARD_ES_PASSWORD": "secret",
        }
    )
    assert settings.kafka_bootstrap_servers == ("kafka1:29092", "kafka2:29093")
    assert settings.kafka_ports == (29092, 29093)
    assert settings.es_ports == (9243,)
    assert settings.es_verify_certs is False
    assert settings.threat_threshold == 0.7
    assert settings.max_clock_skew_seconds == 300.0  # the default
    assert settings.model_path == Path("/tmp/m.joblib")
    assert settings.log_format == "json"
    assert "secret" not in repr(settings)


def test_blank_env_values_are_ignored():
    assert Settings.from_env({"NEURALGUARD_KAFKA_TOPIC": "  "}).kafka_topic == "network-traffic"


@pytest.mark.parametrize(
    "env",
    [
        {"NEURALGUARD_THREAT_THRESHOLD": "abc"},
        {"NEURALGUARD_THREAT_THRESHOLD": "0"},
        {"NEURALGUARD_THREAT_THRESHOLD": "1.5"},
        {"NEURALGUARD_WINDOW_SECONDS": "-1"},
        {"NEURALGUARD_MAX_CLOCK_SKEW_SECONDS": "-5"},
        {"NEURALGUARD_MAX_CLOCK_SKEW_SECONDS": "nan"},
        {"NEURALGUARD_ES_VERIFY_CERTS": "maybe"},
        {"NEURALGUARD_LOG_FORMAT": "xml"},
        {"NEURALGUARD_ES_USERNAME": "elastic"},
        {"NEURALGUARD_ES_API_KEY": "k", "NEURALGUARD_ES_USERNAME": "u",
         "NEURALGUARD_ES_PASSWORD": "p"},
    ],
)
def test_invalid_env_raises(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_with_overrides_ignores_none():
    settings = Settings()
    assert settings.with_overrides(kafka_topic=None) is settings
    assert settings.with_overrides(kafka_topic="t").kafka_topic == "t"


def test_ipv6_and_userinfo_ports():
    settings = Settings(es_hosts=("http://user@[::1]:9201", "http://es"))
    assert settings.es_ports == (9201, 9200)


def test_max_clock_skew_from_env():
    settings = Settings.from_env({"NEURALGUARD_MAX_CLOCK_SKEW_SECONDS": "0"})
    assert settings.max_clock_skew_seconds == 0.0  # 0 turns the check off


def test_credentials_in_an_elasticsearch_url_are_masked_in_the_repr():
    settings = Settings(es_hosts=("https://elastic:s3cr3t@es:9200", "http://es2:9200"))
    text = repr(settings)
    assert "s3cr3t" not in text
    assert "'https://***@es:9200', 'http://es2:9200'" in text
    assert settings.es_hosts[0] == "https://elastic:s3cr3t@es:9200"  # only the repr


def test_alert_corroboration_from_env():
    settings = Settings.from_env(
        {"NEURALGUARD_ALERT_MIN_HITS": "5", "NEURALGUARD_ALERT_CORROBORATION_SECONDS": "12.5"}
    )
    assert (settings.alert_min_hits, settings.alert_corroboration_seconds) == (5, 12.5)
    assert (Settings().alert_min_hits, Settings().alert_corroboration_seconds) == (3, 30.0)


@pytest.mark.parametrize(
    "env",
    [
        {"NEURALGUARD_ALERT_MIN_HITS": "abc"},
        {"NEURALGUARD_ALERT_MIN_HITS": "1.5"},
        {"NEURALGUARD_ALERT_MIN_HITS": "0"},
        {"NEURALGUARD_ALERT_CORROBORATION_SECONDS": "0"},
        {"NEURALGUARD_ALERT_CORROBORATION_SECONDS": "-3"},
        {"NEURALGUARD_ALERT_CORROBORATION_SECONDS": "nan"},
    ],
)
def test_invalid_alert_corroboration_env_raises(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)
