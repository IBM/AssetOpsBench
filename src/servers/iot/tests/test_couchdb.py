"""Infrastructure tests: verify CouchDB connectivity and sample data."""

import os

import pytest
import couchdb3
import requests
from dotenv import load_dotenv

from .conftest import requires_couchdb

load_dotenv()

COUCHDB_URL = os.environ.get("COUCHDB_URL", "")
COUCHDB_HOST = COUCHDB_URL.replace("http://", "").replace("https://", "")
COUCHDB_USER = os.environ.get("COUCHDB_USERNAME", "")
COUCHDB_PASS = os.environ.get("COUCHDB_PASSWORD", "")
COUCHDB_DBNAME = os.environ.get("COUCHDB_DBNAME", "")

FULL_URL = f"http://{COUCHDB_USER}:{COUCHDB_PASS}@{COUCHDB_HOST}"


@pytest.fixture
def couchdb_client():
    return couchdb3.Server(FULL_URL)


@requires_couchdb
class TestCouchDBInfrastructure:
    def test_connection(self):
        resp = requests.get(f"http://{COUCHDB_HOST}", auth=(COUCHDB_USER, COUCHDB_PASS))
        assert resp.status_code == 200

        client = couchdb3.Server(FULL_URL)
        assert client.info() is not None

    def test_database_exists(self, couchdb_client):
        assert COUCHDB_DBNAME in couchdb_client.all_dbs()

    def test_sample_data_populated(self, couchdb_client):
        db = couchdb_client[COUCHDB_DBNAME]
        res = db.find({"asset_id": "Chiller 6"}, limit=1)
        assert len(res["docs"]) > 0
        assert res["docs"][0]["asset_id"] == "Chiller 6"
