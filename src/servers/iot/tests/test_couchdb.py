"""Infrastructure tests: verify CouchDB connectivity and sample registry data."""

import os

import pytest
import couchdb3
import requests
from dotenv import load_dotenv

from .conftest import requires_couchdb

load_dotenv()

COUCHDB_URL = os.environ.get("COUCHDB_URL", "")
COUCHDB_HOST = COUCHDB_URL.replace("http://", "").replace("https://", "")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "")
ASSET_DBNAME = os.environ.get("ASSET_DBNAME", "asset")

FULL_URL = f"http://{COUCHDB_USERNAME}:{COUCHDB_PASSWORD}@{COUCHDB_HOST}"


@pytest.fixture
def couchdb_client():
    return couchdb3.Server(FULL_URL)


@requires_couchdb
class TestCouchDBInfrastructure:
    def test_connection(self):
        resp = requests.get(
            f"http://{COUCHDB_HOST}", auth=(COUCHDB_USERNAME, COUCHDB_PASSWORD)
        )
        assert resp.status_code == 200

        client = couchdb3.Server(FULL_URL)
        assert client.info() is not None

    def test_database_exists(self, couchdb_client):
        assert ASSET_DBNAME in couchdb_client.all_dbs()

    def test_sample_data_populated(self, couchdb_client):
        db = couchdb_client[ASSET_DBNAME]
        res = db.find({"assetnum": "Chiller 6"}, limit=1)
        assert len(res["docs"]) > 0
        assert res["docs"][0]["assetnum"] == "Chiller 6"
