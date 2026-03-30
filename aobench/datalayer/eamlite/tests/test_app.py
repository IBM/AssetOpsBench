import os
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from os import environ, getenv, remove
from random import sample
from string import ascii_letters

from eamlite.eam_models import Workorders
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


class TestApp(unittest.TestCase):

    def setUp(self):
        """setUp Runs for each test

        The app needs a database url at start-up.
        This test uses an sqlite database in a temporary file
        """

        self.ENV_NAME = "DATABASE_URL"

        try:
            self.old_url = environ[self.ENV_NAME]
        except KeyError as e:
            self.old_url = ""

        ""

        random_letters = "".join(sample(ascii_letters, 6))

        self.dbfile = f"eamlite-{random_letters}.db"
        self.con = sqlite3.connect(self.dbfile)

        self.new_url = f"sqlite+pysqlite:///{self.dbfile}"
        environ[self.ENV_NAME] = self.new_url
        # import after setting env
        from eamlite.main import app

        self.client = TestClient(app)

    def test_workorder_fetch(self):

        # Create a couple of sample work orders
        start1 = datetime(2025, 8, 15, 10, 35, tzinfo=timezone.utc)
        end1 = start1 + timedelta(hours=3)
        wo1 = Workorders(
            workoderid=123,
            workordernum=1,
            description="test work order",
            startdate=start1,
            enddate=end1,
            priority=5,
        )

        start2 = datetime(2025, 9, 1, 8, 35, tzinfo=timezone.utc)
        end2 = start2 + timedelta(hours=4)
        wo2 = Workorders(
            workoderid=456,
            workordernum=2,
            description="second test work order",
            startdate=start2,
            enddate=end2,
            priority=1,
        )

        # Store these in a db
        engine = create_engine(self.new_url, echo=True)

        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            session.add(wo1)
            session.add(wo2)
            session.commit()

        # use the app to retrieve
        resp = self.client.get("/workorders?foo=bar")
        self.assertEqual(resp.status_code, 422)

        # # use the app to retrieve
        resp1 = self.client.get("/workorders")
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(len(resp1.json()), 2, "should return 2 work orders")

        # filter on the number
        resp2 = self.client.get("/workorders?workordernum=2")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(len(resp2.json()), 1, "should return 1 work order")

        # filter on the startdate
        resp3 = self.client.get("/workorders?startdate=2025-09-01 08:35:00")
        self.assertEqual(resp3.status_code, 200)
        self.assertEqual(len(resp3.json()), 1, "should return 1 work order")

        # filter on the priority with comparison
        resp4 = self.client.get("/workorders?priority=[gte]3")
        self.assertEqual(resp4.status_code, 200)
        self.assertEqual(len(resp4.json()), 1, "should return 1 work order")

        # filter on the priority with comparison
        resp4 = self.client.get("/workorders?priority=[gte]1")
        self.assertEqual(resp4.status_code, 200)
        self.assertEqual(len(resp4.json()), 2, "should return 2 work order")

        # # use the app to retrieve
        resp5 = self.client.get("/workorders?limit=1")
        self.assertEqual(resp5.status_code, 200)
        self.assertEqual(len(resp5.json()), 1, "should return 1 work orders")

    def tearDown(self):
        """Cleans up"""

        environ[self.ENV_NAME] = self.old_url

        self.con.close()

        # remove temp db file
        remove(self.dbfile)
