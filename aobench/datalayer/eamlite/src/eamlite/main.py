"""
To run this within the docker container, use the following commands on the terminal to minimize configurations:
    cd aobench/datalayer/eamlite && docker run --name eampg -e POSTGRES_USER=eamlite -e POSTGRES_PASSWORD=eamlite -e POSTGRES_DB=eamlite -p 5431:5432 -d postgres:16
   uv sync &&  DATABASE_URL=postgresql+psycopg://eamlite:eamlite@localhost:5431/eamlite uv run uvicorn eamlite.main:app --reload
"""

import inspect
from contextlib import asynccontextmanager

import eamlite.eam_models as models
from eamlite.crud_generator import create_crud_router
from eamlite.database import init_db
from fastapi import FastAPI
from sqlmodel import SQLModel


# refactored deprecated code
# define a lifespan function
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize DB Schema before serving request
    """
    init_db()
    yield


app = FastAPI(title="EAM API", version="1.0.0", lifespan=lifespan)

# Dynamically add routers for all SQLModel models
for name, cls in inspect.getmembers(models, inspect.isclass):
    if issubclass(cls, SQLModel) and getattr(cls, "__table__", None) is not None:
        router = create_crud_router(cls)
        app.include_router(
            router, prefix=f"/{cls.__tablename__.lower()}", tags=[cls.__name__]
        )
