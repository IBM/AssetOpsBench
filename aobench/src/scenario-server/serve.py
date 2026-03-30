from os import environ

import uvicorn
from dotenv import load_dotenv
from litestar import Litestar
from scenario_server.app import get_app

load_dotenv()


def setup() -> Litestar:
    mlflow_tracking_uri: str = environ.get("MLFLOW_TRACKING_URI", "")

    app: Litestar = get_app(
        include_default_handlers=True,
        tracking_uri=mlflow_tracking_uri,
    )
    return app


app: Litestar = setup()

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8099, reload=True)
