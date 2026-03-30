import datetime
import decimal
import re
from typing import Any
from typing import List
from typing import List as TList
from typing import Optional, Tuple, Type

from eamlite.database import get_session
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, create_model
from sqlalchemy import and_
from sqlalchemy.sql.sqltypes import (Boolean, Date, DateTime, Float, Integer,
                                     Numeric, String)
from sqlmodel import Session, SQLModel, select

OPS = {"eq", "gt", "gte", "lt", "lte"}


def parse_iso_datetime(s: str) -> datetime.datetime:
    s = s.strip()
    # accept Z as UTC
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Python's fromisoformat supports offsets like +00:00
    return datetime.datetime.fromisoformat(s)


def parse_filter_value(raw: str, py_type: type):
    """Parse '[gte]value' (or 'value') and convert to py_type.
    Returns (op, typed_value) or raises HTTPException(422) on error.
    """
    m = re.match(r"^\[(\w+)\](.*)$", raw)
    if m:
        op, val_str = m.group(1), m.group(2)
    else:
        op, val_str = "eq", raw

    if op not in OPS:
        raise HTTPException(status_code=422, detail=f"Invalid operator '{op}'")

    try:
        if py_type is int:
            val = int(val_str)
        elif py_type is float:
            val = float(val_str)
        elif py_type is bool:
            low = val_str.lower()
            if low in ("true", "1", "t", "yes", "y"):
                val = True
            elif low in ("false", "0", "f", "no", "n"):
                val = False
            else:
                raise ValueError("invalid boolean")
        elif py_type is decimal.Decimal:
            val = decimal.Decimal(val_str)
        elif py_type is datetime.date:
            # If time included, parse and use .date()
            if "T" in val_str or "Z" in val_str or "+" in val_str:
                dt = parse_iso_datetime(val_str)
                val = dt.date()
            else:
                val = datetime.date.fromisoformat(val_str)
        elif py_type is datetime.datetime:
            val = parse_iso_datetime(val_str)
        else:
            # fallback to string
            val = val_str
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid value for {py_type.__name__}: {val_str}",
        )

    return op, val


def sqlalchemy_type_to_python(sa_type):
    """Map SQLAlchemy column types to Python types for Pydantic."""
    if isinstance(sa_type, Integer):
        return int
    if isinstance(sa_type, String):
        return str
    if isinstance(sa_type, Float):
        return float
    if isinstance(sa_type, Boolean):
        return bool
    if isinstance(sa_type, Numeric):
        return decimal.Decimal
    if isinstance(sa_type, Date):
        return datetime.date
    if isinstance(sa_type, DateTime):
        return datetime.datetime
    return str  # fallback to string if unknown


def generate_filter_model(model: type[SQLModel]):
    fields = {}
    for column in model.__table__.columns:
        fields[column.name] = (
            Optional[str],
            Field(
                default=None,
                description=f"Filter by {column.name} using e.g. [gte]2023-01-01",
            ),
        )

    FilterModel = create_model(
        f"{model.__name__}Filter",
        __base__=BaseModel,
        **fields,
    )
    return FilterModel


def build_filters(filters: BaseModel, model: type[SQLModel]):
    """Return an SQLAlchemy condition (and_(...)) or None if no filters."""
    conditions = []

    for field_name, raw_values in filters.model_dump(exclude_none=True).items():
        column = getattr(model, field_name)
        col_info = model.__table__.columns[field_name]
        py_type = sqlalchemy_type_to_python(col_info.type)

        # raw_values is a list because we will declare fields as List[str]
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for rv in values:
            op, value = parse_filter_value(rv, py_type)

            # If column is datetime but value is date, you may want to
            # normalize (optional). Example below converts aware -> naive UTC:
            if isinstance(value, datetime.datetime):
                # Optionally: convert aware -> naive UTC to match DB storage
                if value.tzinfo is not None:
                    value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)

            if op == "eq":
                conditions.append(column == value)
            elif op == "gt":
                conditions.append(column > value)
            elif op == "gte":
                conditions.append(column >= value)
            elif op == "lt":
                conditions.append(column < value)
            elif op == "lte":
                conditions.append(column <= value)

    return and_(*conditions) if conditions else None


def create_crud_router(model: SQLModel) -> APIRouter:
    router = APIRouter()
    model_name = model.__name__

    FilterModel = generate_filter_model(model)

    def forbid_extra_params(request: Request, filters: FilterModel = Depends()):
        raw_params = request.query_params
        allowed = set(filters.model_fields.keys())
        allowed.add("limit")
        allowed.add("offset")
        for key in raw_params.keys():
            if key not in allowed:
                raise HTTPException(
                    status_code=422, detail=f"Unknown query parameter: {key}"
                )

        return filters

    # Create
    # NOTE : replaced item : model with Type[SQLModel] since variables are not allowed type expression error warning kept popping up
    @router.post("/", response_model=model)
    def create(item: Type[SQLModel], session: Session = Depends(get_session)):
        session.add(item)
        session.commit()
        session.refresh(item)
        return item

    @router.get("/", response_model=List[model])
    def read_all(
        filters: FilterModel = Depends(forbid_extra_params),  # type: ignore
        session: Session = Depends(get_session),
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        """Each query filter accepts either a value e.g., `param1=value` or a comparison operator e.g., `param1=[gt]val`
        Supported operators are: [eq] or no operator; [gt]; [gte]; [lt] and [lte]
        """

        query = select(model)
        condition = build_filters(filters, model)

        if condition is not None:
            query = query.where(condition)

        query = query.limit(limit).offset(offset)
        return session.exec(query).all()

    # Read one
    @router.get("/{item_id}", response_model=model)
    def read_one(item_id: int, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")
        return db_item

    # Update
    @router.put("/{item_id}", response_model=model)
    def update(item_id: int, new_item: model, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")

        for field, value in new_item.dict(exclude_unset=True).items():
            setattr(db_item, field, value)

        session.add(db_item)
        session.commit()
        session.refresh(db_item)
        return db_item

    # Delete
    @router.delete("/{item_id}")
    def delete(item_id: int, session: Session = Depends(get_session)):
        db_item = session.get(model, item_id)
        if not db_item:
            raise HTTPException(status_code=404, detail=f"{model_name} not found")
        session.delete(db_item)
        session.commit()
        return {"ok": True}

    return router
