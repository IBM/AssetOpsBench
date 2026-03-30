import datetime
import decimal
from typing import Optional

from sqlalchemy import (Column, Date, DateTime, ForeignKeyConstraint, Identity,
                        Integer, Numeric, PrimaryKeyConstraint, String,
                        UniqueConstraint)
from sqlmodel import Field, Relationship, SQLModel


class Assetstatus(SQLModel, table=True):
    __table_args__ = (PrimaryKeyConstraint("statusid", name="assetstatus_pkey"),)

    statusid: int = Field(
        sa_column=Column(
            "statusid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    statusname: Optional[str] = Field(
        default=None, sa_column=Column("statusname", String(50))
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )


class Assettypes(SQLModel, table=True):
    __table_args__ = (PrimaryKeyConstraint("assettypeid", name="assettypes_pkey"),)

    assettypeid: int = Field(
        sa_column=Column(
            "assettypeid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    typename: Optional[str] = Field(
        default=None, sa_column=Column("typename", String(50))
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )


class Failurecodes(SQLModel, table=True):
    __table_args__ = (PrimaryKeyConstraint("failurecodeid", name="failurecodes_pkey"),)

    failurecodeid: int = Field(
        sa_column=Column(
            "failurecodeid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    code: Optional[str] = Field(default=None, sa_column=Column("code", String(50)))
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )

    assetfailurehistory: list["Assetfailurehistory"] = Relationship(
        back_populates="failurecodes"
    )


class Jobplans(SQLModel, table=True):
    __table_args__ = (
        PrimaryKeyConstraint("jobplanid", name="jobplans_pkey"),
        UniqueConstraint("plannum", name="jobplans_plannum_key"),
    )

    jobplanid: int = Field(
        sa_column=Column(
            "jobplanid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    plannum: str = Field(sa_column=Column("plannum", String(50), nullable=False))
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )
    estimatedhours: Optional[decimal.Decimal] = Field(
        default=None, sa_column=Column("estimatedhours", Numeric(5, 2))
    )


class Sites(SQLModel, table=True):
    __table_args__ = (
        PrimaryKeyConstraint("siteid", name="sites_pkey"),
        UniqueConstraint("sitenum", name="sites_sitenum_key"),
    )

    siteid: int = Field(
        sa_column=Column(
            "siteid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    sitenum: str = Field(sa_column=Column("sitenum", String(50), nullable=False))
    name: Optional[str] = Field(default=None, sa_column=Column("name", String(100)))
    address: Optional[str] = Field(
        default=None, sa_column=Column("address", String(255))
    )

    locations: list["Locations"] = Relationship(back_populates="sites")


class Workorderstatus(SQLModel, table=True):
    __table_args__ = (PrimaryKeyConstraint("statusid", name="workorderstatus_pkey"),)

    statusid: int = Field(
        sa_column=Column(
            "statusid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    statusname: Optional[str] = Field(
        default=None, sa_column=Column("statusname", String(50))
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )


class Locations(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["siteid"], ["sites.siteid"], name="locations_siteid_fkey"
        ),
        PrimaryKeyConstraint("locationid", name="locations_pkey"),
        UniqueConstraint("locationnum", name="locations_locationnum_key"),
    )

    locationid: int = Field(
        sa_column=Column(
            "locationid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    locationnum: str = Field(
        sa_column=Column("locationnum", String(50), nullable=False)
    )
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )
    siteid: Optional[int] = Field(default=None, sa_column=Column("siteid", Integer))
    locationtype: Optional[str] = Field(
        default=None, sa_column=Column("locationtype", String(50))
    )

    sites: Optional["Sites"] = Relationship(back_populates="locations")
    assets: list["Assets"] = Relationship(back_populates="locations")
    servicerequests: list["Servicerequests"] = Relationship(back_populates="locations")
    workorders: list["Workorders"] = Relationship(back_populates="locations")


class Assets(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["locationid"], ["locations.locationid"], name="assets_locationid_fkey"
        ),
        PrimaryKeyConstraint("assetid", name="assets_pkey"),
        UniqueConstraint("assetnum", name="assets_assetnum_key"),
    )

    assetid: int = Field(
        sa_column=Column(
            "assetid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    assetnum: str = Field(sa_column=Column("assetnum", String(50), nullable=False))
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )
    locationid: Optional[int] = Field(
        default=None, sa_column=Column("locationid", Integer)
    )
    status: Optional[str] = Field(default=None, sa_column=Column("status", String(50)))
    assettype: Optional[str] = Field(
        default=None, sa_column=Column("assettype", String(50))
    )
    manufacturer: Optional[str] = Field(
        default=None, sa_column=Column("manufacturer", String(100))
    )
    modelnum: Optional[str] = Field(
        default=None, sa_column=Column("modelnum", String(100))
    )
    serialnum: Optional[str] = Field(
        default=None, sa_column=Column("serialnum", String(100))
    )
    purchasedate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("purchasedate", Date)
    )
    warrantyenddate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("warrantyenddate", Date)
    )

    locations: Optional["Locations"] = Relationship(back_populates="assets")
    assetfailurehistory: list["Assetfailurehistory"] = Relationship(
        back_populates="assets"
    )
    assetmeters: list["Assetmeters"] = Relationship(back_populates="assets")
    maintenanceschedules: list["Maintenanceschedules"] = Relationship(
        back_populates="assets"
    )
    servicerequests: list["Servicerequests"] = Relationship(back_populates="assets")
    workorders: list["Workorders"] = Relationship(back_populates="assets")


class Assetfailurehistory(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["assetid"], ["assets.assetid"], name="assetfailurehistory_assetid_fkey"
        ),
        ForeignKeyConstraint(
            ["failurecodeid"],
            ["failurecodes.failurecodeid"],
            name="assetfailurehistory_failurecodeid_fkey",
        ),
        PrimaryKeyConstraint("failureid", name="assetfailurehistory_pkey"),
    )

    failureid: int = Field(
        sa_column=Column(
            "failureid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    assetid: Optional[int] = Field(default=None, sa_column=Column("assetid", Integer))
    failurecodeid: Optional[int] = Field(
        default=None, sa_column=Column("failurecodeid", Integer)
    )
    failuredate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("failuredate", Date)
    )
    resolution: Optional[str] = Field(
        default=None, sa_column=Column("resolution", String(255))
    )

    assets: Optional["Assets"] = Relationship(back_populates="assetfailurehistory")
    failurecodes: Optional["Failurecodes"] = Relationship(
        back_populates="assetfailurehistory"
    )


class Assetmeters(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["assetid"], ["assets.assetid"], name="assetmeters_assetid_fkey"
        ),
        PrimaryKeyConstraint("meterid", name="assetmeters_pkey"),
    )

    meterid: int = Field(
        sa_column=Column(
            "meterid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    assetid: Optional[int] = Field(default=None, sa_column=Column("assetid", Integer))
    meterreading: Optional[decimal.Decimal] = Field(
        default=None, sa_column=Column("meterreading", Numeric(10, 2))
    )
    readingdate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("readingdate", Date)
    )

    assets: Optional["Assets"] = Relationship(back_populates="assetmeters")


class Maintenanceschedules(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["assetid"], ["assets.assetid"], name="maintenanceschedules_assetid_fkey"
        ),
        PrimaryKeyConstraint("scheduleid", name="maintenanceschedules_pkey"),
    )

    scheduleid: int = Field(
        sa_column=Column(
            "scheduleid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    assetid: Optional[int] = Field(default=None, sa_column=Column("assetid", Integer))
    frequency: Optional[str] = Field(
        default=None, sa_column=Column("frequency", String(50))
    )
    lastservicedate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("lastservicedate", Date)
    )
    nextservicedate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("nextservicedate", Date)
    )

    assets: Optional["Assets"] = Relationship(back_populates="maintenanceschedules")
    preventivemaintenance: list["Preventivemaintenance"] = Relationship(
        back_populates="maintenanceschedules"
    )


class Servicerequests(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["assetid"], ["assets.assetid"], name="servicerequests_assetid_fkey"
        ),
        ForeignKeyConstraint(
            ["locationid"],
            ["locations.locationid"],
            name="servicerequests_locationid_fkey",
        ),
        PrimaryKeyConstraint("servicerequestid", name="servicerequests_pkey"),
        UniqueConstraint("requestnum", name="servicerequests_requestnum_key"),
    )

    servicerequestid: int = Field(
        sa_column=Column(
            "servicerequestid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    requestnum: str = Field(sa_column=Column("requestnum", String(50), nullable=False))
    assetid: Optional[int] = Field(default=None, sa_column=Column("assetid", Integer))
    locationid: Optional[int] = Field(
        default=None, sa_column=Column("locationid", Integer)
    )
    status: Optional[str] = Field(default=None, sa_column=Column("status", String(50)))
    priority: Optional[int] = Field(default=None, sa_column=Column("priority", Integer))
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )
    requestdate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("requestdate", Date)
    )

    assets: Optional["Assets"] = Relationship(back_populates="servicerequests")
    locations: Optional["Locations"] = Relationship(back_populates="servicerequests")


class Workorders(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["assetid"], ["assets.assetid"], name="workorders_assetid_fkey"
        ),
        ForeignKeyConstraint(
            ["locationid"], ["locations.locationid"], name="workorders_locationid_fkey"
        ),
        PrimaryKeyConstraint("workorderid", name="workorders_pkey"),
        UniqueConstraint("workordernum", name="workorders_workordernum_key"),
    )

    workorderid: int = Field(
        sa_column=Column(
            "workorderid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    workordernum: str = Field(
        sa_column=Column("workordernum", String(50), nullable=False)
    )
    assetid: Optional[int] = Field(default=None, sa_column=Column("assetid", Integer))
    locationid: Optional[int] = Field(
        default=None, sa_column=Column("locationid", Integer)
    )
    status: Optional[str] = Field(default=None, sa_column=Column("status", String(50)))
    type: Optional[str] = Field(default=None, sa_column=Column("type", String(255)))
    priority: Optional[int] = Field(default=None, sa_column=Column("priority", Integer))
    description: Optional[str] = Field(
        default=None, sa_column=Column("description", String(255))
    )
    startdate: Optional[datetime.datetime] = Field(
        default=None, sa_column=Column("startdate", DateTime)
    )
    enddate: Optional[datetime.datetime] = Field(
        default=None, sa_column=Column("enddate", DateTime)
    )

    assets: Optional["Assets"] = Relationship(back_populates="workorders")
    locations: Optional["Locations"] = Relationship(back_populates="workorders")
    itemissues: list["Itemissues"] = Relationship(back_populates="workorders")
    workorderlabor: list["Workorderlabor"] = Relationship(back_populates="workorders")


class Itemissues(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["workorderid"],
            ["workorders.workorderid"],
            name="itemissues_workorderid_fkey",
        ),
        PrimaryKeyConstraint("issueid", name="itemissues_pkey"),
    )

    issueid: int = Field(
        sa_column=Column(
            "issueid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    inventoryid: Optional[int] = Field(
        default=None, sa_column=Column("inventoryid", Integer)
    )
    workorderid: Optional[int] = Field(
        default=None, sa_column=Column("workorderid", Integer)
    )
    quantityissued: Optional[int] = Field(
        default=None, sa_column=Column("quantityissued", Integer)
    )
    issuedate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("issuedate", Date)
    )

    workorders: Optional["Workorders"] = Relationship(back_populates="itemissues")


class Preventivemaintenance(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["scheduleid"],
            ["maintenanceschedules.scheduleid"],
            name="preventivemaintenance_scheduleid_fkey",
        ),
        PrimaryKeyConstraint("pmid", name="preventivemaintenance_pkey"),
    )

    pmid: int = Field(
        sa_column=Column(
            "pmid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    scheduleid: Optional[int] = Field(
        default=None, sa_column=Column("scheduleid", Integer)
    )
    taskdescription: Optional[str] = Field(
        default=None, sa_column=Column("taskdescription", String(255))
    )
    estimatedhours: Optional[decimal.Decimal] = Field(
        default=None, sa_column=Column("estimatedhours", Numeric(5, 2))
    )

    maintenanceschedules: Optional["Maintenanceschedules"] = Relationship(
        back_populates="preventivemaintenance"
    )


class Workorderlabor(SQLModel, table=True):
    __table_args__ = (
        ForeignKeyConstraint(
            ["workorderid"],
            ["workorders.workorderid"],
            name="workorderlabor_workorderid_fkey",
        ),
        PrimaryKeyConstraint("laborid", name="workorderlabor_pkey"),
    )

    laborid: int = Field(
        sa_column=Column(
            "laborid",
            Integer,
            Identity(
                start=1,
                increment=1,
                minvalue=1,
                maxvalue=2147483647,
                cycle=False,
                cache=1,
            ),
            primary_key=True,
        )
    )
    workorderid: Optional[int] = Field(
        default=None, sa_column=Column("workorderid", Integer)
    )
    employeeid: Optional[int] = Field(
        default=None, sa_column=Column("employeeid", Integer)
    )
    hoursworked: Optional[decimal.Decimal] = Field(
        default=None, sa_column=Column("hoursworked", Numeric(5, 2))
    )
    workdate: Optional[datetime.date] = Field(
        default=None, sa_column=Column("workdate", Date)
    )

    workorders: Optional["Workorders"] = Relationship(back_populates="workorderlabor")
