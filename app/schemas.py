from typing import Literal

from pydantic import Field, model_validator

from app.core.http import StrictModel


class Dataset(StrictModel):
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    csv_text: str = Field(min_length=1, max_length=1000000)


class Column(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    dtype: Literal["string", "integer", "number", "date"] = "string"
    required: bool = True
    unique: bool = False
    minimum: float | None = Field(default=None, allow_inf_nan=False)
    maximum: float | None = Field(default=None, allow_inf_nan=False)
    allowed_values: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def boundaries(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")
        return self


class Contract(StrictModel):
    columns: list[Column] = Field(min_length=1, max_length=50)
    allow_extra_columns: bool = False

    @model_validator(mode="after")
    def names(self):
        if len({c.name for c in self.columns}) != len(self.columns):
            raise ValueError("Contract contains duplicate column names")
        return self


class Investigation(StrictModel):
    dataset_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    contract: Contract
    planner: Literal["rules", "ollama"] = "rules"


class ToolCall(StrictModel):
    name: Literal["inspect_schema", "profile_records", "retrieve_handbook"]
    dataset_id: str


class ToolPlan(StrictModel):
    calls: list[ToolCall] = Field(min_length=3, max_length=3)


class PolicyClaim(StrictModel):
    text: str
    quote: str
    chunk_id: str


class RepairAdvice(StrictModel):
    operations: list[Literal["trim_whitespace", "drop_exact_duplicates", "escape_formulas"]] = (
        Field(max_length=3)
    )
    claims: list[PolicyClaim] = Field(min_length=1, max_length=5)
    manual_issue_codes: list[str] = Field(max_length=30)
