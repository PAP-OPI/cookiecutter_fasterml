import dataclasses
from pydantic import BaseModel
import yaml2pyclass



class Config(yaml2pyclass.CodeGenerator, BaseModel):
    pass