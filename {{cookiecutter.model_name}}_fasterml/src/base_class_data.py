import dataclasses
from pydantic import BaseModel
import yaml2pyclass



class Config(BaseModel, yaml2pyclass.CodeGenerator):
    pass