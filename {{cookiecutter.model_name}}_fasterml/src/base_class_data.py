from pydantic import BaseModel
from yaml2pyclass import CodeGenerator

class BaseClass(BaseModel, CodeGenerator):
    pass