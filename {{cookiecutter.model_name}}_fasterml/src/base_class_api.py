from base_class_data import Config
from pydantic import BaseModel


class BaseClass(BaseModel, Config):
    pass
