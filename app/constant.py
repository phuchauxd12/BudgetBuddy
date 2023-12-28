# -*- coding: utf-8 -*-
#from enum import StrEnum
from pydantic import BaseModel

import os
from dotenv import load_dotenv
import os

# Load environment variables from the local.env file
load_dotenv(os.path.join(os.path.dirname(__file__), "local.env"))

class Constant():
    MONGODB_URI = os.environ["MONGODB_URI"]


class Message(BaseModel):
    message: str
