# Define the Response
from pydantic import BaseModel
from typing import List, Optional

class BoudingBox(BaseModel):
  x1: int
  y1: int
  x2: int
  y2: int

class ClassPred(BaseModel):
  id: int
  name: str
  # nid: str

class PredictionDetector(BaseModel):
  filename: str
  contenttype: str
  boudingboxes: List[BoudingBox] = []
  classes: List[ClassPred] = []
  confidences: List[float] = []
  image_with_bboxes: Optional[bytes] = None
  image_url: str = ""

class PredictionClassifier(BaseModel):
  filename: str
  contenttype: str
  classes: List = []
  iids: List = []
  confidences: List[float] = []
