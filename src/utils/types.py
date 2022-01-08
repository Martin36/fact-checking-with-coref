from typing import List, TypedDict

class FeverDataSample(TypedDict):
  id: int
  claim: str
  label: str
  verifiable: str
  evidence: List[List[List[str]]]
  
class FeverDataSampleWithEvidenceText(FeverDataSample):
  evidence_text: List[List[str]]


class DocumentLevelFever(TypedDict):
  id: int
  claim: str
  page: str
  sentences: List[str]
  label_list: List[int]
  sentence_IDS: List[int]