from typing import List, TypedDict, Union

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
  
  
class DocRetrievalResult(TypedDict):
  id: int
  verifiable: str # "VERIFIABLE" | "NOT VERIFIABLE"
  label: str
  claim: str
  evidence: List[List[Union[int, str]]]
  noun_phrases: List[str]
  predicted_pages: List[str]
  wiki_results: List[str]
  