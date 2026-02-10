from pydantic import BaseModel, RootModel, field_validator
from typing import Optional


class SimpleEvidenceDict(BaseModel):
    diagnosis_code: str
    text_evidence: str


class EvidenceDict(BaseModel):
    diagnosis_code: str
    text_evidence: str
    deduction: Optional[str] = None
    @field_validator("deduction", mode="before")
    def coerce_bool_to_str(cls, v):
        # If the value is a bool, convert to string
        if isinstance(v, bool):
            return str(v).lower()  # "true" or "false"
        return v    
    similarity: Optional[str] = None
    expanded_text_span: Optional[str] = None
    retrieved_document: Optional[str] = None
    retrieval_source: Optional[str] = None
    diagnosis_LD: Optional[str] = None



class EvidenceDictsList(RootModel[list[EvidenceDict]]):
    pass


class DetailedEvidence(BaseModel):
    diagnosis_code: Optional[str] = None
    diagnosis: Optional[str] = None
    text_evidence: Optional[str] = None            
    similarity: Optional[str] = None
    deduction: Optional[str] = None
    @field_validator("deduction", mode="before")
    def coerce_bool_to_str(cls, v):
        # If the value is a bool, convert to string
        if isinstance(v, bool):
            return str(v).lower()  # "true" or "false"
        return v
    expanded_text_span: Optional[str] = None
    retrieved_document: Optional[str] = None
    retrieval_source: Optional[str] = None
    diagnosis_LD: Optional[str] = None
    alternative_similar_diagnosis: Optional[str] = None



class DetailedEvidencesList(RootModel[list[DetailedEvidence]]):
    pass








def enrich_detailed_evidence(self, detailed_evidence: DetailedEvidence, k_retrieval: dict) -> DetailedEvidence:

    detailed_evidence.expanded_text_span = k_retrieval['expanded_text_span']
    detailed_evidence.retrieved_document = k_retrieval['retrieved_text_span']
    detailed_evidence.retrieval_source   = k_retrieval['retrieval_source']
    detailed_evidence.diagnosis_LD   = k_retrieval['Long Description']
    
    return detailed_evidence