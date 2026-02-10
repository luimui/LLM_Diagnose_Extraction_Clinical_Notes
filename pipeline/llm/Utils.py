from ..schemas.Evidence_Objects import DetailedEvidence

def enrich_detailed_evidence(detailed_evidence: DetailedEvidence, k_retrieval: dict) -> DetailedEvidence:
    """
    Convert SimilarityEvidenceDict to EvidenceDict.
    
    Args:
        sim_obj: instance of SimilarityEvidenceDict
        threshold: similarity threshold to set deduction True/False
    
    Returns:
        EvidenceDict
    """
    detailed_evidence.expanded_text_span = k_retrieval['expanded_text_span']
    detailed_evidence.retrieved_document = k_retrieval['retrieved_text_span']
    detailed_evidence.retrieval_source   = k_retrieval['retrieval_source']
    detailed_evidence.diagnosis_LD   = k_retrieval['Long Description']
    
    # map fields and return new object
    return detailed_evidence