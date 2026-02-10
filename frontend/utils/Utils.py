def flatten_result(r):
    return {
        "diagnosis_code": r.diagnosis_code,
        "diagnosis": r.diagnosis,
        "diagnosis_LD": r.diagnosis_LD,
        "text_evidence": r.text_evidence,
        "deduction": r.deduction,
        "expanded_text_span": r.expanded_text_span
    }


def color_row(row):
    if row["deduction"]:
        return ["background-color: #d4edda"] * len(row)  # green
    else:
        return ["background-color: #f8d7da"] * len(row)  # red