import streamlit as st
from pipeline.pipeline import Pipeline
import pandas as pd
from frontend.utils.Utils import flatten_result, color_row
from config.config import Config


# ---------- Page setup ----------
st.set_page_config(
    page_title="Diagnose Extraction RAG LLM",
    layout="wide",
)

st.title("Diagnose Extraction RAG LLM")




# ---------- Pipeline  ----------
@st.cache_resource
def get_pipeline():
    """
    Create and cache the Pipeline.
    Runs once per session unless cache is cleared.
    """
    return Pipeline()

pipeline = get_pipeline()




# ----------NER+LLM-----------
st.subheader("Input Document")
with open('data/example_letter.txt', 'r', encoding='utf-8') as file:
            text = file.read()
            letter_input = st.text_area(
                                        "Enter your text here",
                                        value=text,  # default text
                                        height=300                  # controls how many lines are visible
                                        )
            

if st.button("Run LLM Pipeline"):
    k_icd_codes_from_expaned_text_spans_full, html_colored_text, evidence_dicts_list_full = pipeline.run(text=letter_input
                                                                                                        , no_trashs=Config.NO_TRASHS
                                                                                                        , ks=Config.KS
                                                                                                        , score_thresholds=Config.SCORE_THRESHOLDS
                                                                                                        , segmentations=Config.SEGMENTATIONS
                                                                                                        , window_sizes=Config.WINDOW_SIZES)

    


        
    st.subheader("Input Document: Natural Entity Recognition BIO scheme")
    st.markdown(
    """
    <h4>
        <span style="color: red;">B-PROBLEM </span>    <span style="color: purple;">I-PROBLEM </span>
    </h4>
    """,
    unsafe_allow_html=True
)
    st.markdown('<div style="background-color: #f5f5f5;">' + html_colored_text + '</div>', unsafe_allow_html=True)

    st.subheader("Diagnose Extraction Result")
    df = pd.DataFrame(k_icd_codes_from_expaned_text_spans_full)
    st.table(df)

    
    # df = pd.DataFrame([flatten_result(evidence) for evidence in evidence_dicts_list_full])
    # df["deduction"] = df["deduction"].str.lower().map({"true": True, "false": False})
    # df = df.sort_values(by="deduction", ascending=False)
    # df = df.style.apply(color_row, axis=1)
    # df = pd.DataFrame(evidence_dicts_list_full)
    # st.table(df)





if st.button("Reload pipeline"):
    get_pipeline.clear()
    st.rerun()




    