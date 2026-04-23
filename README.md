This repo requires a running LLM instance to be called, see https://github.com/luimui/LLM_Diagnose_Extraction_Clinical_Notes/blob/27ab86481f9f0f3da36861c08f252b323623fabf/pipeline/llm/Ollama_Client.py#L12

Install git lfs to pull large files (Retrieval VectorDB and NER Model)

Clone the Repo locally,

This is tested in a Miniconda environment Python=3.9.25:

`conda create ClinicalNotesEnvironment python=3.9.25`
`conda activate ClinicalNotesEnvironment`
`pip install -r requirements.txt`
`streamlit run streamlit.py`
