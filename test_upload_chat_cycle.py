from utils.column_mapping import infer_column_mapping
from utils.data_loader import load_sample_dataset
from utils.rag import build_dashboard_knowledge_base, answer_question

import traceback

print("Loading sample dataset and running mapping...")
try:
    ds = load_sample_dataset(None)
except Exception as e:
    print("Failed to load sample dataset via load_sample_dataset(None); trying file path")
    from pathlib import Path
    ds = load_sample_dataset(Path(__file__).parent / "Sample - Superstore.csv")

cols = ds.columns.tolist()
print("Columns:", cols)

try:
    mapping_result = infer_column_mapping(cols)
    print("Mapping result:", mapping_result)
except Exception:
    print("Exception during infer_column_mapping:")
    traceback.print_exc()

# Build KB and run a sample question
kb = build_dashboard_knowledge_base(
    summary_chunks=["Total sales: 100000", "Total profit: 5000"],
)
print("Knowledge base chunks:", kb)

question = "Why were some transactions flagged as anomalies?"
try:
    answer = answer_question(question, kb)
    print("Answer:", answer)
except Exception:
    print("Exception during answer_question:")
    traceback.print_exc()
