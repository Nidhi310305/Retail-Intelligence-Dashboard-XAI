import os
import time
from pathlib import Path

# Ensure GEMINI key is unset for this test
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

from utils.data_loader import load_sample_dataset
from utils.column_mapping import infer_column_mapping
from utils.llm import generate_insights
from utils.rag import build_dashboard_knowledge_base, answer_question

print('GEMINI_API_KEY:', os.getenv('GEMINI_API_KEY'))
print('Starting timeout tests...')

# Load sample dataset
ds = load_sample_dataset(Path(__file__).parent / "Sample - Superstore.csv")
cols = ds.columns.tolist()

start = time.time()
print('Running infer_column_mapping...')
res = infer_column_mapping(cols)
print('infer_column_mapping duration:', time.time() - start)
print('mapping unresolved_roles:', res.get('unresolved_roles'))

# Test generate_insights
kb = build_dashboard_knowledge_base(summary_chunks=["Total sales: 100000", "Total profit: 5000"]) 
context = "\n".join(kb)
question = "Summarize key points"
start = time.time()
print('Running generate_insights...')
ins = generate_insights(context, question)
print('generate_insights duration:', time.time() - start)
print('generate_insights result:', ins)

# Test answer_question
start = time.time()
print('Running answer_question...')
ans = answer_question('Why are there anomalies?', kb)
print('answer_question duration:', time.time() - start)
print('answer_question result:', ans)
