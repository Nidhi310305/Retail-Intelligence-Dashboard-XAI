from utils.column_mapping import infer_column_mapping
import traceback

cols = ['Order Date','Ship Date','Customer ID','Sales','Profit','Discount','Quantity','Region','Category']
print("Calling infer_column_mapping with sample columns:", cols)
try:
    res = infer_column_mapping(cols)
    print("Result:", res)
except Exception as e:
    print("Exception during infer_column_mapping:")
    traceback.print_exc()
