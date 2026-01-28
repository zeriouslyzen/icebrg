import sqlite3
import json
import os
from collections import Counter

db_path = os.path.expanduser("~/Documents/iceburg_matrix/matrix.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

keys_counter = Counter()
cursor.execute("SELECT properties FROM entities LIMIT 10000")

for row in cursor:
    try:
        props = json.loads(row[0])
        keys_counter.update(props.keys())
    except:
        pass

print("Top 50 Property Keys:")
for k, v in keys_counter.most_common(50):
    print(f"{k}: {v}")
