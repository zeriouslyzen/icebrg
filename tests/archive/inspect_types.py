import sqlite3
import json
import os
from collections import Counter, defaultdict

db_path = os.path.expanduser("~/Documents/iceburg_matrix/matrix.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Count Entity Types
print("--- Entity Types ---")
cursor.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC LIMIT 20")
for row in cursor:
    print(f"{row[0]}: {row[1]}")

# Sample keys for top types
target_types = ['Person', 'Company', 'Organization', 'Occupancy', 'Directorship', 'Ownership', 'Family', 'Association', 'Membership', 'Sanction']
print("\n--- Keys per Type ---")

for t in target_types:
    cursor.execute("SELECT properties FROM entities WHERE entity_type = ? LIMIT 10", (t,))
    keys = Counter()
    count = 0
    for row in cursor:
        count += 1
        try:
            props = json.loads(row[0])
            keys.update(props.keys())
        except: pass
    
    if count > 0:
        common_keys = [k for k, v in keys.most_common(5)]
        print(f"{t}: {common_keys}")
