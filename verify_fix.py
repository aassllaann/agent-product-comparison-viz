from electronics_agents import HeadphoneAgent
from models import Headphone

# Mock/Test
agent = HeadphoneAgent()
session = agent.get_db_session()

# Create dummy data if not exists (or rely on existing)
# We assume database has some data. If not, we might need to populate.
# Let's just check existing data.

print("--- Testing Type Filtering ---")

# Mock Intent
intent_over_ear = {
    "product_type": "头戴式",
    "usage": "",
    "max_price": 100000
}

intent_in_ear = {
    "product_type": "入耳式",
    "usage": "",
    "max_price": 100000
}

# Fetch candidates (simulate _get_preset_products or fallback)
candidates = session.query(Headphone).limit(10).all()
print(f"Total candidates: {len(candidates)}")
for c in candidates:
    print(f"  - {c.Brand} {c.Model} ({c.Type})")

print("\n[Test 1] Filtering for '头戴式'")
filtered_over = agent._filter_and_sort(candidates, intent_over_ear, Headphone)
for item in filtered_over:
    if "头戴式" not in item.Type:
        print(f"FAILED: Found non-over-ear item: {item.Type}")
    else:
        print(f"  Passed: {item.Model} is {item.Type}")

print("\n[Test 2] Filtering for '入耳式'")
filtered_in = agent._filter_and_sort(candidates, intent_in_ear, Headphone)
for item in filtered_in:
    if "入耳式" not in item.Type:
        print(f"FAILED: Found non-in-ear item: {item.Type}")
    else:
        print(f"  Passed: {item.Model} is {item.Type}")
        
print("\nDone.")
