import sys
from main_agent import CameraAgent, Camera
from models import SessionLocal

def test_travel_budget_logic():
    print("--- Test: Travel Recommendation (Budget 10000) ---")
    agent = CameraAgent()
    
    # Simulate parsed intent from LLM
    # If user says "旅行", usage might be '旅行' or 'travel' depending on LLM. 
    # But now we handle Chinese keywords.
    intent = {
        "summary": "旅行拍照",
        "usage": "旅行",
        "max_price": 10000,
        "sort_field": "Portability_Score"
    }
    
    print(f"Intent: {intent}")
    
    # 1. Check Scenario Detection
    usage = intent.get('usage', '').lower()
    summary = intent.get('summary', '').lower()
    target_scenario = None
    
    for scenario, keywords in agent.SCENARIO_KEYWORDS.items():
        if any(k in usage for k in keywords) or any(k in summary for k in keywords):
            target_scenario = scenario
            break
            
    print(f"Detected Scenario: {target_scenario}")
    assert target_scenario == 'travel', f"Expected 'travel', got {target_scenario}"
    
    # 2. Get Presets
    candidates = agent._get_preset_cameras(target_scenario)
    print(f"Candidates count: {len(candidates)}")
    print("Candidates:", [f"{c.Brand} {c.Model} ({c.Price})" for c in candidates])
    
    # 3. Filter
    filtered = agent._filter_candidates(candidates, intent)
    print("Filtered Candidates:", [f"{c.Brand} {c.Model} ({c.Price})" for c in filtered])
    
    # Check X100V is NOT present (Price 11000 > 10000)
    x100v = next((c for c in filtered if "X100V" in c.Model), None)
    assert x100v is None, "X100V should be filtered out due to budget"
    
    # Check Leica Q3 is NOT present (Not in travel preset)
    leica = next((c for c in filtered if "Q3" in c.Model), None)
    assert leica is None, "Leica Q3 should not be in travel preset results"
    
    # Check we have valid results
    assert len(filtered) >= 3, "Should have enough results from preset"
    
    print("Test Logic Passed!")

if __name__ == "__main__":
    test_travel_budget_logic()
