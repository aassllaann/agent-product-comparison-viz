import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from main_agent import CameraAgent, Camera
from models import SessionLocal

def test_vlog_recommendation():
    print("--- Test 1: Vlog Recommendation (Budget 8000) ---")
    agent = CameraAgent()
    
    # Simulate user intent for Vlog
    user_msg = "Recommended a camera for vlogging, budget around 8000"
    
    # Mocking _parse_intent to avoid API call cost/determinism issues for this test
    # But for a real test we might want to test the full flow. 
    # Here we test the logic after intent parsing.
    
    intent = {
        "summary": "recommend camera for vlogging",
        "usage": "vlog",
        "max_price": 8000,
        "sort_field": "Video_Score",
        "weight_pref": ""
    }
    
    print(f"Intent: {intent}")
    
    # Directly test logic parts
    candidates = agent._get_preset_cameras('vlog')
    print(f"Preset Candidates: {[c.Model for c in candidates]}")
    
    filtered = agent._filter_candidates(candidates, intent)
    print(f"Filtered Candidates: {[c.Model for c in filtered]}")
    
    assert any("ZV-E10" in c.Model for c in filtered), "Expected ZV-E10 in recommendations"
    print("Test 1 Passed!")

def test_fallback_logic():
    print("\n--- Test 2: Fallback Logic (Low Budget) ---")
    agent = CameraAgent()
    
    intent = {
        "summary": "cheap camera",
        "usage": "",
        "max_price": 2000,
        "sort_field": "LowLight_Score",
        "weight_pref": ""
    }
    
    # Should find no preset fits really (assuming presets are all > 2000 or empty usage)
    candidates = agent._get_preset_cameras('vlog') # Force check vlog presets just in case
    filtered = agent._filter_candidates(candidates, intent)
    print(f"Filtered Presets (Should be empty/low): {[c.Model for c in filtered]}")
    
    if len(filtered) < 3:
        exclude_ids = [c.id for c in filtered]
        fallback = agent._fallback_search(intent, exclude_ids)
        print(f"Fallback Results: {[c.Model for c in fallback]}")
        assert len(fallback) > 0, "Should have found something in fallback"
        assert all(c.Price <= 2000 for c in fallback), "Fallback matched price constraint"
    
    print("Test 2 Passed!")

def test_new_models():
    print("\n--- Test 3: New Models Verification ---")
    agent = CameraAgent()
    
    # 1. Test Finding "Action 4" by name/alias or broad search
    # We use fallback search logic or just query DB directly through agent methods if possible, 
    # but let's try to simulate a user asking for it.
    intent = {
        "summary": "Action 4",
        "usage": "vlog",
        "max_price": 3000,
        "sort_field": "Video_Score", 
        "weight_pref": ""
    }
    
    # Force the agent to look for Action 4
    # Note: 'Action' is a keyword in 'vlog' preset, so it should be found there.
    candidates = agent._get_preset_cameras('vlog')
    print(f"Vlog Candidates: {[c.Model for c in candidates]}")
    
    found_action = any("Action 4" in c.Model or "Action 4" in str(c.Alias) for c in candidates)
    assert found_action, "Expected 'Osmo Action 4' in vlog candidates"
    
    # 2. Test Finding "Pocket 3"
    found_pocket = any("Pocket 3" in c.Model for c in candidates)
    assert found_pocket, "Expected 'Osmo Pocket 3' in vlog candidates"
    
    # 3. Test Price Check
    action_cam = next(c for c in candidates if "Action 4" in c.Model)
    print(f"Action 4 Price: {action_cam.Price}")
    assert 2000 <= action_cam.Price <= 2500, "Price for Action 4 should be around 2199"

    print("Test 3 Passed!")

if __name__ == "__main__":
    try:
        test_vlog_recommendation()
        test_fallback_logic()
        test_new_models()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
