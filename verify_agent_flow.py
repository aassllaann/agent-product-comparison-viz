from main_agent import CameraAgent
import sys

def test_agent():
    print("Initializing CameraAgent...")
    try:
        agent = CameraAgent()
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    print("Running handle_chat...")
    # Mock return from _get_individual_reasons to avoid API call if possible, 
    # but here we want to test the full flow including the parts we modified.
    # The modified part is step 4 and 5 (chart generation and analysis).
    # We rely on the fact that _parse_intent works (tested before).
    
    try:
        reply, charts, results, chart_analyses = agent.handle_chat("推荐一款5000元的相机")
        
        print("Reply generated.")
        print(f"Charts count: {len(charts)}")
        print(f"Analyses count: {len(chart_analyses)}")
        
        if len(charts) != 3:
            raise ValueError(f"Expected 3 charts, got {len(charts)}")
        
        # Check if chart_analyses[2] contains expected text
        print(f"Analysis 3: {chart_analyses[2]}")
        if "综合多维能力分析" not in chart_analyses[2]:
            raise ValueError("Analysis 3 text does not match expected format")
            
        print("Agent flow verification passed!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    test_agent()
