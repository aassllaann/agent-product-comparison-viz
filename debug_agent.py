from electronics_agents import CameraAgent, PhoneAgent
import streamlit as st
from multi_agent import MultiCategoryAgent

# Mock session state for testing if needed, though we run this as standalone script
if 'agent' not in st.session_state:
    st.session_state.agent = MultiCategoryAgent()

def test_agent():
    print("Testing Agent Logic...")
    agent = MultiCategoryAgent()
    query = "推荐一款 5000 元左右的适合 Vlog 的相机"
    print(f"Query: {query}")
    
    try:
        reply, charts, results, analyses, eco = agent.handle_chat(query, history=[])
        print("Reply:", reply)
        print("Results count:", len(results) if results else 0)
        print("Charts count:", len(charts) if charts else 0)
        print("Analyses count:", len(analyses) if analyses else 0)
        
        if results:
            print("First Result:", results[0].Model)
        
        if charts:
             print("Charts generated successfully.")
             
    except Exception as e:
        print("ERROR during handle_chat:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
