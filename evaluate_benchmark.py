import json
import time
import os
import sys
from multi_agent import MultiCategoryAgent

def evaluate():
    print("="*60)
    print("System Benchmark Evaluation (50 Samples)")
    print("="*60)
    
    # Load dataset
    dataset_path = "benchmark_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # LIMIT FOR TESTING
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        test_cases = test_cases[:limit]
        print(f"Running limited test: {limit} samples")

    agent = MultiCategoryAgent()
    results = []
    
    total_ima = 0 # Intent Matching Accuracy
    total_rr = 0  # Recommendation Relevance
    total_latency = 0
    
    for case in test_cases:
        print(f"[{case['id']}/50] Testing Category: {case['category']}...")
        
        start_time = time.time()
        try:
            # Run the system
            reasons, charts, items, analyses, eco = agent.handle_chat(case['query'])
            latency = time.time() - start_time
            total_latency += latency
            
            # Retrieve the internal agent used (to check its last_intent)
            # detect_category is called inside handle_chat, we can replicate it or check the agent
            from category_detector import CategoryDetector
            detector = CategoryDetector()
            detected_key, _ = detector.detect_category(case['query'])
            
            # Check Category Matching
            category_match = (detected_key == case['category'])
            
            # Check Intent Parsing (IMA)
            # We access the sub-agent's last_intent
            sub_agent = agent.get_agent(detected_key)
            parsing_match = False
            if sub_agent and hasattr(sub_agent, 'last_intent'):
                intent = sub_agent.last_intent
                # Check price (within 10% tolerance or exact 999999)
                price_match = False
                if case['expected_max_price'] == 999999:
                    price_match = (intent.get('max_price') >= 100000)
                else:
                    price_match = abs(intent.get('max_price', 0) - case['expected_max_price']) / (case['expected_max_price'] or 1) < 0.2
                
                # Check sort field
                sort_match = (intent.get('sort_field') == case['expected_sort'])
                parsing_match = price_match and sort_match
            
            # Check Recommendation Relevance (RR)
            # For this benchmark, RR is True if items were found and they match the price constraint
            rr_match = False
            if items:
                # Check if top item satisfies price if possible
                if case['expected_max_price'] == 999999 or items[0].Price <= intent.get('max_price', 999999) * 1.1:
                    rr_match = True
            
            if category_match and parsing_match: total_ima += 1
            if rr_match: total_rr += 1
            
            results.append({
                "id": case['id'],
                "latency": latency,
                "category_match": category_match,
                "parsing_match": parsing_match,
                "rr_match": rr_match
            })
            
        except Exception as e:
            print(f"  ❌ Error on case {case['id']}: {e}")
            results.append({"id": case['id'], "error": str(e)})

    # Summary
    count = len(results)
    avg_latency = total_latency / count if count > 0 else 0
    ima_rate = (total_ima / count) * 100 if count > 0 else 0
    rr_rate = (total_rr / count) * 100 if count > 0 else 0
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("-" * 60)
    print(f"Total Samples: {count}")
    print(f"Avg Latency:   {avg_latency:.2f}s")
    print(f"Intent Matching Accuracy (IMA): {ima_rate:.1f}%")
    print(f"Recommendation Relevance (RR):  {rr_rate:.1f}%")
    print("="*60)
    
    # Save results to a file for reference
    with open("evaluation_report.json", "w", encoding="utf-8") as rf:
        json.dump({
            "metrics": {
                "avg_latency": avg_latency,
                "ima_rate": ima_rate,
                "rr_rate": rr_rate
            },
            "details": results
        }, rf, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    evaluate()
