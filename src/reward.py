import re

def parse_answer(generated_answer: str) -> str:
    match = re.search(r'\\boxed{', generated_answer)
    if not match:
        return generated_answer.strip()
    
    start_idx = match.end()
    brace_count = 1
    
    for i, char in enumerate(generated_answer[start_idx:]):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        
        if brace_count == 0:
            return generated_answer[start_idx : start_idx + i].strip()
            
    return generated_answer[start_idx:].strip()

def get_reward(parsed_answer: str, ground_truth: str) -> float:
    clean_parsed = parsed_answer.replace(" ", "").strip()
    clean_truth = ground_truth.replace(" ", "").strip()
    
    prefixes_to_strip = ["x=", "y=", "n=", "k="]
    for prefix in prefixes_to_strip:
        if clean_parsed.startswith(prefix):
            clean_parsed = clean_parsed[len(prefix):]
            
    if clean_parsed == clean_truth:
        return 1.0
    else:
        return 0.0

def batch_reward_function(completions, ground_truths, **kwargs) -> list[float]:
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        text = completion[0]['content'] if isinstance(completion, list) else completion
        parsed = parse_answer(text)
        rewards.append(get_reward(parsed, truth))
        
    return rewards

if __name__ == "__main__":
    print(parse_answer(r"\boxed{\frac{1}{2}}"))    