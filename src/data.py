from datasets import load_dataset

def format_example(example):
    example["prompt"] = f"""Solve the following mathematical problem step by step.
Let's think step by step and output the final answer within \\boxed{{}}.

Problem: {example['problem']}"""
    return example

def load_data():
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        dataset = dataset.map(format_example)
        dataset = dataset.remove_columns(["subject", "level", "unique_id"])
        return dataset
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    dataset = load_data()
    print(dataset["test"][0])
