'''
TODO: Add retry logic
TODO: Add rate limiting
'''

import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class RLTFJudge:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def get_critique(self, prompt: str, generated_solution: str) -> str:
        try:
            judge_prompt = f"""You are an expert grader for math/logic problems.
                Problem: {prompt}
                Student Solution: {generated_solution}
                Your task:
                • Analyze the student solution step by step.
                • Focus on correctness and logical consistency.
                • Identify potential mistake(s), if any.
                • Provide concrete, actionable hints to improve the solution.
                • Keep the Critique section under 200 words.
                Format your response exactly as:
                Thinking: [Your step-by-step analysis]
                Critique: [Your final critique in under 200 words, ending with either “Your previous attempt was
                correct.” or “Your previous attempt was incorrect.”]
            """

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=judge_prompt,
            )

            return response.text

        except Exception as e:
            print(
                e,
                f"Error: Failed to get critique from judge for prompt: {prompt}"
            )            
            return ""

    async def batch_get_critiques(self, prompts: list[str], solutions: list[str]) -> list[str]:
        critiques = await asyncio.gather(
            *[self.get_critique(p, s) for p, s in zip(prompts, solutions)]
        )
        return list(critiques)

if __name__ == "__main__":
    async def test_judge(question: str, solution: str):
        judge = RLTFJudge()
        print("Testing judge...")
        res = await judge.get_critique(question, solution)
        print(f"Critique: {res}")

    question = input("Enter the question:")
    solution = input("Enter the solution:")
    asyncio.run(test_judge(question, solution))
