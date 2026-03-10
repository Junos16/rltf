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

    async def get_critique(self, prompt: str, generated_answer: str) -> str:
        try:
            judge_prompt = f"""
            Please evaluate this student's attempt.
            Problem: {prompt}
            Student's Answer: {generated_answer}
            """

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction="""You are a strict but helpful math/logic teacher.You are an expert mathematical and logical evaluator.
                        Your job is to read a problem and a student's attempted solution.
                        If the student's final answer is correct and their reasoning is sound, output EXACTLY:
                        "Your previous answer is correct."
                        If the student's answer is incorrect or their reasoning is flawed, output a concise critique.
                        CRITICAL RULES:
                        1. DO NOT provide the final correct answer.
                        2. DO NOT rewrite the entire solution for them.
                        3. Identify the specific step where they made a mistake (e.g., arithmetic error, logical fallacy) and state what was wrong with that specific step.
                        4. Keep your critique under 3 sentences.
                    """,
                ),
                contents=judge_prompt,
            )

            return response.text

        except:
            print(
                f"Error: Failed to get critique from judge for prompt: {prompt}"
            )            
            return ""

    async def batch_get_critiques(self, prompts: list[str], generated_answers: list[str]) -> list[str]:
        critiques = await asyncio.gather(
            list(map(self.get_critique, prompts, generated_answers))
        )
        
        return critiques

if __name__ == "__main__":
    async def test_judge(question: str, answer: str):
        judge = RLTFJudge()
        print("Testing judge...")
        res = await judge.get_critique(question, answer)
        print(f"Critique: {res}")

    question = input("Enter the question:")
    answer = input("Enter the answer:")
    asyncio.run(test_judge(question, answer))
