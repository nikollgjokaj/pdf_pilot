
"""
HandoutAssistant.py

This file contains the HandoutAssistant class which is designed to process PDF handouts and answer questions based on their content. 

The HandoutAssistant class is responsible for:

1. Converting PDF to text.
2. Segmenting the text using AI21 Studio's API.
3. Loading the segmented questions.
4. Identifying relevant segments based on user questions using Hugging Face's Transformers.
5. Generating a prompt for OpenAI's GPT-3.
6. Extracting answers and segment IDs from GPT-3's response.

The file also contains a PDFHandler class that is responsible for highlighting relevant segments in the input PDF.
"""


import os
import fitz
import json
import requests
from transformers import pipeline
import openai
import re


class NLP:
    def __init__(self):
        self.pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def get_response(self, user_question, context):
        return self.pipeline(question=user_question, context=context)


class PDFHandler:
    @staticmethod
    def pdf_to_text(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text

    @staticmethod
    def highlight_text(input_pdf, output_pdf, text_to_highlight):
        phrases = text_to_highlight.split('\n')
        with fitz.open(input_pdf) as doc:
            for page in doc:
                for phrase in phrases:
                    areas = page.search_for(phrase)
                    if areas:
                        for area in areas:
                            highlight = page.add_highlight_annot(area)
                            highlight.update()
            doc.save(output_pdf)


class AI21Segmentation:
    @staticmethod
    def segment_text(text):
        url = "https://api.ai21.com/studio/v1/segmentation"
        payload = {
            "sourceType": "TEXT",
            "source": text
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer <your_21AI_key>"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            json_response = response.json()
            return json_response.get("segments")
        else:
            print(f"An error occurred: {response.status_code}")
            return None


class OpenAIAPI:
    def __init__(self):
        openai.api_key = "<Here your API key>"

    def get_answer_and_id(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer_text = response.choices[0].text.strip()
        lines = response.choices[0].text.strip().split('\n')
        answer = lines[0].strip()
        try:
            segment_id = int(re.search(r'<ID: (\d+)>', answer).group(1))
            answer = re.sub(r'<ID: \d+>', '', answer).strip()
        except AttributeError:
            segment_id = None
        return answer, segment_id


class HandoutAssistant:
    def __init__(self):
        self.nlp = NLP()
        self.openai_api = OpenAIAPI()

    def load_handout_questions(self, segmented_text):
        for idx, question_data in enumerate(segmented_text):
            question_data["id"] = idx + 1
        return segmented_text

    def get_relevant_segments(self, questions_data, user_question):
        relevant_segments = []
        for question_data in questions_data:
            context = question_data["segmentText"]
            response = self.nlp.get_response(user_question, context)
            if response["score"] > 0.5:
                relevant_segments.append({
                    "id": question_data["id"],
                    "segment_text": context,
                    "score": response["score"]
                })
        relevant_segments.sort(key=lambda x: x["score"], reverse=True)
        return relevant_segments[:10]

    def generate_prompt(self, question, relevant_segments):
        prompt = f"""
You are an AI Q&A bot. You will be given a question and a list of relevant text segments with their IDs. Please provide an accurate and concise answer based on the information provided, or indicate if you cannot answer the question with the given information. Also, please include the ID of the segment that helped you the most in your answer by writing <ID: > followed by the ID number.

Question: {question}

Relevant Segments:"""
        for segment in relevant_segments:
            prompt += f'\n{segment["id"]}. "{segment["segment_text"]}"'
        return prompt

    def process_pdf_and_get_answer(self, pdf_path, question):
        text = PDFHandler.pdf_to_text(pdf_path)
        segmented_text = AI21Segmentation.segment_text(text)
        questions_data = self.load_handout_questions(segmented_text)

        relevant_segments = self.get_relevant_segments(questions_data, question)

        if relevant_segments:
            prompt = self.generate_prompt(question, relevant_segments)
            openai_answer, segment_id = self.openai_api.get_answer_and_id(prompt)

            segment_text = next((seg["segment_text"] for seg in relevant_segments if seg["id"] == segment_id), None)
            
            return openai_answer, segment_id, segment_text
        else:
            return None, None, None


if __name__ == "__main__":
    assistant = HandoutAssistant()
    pdf_path = '/Users/eliaszobler/handout.pdf'

    while True:
        user_question = input("\nUser: ").strip()
        if user_question.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        if not user_question:
            print("Please enter a valid question.")
            continue

        answer, segment_id, segment_text = assistant.process_pdf_and_get_answer(pdf_path, user_question)

        if answer and segment_id:
            print(f"Answer: {answer}")
            print(f"Segment ID: {segment_id}")
        else:
            print("Sorry, I couldn't find an answer to your question in the handout.")

