import openai
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# Create a prompt template to be used in the chain.

template = """

    You are a management assistant who writes meeting minutes. You always manage to capture the important points.

    Below you will find a transcript of a recorded meeting.

    This report needs to be clearly and concisely written in English. Please conclude with action points at the bottom. Also, provide suggestions for topics to discuss in the next meeting.

    Transcript = {transcript}

    Response in markdown:


    """

prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template,
    )

# Load env files
load_dotenv()
openai_api_key = os.environ.get('openai_api_key')

# Transcribe audio with Whisper API
audio_file_path = "Recording.m4a"
audio_file= open(audio_file_path, "rb")
transcript_raw = openai.Audio.transcribe("whisper-1", file=audio_file)
print(transcript_raw['text'])
# Create LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

# Create chain
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(template))

# Run chain
summary = chain(transcript_raw['text'])
print(summary)
