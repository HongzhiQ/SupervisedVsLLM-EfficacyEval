import os
import openai

# your ChatGPT API key
openai.api_key = 'API-KEY'  # Please replace with your API key.

job_info = openai.FineTuningJob.create(training_file="your file id", model="gpt-3.5-turbo")
print(job_info)