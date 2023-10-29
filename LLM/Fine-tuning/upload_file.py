import openai

# your ChatGPT API key
openai.api_key = 'API-KEY'  # Please replace with your API key.

file_info = openai.File.create(
  file=open("../../data/cognitive distortion/cognitive_distortion_train.jsonl", "rb"),
  purpose='fine-tune'
)
print(file_info)

