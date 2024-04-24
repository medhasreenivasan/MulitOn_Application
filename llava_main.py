
import cv2
import os
import torch
from PIL import Image
import re
from multion.client import MultiOn
from typing import Union

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import uvicorn

import torch
from transformers import BitsAndBytesConfig,pipeline
from pyngrok import conf

multion = MultiOn(api_key="MULTION_API_KEY")

conf.get_default().auth_token = "NGROK_AUTH_TOKEN"

## quantization configuration

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})


async def complete_prompt(visual, input):
  PROMPT = f"""You are an AI that can understand images and make deep analyis base on
  the visual description of current scenario.
  You can place orders online. You can also help writing creative things, like jokes, and poems.

  Example:
  Human: Order this food from doordash
  Visual: An image of Kung Pao Chicken.
  AI: Sure, I can do it for you. [order kung pao chicken from doordash]

  Order food can be done with something like [order <item> from Amazon], one action square bracket.
   
  In your response, please include the command within square brackets.
  Generic example:
  Human: <the input task give>
  Visual: <caption the given image>
  AI: Sure, I can do it for you. [<generate this command based on the human input and the visual representation>]

  Based on the image uploaded and the example above answer the following:

  Human:{input}
  Visual: ...
  AI: ...

  """

  llava_prompt = f"USER: <image>\n{PROMPT}?\nASSISTANT:"

  outputs = pipe(visual, prompt=llava_prompt, generate_kwargs={"max_new_tokens": 200})

  return outputs[0]["generated_text"].split('ASSISTANT:')[1].strip().replace('\n', '')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/get_visual_prompt")
async def get_visual_prompt(image: UploadFile = File(...),user_input: str = Form(...)):
  # Read image from upload
  img = Image.open(image.file)
  img = img.convert("RGB")

  complete_visual_prompt = await complete_prompt(img, user_input)

  pattern = r'\[([^\[\]]+)\]'
  input_command = re.findall(pattern, complete_visual_prompt[1])

  if input_command:
    browse = multion.browse(cmd=input_command[0],local=True)
  else:
    browse = f"Input command not generated:{complete_visual_prompt}"

  return {"browse": browse}

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:',ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app,port=8000)
