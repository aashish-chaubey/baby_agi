"""
Main App
@author: Ashish Chaudhary
@date: 17-06-2023
"""

import os
from typing import Optional

from flask import Flask, render_template
from langchain import OpenAI

from utils.baby_agi.agi import BabyAGI
from utils.connect_vector_store import vectorstore

os.environ["OPENAI_API_KEY"] = "sk-xxx"  # https://platform.openai.com (Thx Michael from Twitter)
os.environ['SERPAPI_API_KEY'] = 'b03xxx' # https://serpapi.com/

app = Flask(__name__)

OBJECTIVE = "Be an IPL Cricket Reporter of the most recent match"
llm = OpenAI(temperature=0)

# Logging of LLMChains
VERBOSE = False
# If None, will keep on going forever
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    verbose=VERBOSE,
    max_iterations=max_iterations
)

@app.route("/")
def hello_world():
    """
    Test Function
    """
    response = baby_agi({"objective": OBJECTIVE})
    return render_template("index.html", title="Index", result=response)
