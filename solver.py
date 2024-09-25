import sqlite3 
import urllib
import base64
import os
import autogen
import pandas as pd
import yfinance as yf
import promptflow
import datetime
import time
from promptflow.tracing import trace as trace_greg, start_trace
import requests
import torch
from torch import nn
import openai
from openai import AzureOpenAI
import numpy as np

from PIL import Image
from io import BytesIO
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service as ChromeService 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager 

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from promptflow.tracing._integrations._openai_injector import inject_openai_api

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

#htmlunitdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import logging
import os
import time
from dotenv import load_dotenv
load_dotenv()


#lightgbm tool
def tool_train_lightgbm_model(filename,target_column):
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    import lightgbm as lgb

    print("Loading data...")
    # load or create your dataset
    #regression_example_dir = Path(__file__).absolute().parents[1] / "regression"
    df = pd.read_csv(filename, header=None, sep="\t")
    train, test = train_test_split(df, test_size=0.2)
    # split data into training and testing sets
    # y is the column to predict
    X_train = train.drop(target_column, axis=1)
    X_test =  test.drop(target_column, axis=1)
    y_train = train[target_column]
    y_test =  test[target_column]
    
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }

    print("Starting training...")
    # train
    gbm = lgb.train(
        params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )

    print("Saving model...")
    # save model to file
    gbm.save_model("model.txt")

    print("Starting predicting...")
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"The RMSE of prediction is: {rmse_test}")
    return gbm

#test the azure.ai.inference sdk
if False:
    import os
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage
    from azure.ai.inference.models import UserMessage
    from azure.core.credentials import AzureKeyCredential
    import time
    start = time.time()
    client = ChatCompletionsClient(
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.environ["GIT_TOKEN"]),
    )

    response = client.complete(
        messages=[
            SystemMessage(content=""""""),
            UserMessage(content="what is the capital of France?"),
        ],
        model="gpt-4o",
        temperature=1,
        max_tokens=4096,
        top_p=1
    )

    print(response.choices[0].message.content)
    end = time.time()
    print("time taken", end-start)

#define some OCR classes
if True:
    # CODE FOR OCR
    import requests
    import json
    from dataclasses import dataclass, asdict
    from typing import List

    @dataclass
    class Metadata:
        width: int
        height: int

    @dataclass
    class BoundingPolygon:
        x: int
        y: int

    @dataclass
    class Word:
        text: str
        boundingPolygon: List[BoundingPolygon]
        confidence: float  

    @dataclass
    class Line:
        text: str
        boundingPolygon: List[BoundingPolygon]
        words: List[Word]    

    @dataclass
    class Block:
        lines: List[Line]

    @dataclass
    class ReadResult:
        blocks: List[Block]

    @dataclass
    class AnalyzeResult:
        modelVersion: str
        metadata: Metadata
        readResult: ReadResult

    @dataclass
    class AnalyzeRequest:
        uri: str

    class OCR:
        def recognize_text(self):
            # For prod environment, please find the endpoint and resource key of your computer vision resource from Azure portal.
            endpoint = os.environ["VISION_ENDPOINT"]
            url = f"{endpoint}computervision/imageanalysis:analyze?features=read&gender-neutral-caption=false&api-version=2023-10-01"
            key =os.environ["VISION_KEY"]

            headers = {
                'Ocp-Apim-Subscription-Key': key,
                'Content-Type': 'application/json; charset=utf-8'
            }

            # with image url
            image_url = "C:/Users/buehrer/Downloads/fidelity_statement.pdf"

            # Create an instance of the AnalyzeRequest class
            analyze_request = AnalyzeRequest(uri=image_url)

            # Serialize the instance to a dictionary
            json_data = asdict(analyze_request)

            response = requests.post(url, headers=headers, json=json_data)
            response_content = response.text

            # Print the JSON response for debugging
            print("Response Content:", response_content)

            # Deserialize and print the result
            data = json.loads(response_content)
            try:
                deserialized_object = self.from_dict(AnalyzeResult, data)

                print(f"Model Version: {deserialized_object.modelVersion}")
                print(f"Metadata - Width: {deserialized_object.metadata.width}, Height: {deserialized_object.metadata.height}")
                for block in deserialized_object.readResult.blocks:
                    for line in block.lines:
                        print(f"Line Text: {line.text}")
                        for boundingPoint in line.boundingPolygon:
                            print(f"Bounding Polygon Point - X: {boundingPoint.x}, Y: {boundingPoint.y}")
                        for word in line.words:
                            print(f"Word Text: {word.text}, Confidence: {word.confidence}")
                            for wordBoundingPoint in word.boundingPolygon:
                                print(f"Word Bounding Polygon Point - X: {wordBoundingPoint.x}, Y: {wordBoundingPoint.y}")
                        
            except KeyError as e:
                print(f"KeyError: {e}. Please check the JSON response structure.")

        def from_dict(self, data_class, data):
            if isinstance(data, list):
                return [self.from_dict(data_class.__args__[0], item) for item in data]
            if isinstance(data, dict):
                fieldtypes = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
                return data_class(**{k: self.from_dict(fieldtypes[k], v) for k, v in data.items()})
            return data
""" """  """ """
#test azure open ai for gpt4o
if False:
    client = AzureOpenAI(
        api_key = os.getenv("GPT4O_KEY2"),
        api_version = "2024-02-01",
        azure_endpoint = os.getenv("GPT4O_BASE_URL2")
    )
    response = client.chat.completions.create(
        model="gpt-4o-greg",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "generate an image that shows a tiger"}
        ]
    )
    print(response.choices[0].message)

#test openai (not azure openai
if False:
    #completions API
    from openai import OpenAI
    client = OpenAI()
    #ogg wav, mp3
    encoded_audio = base64.b64encode(open('./sound.wav', 'rb').read()).decode('ascii')
    #new response type 'multimodal'
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 
                [
                    {
                        "type": "audio_url",
                        "image_url": 
                        {
                            "url": f"data:audio/wav;base64,{encoded_audio}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "tell me the text in the audio"
                    }
                ]
            }
        ]
    )
    print(completion.choices[0].message)

#test bert model from hf
if False:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)

#test phi3 model from hf
if False:
    #print the time
    print(str(datetime.datetime.now()))
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    torch.random.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    
    messages = [
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])
    print(str(datetime.datetime.now()))

# test structured output from gpt4o
# the method below uses python classes to define the output, which is way easier to get right
# since defining the json explicitly is error prone
# NOTE: most use gpt4o 8-1 as that is the latest version
if False:
    from pydantic import BaseModel
    from openai import AzureOpenAI

    client = AzureOpenAI(
    azure_endpoint = os.getenv("GPT4O_ENDPOINT2"), 
    api_key=os.getenv("GPT4O_KEY2"),  
    api_version="2024-08-01-preview"
    )


    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-greg", # replace with the model deployment name of your gpt-4o 2024-08-06 deployment
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed

    print(event)
    print(completion.model_dump_json(indent=2))
    #'{\n  "id": "chatcmpl-A4dPvCnjhoEebok2Y1lXAZbdvqNAy",\n  "choices": [\n    {\n      "finish_reason": "stop",\n      "index": 0,\n      "logprobs": null,\n      "message": {\n        "content": "{\\"name\\":\\"Science Fair\\",\\"date\\":\\"Friday\\",\\"participants\\":[\\"Alice\\",\\"Bob\\"]}",\n        "refusal": null,\n        "role": "assistant",\n        "function_call": null,\n        "tool_calls": [],\n        "parsed": {\n          "name": "Science Fair",\n          "date": "Friday",\n          "participants": [\n            "Alice",\n            "Bob"\n          ]\n        }\n      },\n      "content_filter_results": {\n        "hate": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "self_harm": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "sexual": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "violence": {\n          "filtered": false,\n          "severity": "safe"\n        }\n      }\n    }\n  ],\n  "created": 1725668195,\n  "model": "gpt-4o-2024-08-06",\n  "object": "chat.completion",\n  "service_tier": null,\n  "system_fingerprint": "fp_b2ffeb16ee",\n  "usage": {\n    "completion_tokens": 17,\n    "prompt_tokens": 32,\n    "total_tokens": 49\n  },\n  "prompt_filter_results": [\n    {\n      "prompt_index": 0,\n      "content_filter_results": {\n        "hate": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "self_harm": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "sexual": {\n          "filtered": false,\n          "severity": "safe"\n        },\n        "violence": {\n          "filtered": false,\n          "severity": "safe"\n        }\n      }\n    }\n  ]\n}'

def run_ocr(filename):

    # Configuration
    #print('encoidng image')
    #encoded_image = base64.b64encode(open(filename, 'rb').read()).decode('ascii')
    #print('done encoidng image')

    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # [START read]
    # Load image to analyze into a 'bytes' object
    with open(filename, "rb") as f:
        image_data = f.read()

    # Extract text (OCR) from an image stream. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Print text (OCR) analysis results to the console
    print("Image analysis results:")
    print(" Read:")
    res=""
    if result.read is not None:
        for line in result.read.blocks[0].lines:
            res=res+line.text+"\n"
        '''  
        for line in result.read.blocks[0].lines:
            print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            for word in line.words:
                print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")'''
                
    # [END read]
    print(f" Image height: {result.metadata.height}")
    print(f" Image width: {result.metadata.width}")
    print(f" Model version: {result.model_version}")
    return res

vision_model="4o"
#vision_model="4v"
#vision_model="4o-mini"

#run common inference sdk
if False:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    from promptflow.client import load_flow
    import promptflow.evals
    from promptflow.evals.evaluate import evaluate
    from promptflow.evals.evaluators import ContentSafetyEvaluator
    endpoint = os.getenv(f"AZUREAI_PHI3_MINI_URL")
    key = os.getenv(f"AZUREAI_PHI3_MINI_KEY")
    print("endpoint", endpoint)
    print("key", key)
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )
    #not used
    messages = [
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
    combined_message = UserMessage(content=f"what time is it")
    messages = [combined_message]
    response = client.create(messages=messages, temperature=0, max_tokens=1000)
    x=1  
        
    '''response = evaluate(
        evaluation_name=evaluation_name,
        data=data_file,
        target=SalesDataInsights(model_type=model),
        evaluators={ 
        # Check out promptflow-evals package for more built-in evaluators
        # like gpt-groundedness, gpt-similarity and content safety metrics.
            "content_safety": ContentSafetyEvaluator(project_scope={
                "subscription_id": "15ae9cb6-95c1-483d-a0e3-b1a1a3b06324",
                "resource_group_name": "danielsc",
                "project_name": "build-demo-project"
            }),
            "execution_time": execution_time_evaluator,
            "error": error_evaluator,
            "sql_similarity": sql_similarity_evaluator,
        },
        evaluator_config={
            "sql_similarity": {
                "response": "${target.query}",
                "ground_truth": "${data.ground_truth_query}"
            },
            "execution_time": {
                "execution_time": "${target.execution_time}"
            },
            "error": {
                "error": "${target.error}"
            },
            "content_safety": {
                "question": "${target.query}",
                "answer": "${target.data}"
            }
        }
    )'''

#defines image processing/pickling functions when using gpt4o
if True:
    #gpt4v
    @trace_greg
    def get_image_text(filename):
        #print the time
        #print(str(datetime.datetime.now()))
        #res = run_ocr(filename)
        #return res
        # Configuration
        print('encoidng image')
        encoded_image = base64.b64encode(open(filename, 'rb').read()).decode('ascii')
        print('done encoidng image')
        if vision_model=="4o":
            headers = {
                "Content-Type": "application/json",
                "api-key": os.getenv("GPT4O_KEY2"),
            }
        elif vision_model=="4v":
            headers = {
                "Content-Type": "application/json",
                "api-key": os.getenv("GPT4K1"),
            }    
        elif vision_model=="4o-mini":
            headers = {
                "Content-Type": "application/json",
                "api-key": os.getenv("GPT4O_MINI_KEY2"),
            } 

        # Payload for the request
        payload = {
            "messages": 
            [
                {
                    "role": "system",
                    "content": 
                    [
                        {
                            "type": "text",
                            "text": "You are an AI assistant specializing in form recognition for extracting information."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": 
                    [
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "List all the key-value pairs in the image.  Provide the information in table format using ascii so it prints to the command line in a human readable way."
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 800
        }


        # Send request
        print('running ocr via openai on image')
        try:
            if vision_model=="4o":
                response = requests.post(os.getenv('GPT4O_ENDPOINT2'), headers=headers, json=payload)
            elif vision_model=="4v":
                response = requests.post(os.getenv('GPT4V_ENDPOINT'), headers=headers, json=payload)
            elif vision_model=="40-mini":
                response = requests.post(os.getenv('GPT4O_MINI_ENDPOINT2'), headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
        #print('done running ocr on image')
        #print(str(datetime.datetime.now()))
        return response.json()

    # Set default download folder for ChromeDriver
    @trace_greg
    def fetch_url(address,storagename):
        # SELENIUM SETUP
        try:
            logging.getLogger('WDM').setLevel(logging.WARNING)  # just to hide not so rilevant webdriver-manager messages
            #chrome
            options = webdriver.ChromeOptions() 
            options.add_argument('user-data-dir=C:/projects/agen3/User Data') 
            options.add_argument('profile-directory=Default')
            #options.add_argument('profile-directory=Profile 5')
            #options.headless = True 
            #options.add_argument("--headless=new")
            
            # driver = webdriver.Chrome(executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", options=options)
            # service = Service(executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
            # options = webdriver.ChromeOptions()
            # driver = webdriver.Chrome(service=service, options=options)

            driver=webdriver.Chrome()

            driver.get("https://www.facebook.com/")
            #chrome_options.add_experimental_option("prefs", prefs)
            #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            #firefox
            #driver = webdriver.Firefox()
            
            driver.implicitly_wait(1)
            print("getting address " + address)
            driver.maximize_window()
            driver.get(address)
            driver.set_window_size(1920, 1080)  # to set the screenshot width
            driver.implicitly_wait(1)
            #button1=driver.find_element(By.PARTIAL_LINK_TEXT,"Accept all")
            #button1.click()
            print("saving screenshot")
            save_screenshot(driver, storagename)
            print("saved at "+ str(storagename))
            driver.quit()
        except Exception as e1:
            print("error in fetch_url")
            driver.quit()
            print(str(e1))
        
    def save_screenshot(driver, file_name):
        #height, width = scroll_down(driver)
        #driver.set_window_size(width, height)
        img_binary = driver.get_screenshot_as_png()
        img = Image.open(BytesIO(img_binary))
        img.save(file_name)
        # print(file_name)
        print("Screenshot saved!")
        
    def scroll_down(driver):
        total_width = driver.execute_script("return document.body.offsetWidth")
        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        viewport_width = driver.execute_script("return document.body.clientWidth")
        viewport_height = driver.execute_script("return window.innerHeight")
        rectangles = []
        i = 0
        while i < total_height:
            ii = 0
            top_height = i + viewport_height
            if top_height > total_height:
                top_height = total_height
            while ii < total_width:
                top_width = ii + viewport_width
                if top_width > total_width:
                    top_width = total_width
                rectangles.append((ii, i, top_width, top_height))
                ii = ii + viewport_width
            i = i + viewport_height
        previous = None
        part = 0
        for rectangle in rectangles:
            if not previous is None:
                driver.execute_script("window.scrollTo({0}, {1})".format(rectangle[0], rectangle[1]))
                time.sleep(0.5)
            # time.sleep(0.2)
            if rectangle[1] + viewport_height > total_height:
                offset = (rectangle[0], total_height - viewport_height)
            else:
                offset = (rectangle[0], rectangle[1])
            previous = rectangle
        return total_height, total_width

def setup_app_insights():
    
    inject_openai_api()

    # dial down the logs for azure monitor -- it is so chatty
    azmon_logger = logging.getLogger('azure')
    azmon_logger.setLevel(logging.WARNING)

    # Set the Tracer Provider
    trace.set_tracer_provider(TracerProvider())

    # Configure Azure Monitor as the Exporter
    print("using the follwoing connection string", os.getenv('APPINSIGHTS_CONNECTION_STRING'))
    trace_exporter = AzureMonitorTraceExporter(
        connection_string=os.getenv('APPINSIGHTS_CONNECTION_STRING')
    )

    # Add the Azure exporter to the tracer provider
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(trace_exporter)
    )

    # Configure Console as the Exporter
    file = open('spans.json', 'w')

    # Configure Console as the Exporter and pass the file object
    console_exporter = ConsoleSpanExporter(out=file)

    # Add the console exporter to the tracer provider
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(console_exporter)
    )
    # Get a tracer
    return trace.get_tracer(__name__) 

#test ocr vs aoai
if False:
    res=run_ocr("C:/Users/buehrer/Downloads/-730444229080234320.jpg")
    print(res)
    res=get_image_text("C:/Users/buehrer/Downloads/fidelity_image.png")
    print(res)
    
    
#aoai embeddings
if False:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("ADA2_ENDPOINT"),
        #azure_ad_token_provider=token_provider,
        api_version="2024-02-01",
    )
    def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
        res = client.embeddings.create(input = [text], model=model)
        emb = res.data[0].embedding
        return emb
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#image pickling
#@search_scraper.register_for_llm(name="search scraper tool", description="use the web to find answers to questions")
@trace_greg
def get_web_info(query:str)->str:
    rand = str(time.time()).split(".")[0]
    storage = os.path.join("c:/tmp", 'Screenshot'+rand+'.png')
    url = "https://www.google.com/search?q="+urllib.parse.quote_plus(query)+"&sclient=gws-wiz&hl=en&lang=en"
    print("scaping " + url)
    fetch_url(url,storage)
    print("wrote to " + str(storage))
    res = get_image_text(storage)
    print("response from ocr call is " + str(res))
    if type(res) == str:
        return res
    else:
        return res['choices'][0]['message']['content']

#sql test 
class sql_skill:
    conn = None
    cur  = None
    def insert_file(self, filename, tablename):
        df = pd.read_csv(filename)
        df.to_sql(tablename, self.conn, if_exists='replace', index=False)
        
    def init(self):
        self.conn = sqlite3.connect('file:cachedb?mode=memory&cache=shared')
        self.cur  = self.conn.cursor()
    def create_table(self,tablename,num_cols):
        #make create table string
        qry = "create table " + tablename + " ( "
        for i in range(0,num_cols):
            qry += "col" + str(i) + " text, "
        qry = qry[:-2] + " );"
        self.cur.execute(qry)    
    def insert(self, tablename,values):
        qry = "insert into " + tablename + " values ( "
        for v in values:
            qry += v + ", "
        qry = qry[:-2] + " )"
        self.cur.execute(qry)
    def select(self, tablename,columns):
        qry = "select " + columns + " from " + tablename
        self.cur.execute(qry)
        return self.cur.fetchall()
    def delete(self, tablename,condition):
        qry = "delete from " + tablename + " where " + condition
        self.cur.execute(qry)  
    def clear_memory(self):
        tables = list(self.cur.execute("select name from sqlite_master where type is 'table'"))
        self.cur.executescript(';'.join(["drop table if exists %s" %i for i in tables]))
        self.cur.execute("VACUUM")    
if False:
    x=1
    # sql =  sql_skill()
    # sql.init()
    # sql.clear_memory()
    # tablename="session"+str(1)
    # sql.create_table(tablename=tablename,num_cols=3)
    # sql.insert(tablename,values=["'a'","'b'","'c'"])
    # sql.insert(tablename,values=["'a'","'b'","'c'"])
    # sql.insert(tablename,values=["'a'","'b'","'c'"])
    # data = sql.select(tablename,"*")
    # print(data)
    
#embeddings test    
if False:   
    client = AzureOpenAI(
    api_key = os.getenv("ADA2_KEY"),  
    api_version = "2023-05-15",
    azure_endpoint = os.getenv("ADA2_ENDPOINT")
    )
    test1 = "hello my name is greg what is your name?"
    test2 = "hello my name is sally what is your name?"
    test3 = "what is the most popular name from 1971?"
    emb1=generate_embeddings(test1)
    emb2=generate_embeddings(test2)
    emb3=generate_embeddings(test3)
    sim12=cosine_similarity(emb1,emb2)
    sim21=cosine_similarity(emb2,emb1)
    sim23=cosine_similarity(emb2,emb3)
    sim13=cosine_similarity(emb1,emb3)
    x=1



gpt4v = [
    {
        'model': 'gpt-4-vision-preview',
        'api_key': os.environ['GPT4K1'],
        'base_url': os.environ['GPT4V_BASE_URL'],
        'api_type': 'azure',
        'api_version': '2023-07-01-preview',
    },
]
gpt4 = [
    {
        'model': 'gpt-4',
        'api_key': os.environ['GPT4_API_KEY'],
        'base_url': os.environ['GPT4_BASE_URL'],
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
]
gpt4o = [
    {
        'model': 'gpt-4o-mini',
        #'model': 'gpt-4o-greg',
        'api_key': os.environ['GPT4O_MINI_KEY2'],
        'base_url': os.environ['GPT4O_MINI_BASE_URL2'],
        'api_type': 'azure',
        'api_version': '2024-02-15-preview', # '2024-02-01',
    },
]
gpt35turbo = [
    {
        'model': 'gpt-35-turbo-1106',
        'api_key': os.environ['OPENAI_API_KEY'],
        'base_url':  os.environ['OPENAI_BASE_URL'],
        'api_type': 'azure',
        'api_version': '2023-03-15-preview',
    },
]
function_list = [{
        "name": "get_web_info",
        "description": "use the web to find answers to questions",
        "parameters": 
        {
            "type": "object",
            "properties": 
            {
                " query": 
                    {
                        "type": "string",
                        "description": "input query to search for on the web"
                    }
            },
            "required": ["query"]
        }
    },
    {
        "name": "tool_train_lightgbm_model",
        "description": "trains a machine learnt model to predict the target variable of a dataset",
        "parameters": 
        {
            "type": "object",
            "properties": 
            {
                "filename": 
                    {
                        "type": "string",
                        "description": "location of the file to use for training"
                    },
                "target_column": 
                    {
                        "type": "integer",
                        "description": "column of the target variable in the dataset"
                    }
            },
            "required": ["filename","target_column"]
        }
    }            
]

USING_GPT4o = True

if USING_GPT4o:
    os.environ['OPENAI_API_KEY'] = os.environ['GPT4O_KEY2']
    os.environ['OPENAI_BASE_URL'] = os.environ['GPT4O_BASE_URL2']
    os.environ['OPENAI_API_VERSION'] = '2024-02-01'
    #llm configs
    llm_base={
        "timeout": 600,
        "cache_seed": None,
        "config_list": gpt4o,
        "temperature": 0.1,
        "cache_seed": None,
    }
    llm_web_scraper={
        "timeout": 600,
        "cache_seed": None,
        "config_list": gpt4o,
        "temperature": 0.1,
        "cache_seed": None,
        "functions": function_list,
    }
else:
     #llm configs
    llm_base={
        "timeout": 600,
        "cache_seed": None,
        "config_list": gpt4,
        "temperature": 0.1,
        "cache_seed": None,
    }
    llm_web_scraper={
        "timeout": 600,
        "cache_seed": None,
        "config_list": gpt4,
        "temperature": 0.1,
        "cache_seed": None,
        "functions": function_list,
    }   
if False:
    for i in range(0,10):
        #time the run 
        starttime = time.time()
        # run_ocr("C:/tmp/Screenshot1722180919.png")
        #get_image_text("C:/tmp/Screenshot1722180919.png")
        endtime = time.time()
        diff = endtime - starttime
        print("time to run " + str(diff))
    
# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="planning_assistant",
    #function_map = {"function to get search results from the web": get_web_results},
    description="planning_assistant is a helpful AI planning assistant. planning_assistant cannot not run code but planning_assistant can write code to the user to run.",
    llm_config=llm_base,
    system_message="""You are a helpful AI assistant that solves tasks.
    
Solve the task step by step if possible, using chain of thought reasoning where appropriate. 
Be clear which step uses code, and which step requires language skills.

Reply "TERMINATE" in the end when everything is done.
    """
)


# create an AssistantAgent instance named "assistant"
web_scraper = autogen.AssistantAgent(
    name="web_scraper",
    description="You are a helpful AI programming assistant whose programs find web urls from google and bing to answer questions.  web_scraper can write code but you do not run code.",
    llm_config=llm_base,
    system_message="""You are a helpful web browsing assistant.
    
Solve tasks using your coding and language skills by seearching on bing.com or google.com for urls, then scraping those urls for their html.
Always suggest code that gets the urls from bing.com or google.com, then fetches the html of the first 3 algorithm urls.
Always try to scrape all three of the first three algorithm urls.
If the search results page has an instant answer, use the information in the instant answer.

For answers on google seearch pages, use the <div data-domdiff> tag to find the instant answer.
For answers on bing seearch pages, use the <div class="b_focusTextLarge"> tag to find the instant answer.
For scraping google, use the 'h3' tag to find the algo urls and extract the text from them.
For scraping bing, use the 'li' tag with the 'b_algo' class to find the algo urls and extract the text from them.

1. In all cases, suggest python code (in a python coding block) for the user to execute.
2. If the result indicates there is an error, fix the error and output the code again. 
3. Always suggest the full code instead of partial code or code changes.
4. Never suggest incomplete code which requires users to modify.
5. Never ask users to copy and paste the result.
6. Never suggest the same code twice, since it was already tried and failed.
7. For any question where the answer is about recent events, always find the current day first and use that day to find the answer.

You will assume the needed packages are already installed, so do NOT attempt to pip install any python packages, unless the code errros out.
The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
 If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
    """
)

# create an AssistantAgent instance named "assistant"
science_coder = autogen.AssistantAgent(
    name="science_coder",
    description="You are a helpful AI programming assistant whose programs use math and science to handle complex tasks.  As science_coder you can write code and you can run code.",
    llm_config=llm_base,
    code_execution_config={"work_dir": "web","use_docker": False},
    system_message="""You are a helpful web browsing assistant.
    
Solve science and math tasks by creating code to compute the answers.
In the following cases, suggest python code (in a python coding block) for the user to execute.
    1. When you need to perform some empirical computation with code, use the code to perform the task and output the result. Finish the task smartly.
    2. If you output or change a file as part of the solution, validate the file has been generated or modified correctly.
    """
)

# create an AssistantAgent instance named "assistant"
regular_coder = autogen.AssistantAgent(
    name="regular_coder",
    description="You are a helpful AI programming assistant who writes programs to solve tasks.  As regular_coder, you cannot not run code but you can write code to the user to run.",
    llm_config=llm_base,
    system_message="""Solve tasks using your coding and language skills.
    
In the following cases, suggest python code (in a python coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
    3. If you output or change a file as part of the solution, validate the file has been generated or modified correctly.
    4. For any question where the answer is about recent events, always find the current day first and use that day to find the answer.  Include todays date in your answer.
    
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. You highly prefer python code.  
You can use the requests library to write code to search bing.com or google.com for information if needed.
That method requires then using the requests library to get the algo urls using li and the b_algo class.
Then the code must use the requests library to get the text of the first algo url.
However this approach is just one approach and if this fails you must try a different approach.
You will assume the needed packages are already installed, so do NOT attempt to pip install any python packages, unless the code errros out.
The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done."""
)

# create an AssistantAgent instance named "assistant"
code_runner = autogen.AssistantAgent(
    name="code_runner",
    description="You are a helpful AI programming assistant who runs programs on behalf of other agents.  If a program is sent to you, you run it and return the output.",
    llm_config=llm_base,
    code_execution_config={"work_dir": "web","use_docker": False},
    system_message="""You are a helpful AI programming assistant who runs programs on behalf of other agents.  
    If a program is sent to you, you run it and return the output."""
)


# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    description="""You proxy messages between the user and a collection of langauge model agents. As the user_agent you do not attempt to answer questions.  
    You keep track of the original question and make sure the question gets answered.""",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    llm_config=llm_base,
    code_execution_config={"work_dir": "web","use_docker": False},
    system_message="""You are an assistant for proxying messages between the user and a collection of special agents.
    You are tasked with getting the user's problem solved by working with the chat manager agent.
    You read the recent messages and decide whether or not they contain the answer to the question.
    
    If the answer is not in the response, ask the chat manager to revisit the question.
    If the answer is in the response but is not concise, you ask the chat manager to provide a more concise answer from the text.
    If the task has been solved you should reply TERMINATE.
    
    """
)
#old one
'''    You are tasked with getting the user's problem solved by working with the chat manager agent.
    
    You read the recent messages and decide whether or not they contain the answer to the question.
    You do not try to answer the question.
    If the answer is not in the response, ask the chat manager to revisit the question.
    If the response has the word terminate in it, you should reply TERMINATE.
    If the task has been solved you should reply "TERMINATE.'''


# create an AssistantAgent instance named "assistant"
web_scraper_pickler = autogen.AssistantAgent(
    name="web_scraper_pickler",
    function_map = {"get_web_info": get_web_info},
    description="""web_scraper_pickler is a helpful AI assistant who uses a function map to get the latest information from the web. 
    web_scraper_pickler loves to solve problems by making function calls that answer real time questions.
    web_scraper_pickler can write code but you do not run code.  
    
    As a web browsing assistant, web_scraper_pickler does have access to the web and can use the web to find answers to questions.
    That is the key skill of web_scraper_pickler, you can find information on the web and use that information to answer questions.""",
    llm_config=llm_web_scraper,
    system_message="""You are a helpful AI assistant who uses a function map to get the latest information from the web. 
    You love to solve problems by making function calls that answer real time questions."""
)

# create an AssistantAgent instance named "assistant"
train_lightgbm_agent = autogen.AssistantAgent(
    name="train_lightgbm_agent",
    function_map = {"tool_train_lightgbm_model": tool_train_lightgbm_model},
    description="""  """,
    llm_config=llm_web_scraper,
    system_message="""  """
)

#create an assistant named chat manager
groupchat = autogen.GroupChat(
        #agents=[user_proxy, code_runner, web_scraper_pickler, assistant],   
        agents=[user_proxy, web_scraper_pickler, assistant],   
            messages=[],  admin_name="admin", max_round=20, speaker_selection_method="auto"
    )
manager = autogen.GroupChatManager(groupchat=groupchat, description="chat manager", llm_config=llm_base)


task1 = """how tall is the sears tower?"""
task2 = """did the mariners win last night?"""
task3 = """what time is it in seattle?"""

print(str(datetime.datetime.now()))
start_trace()

@trace_greg
def run_it(task1):
    print("starting chat, task is '"+task1+"'")
    res = manager.initiate_chat(
        manager,
        message=task1,
    )
    print(str(res))
print(str(datetime.datetime.now()))

run_it("""how tall is the sears tower?""")
run_it("""did the mariners win last night?""")
run_it("""what religions do not allow cross religion marriages""")
run_it("""what is 8*7""")

task1="what time is it in seattle?"
while True:
    run_it(task1)
    task1=input("enter a task: ")



if False:
    '''
    from openai import OpenAI
    import os
    import openai

    import urllib.request
    import json
    import os
    import ssl

    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    def maas_call(req):
        # Allow Self-signed certificate
        allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
        # Request data goes here
        # The example below assumes JSON formatting which may be updated
        # depending on the format your endpoint expects.
        # More information can be found here:
        # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
        data = {
            "input_data": {
                "input_string": [
                    req
                ],
                "parameters": {
                    "max_new_tokens": 1000,
                    #"do_sample": True,
                    "return_full_text": True
                }
            }
        }
        body = str.encode(json.dumps(data))
        url = 'https://gbbs-greg-9-mbzrr.eastus.inference.ml.azure.com/score'
        # Replace this with the primary/secondary key or AMLToken for the endpoint
        api_key = '9ikk3cjyE318I4XJ5cHcLTax2U1rfgS6'
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'mistralai-mixtral-8x7b-v01-4' }

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            print(result)
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))

    #read a file
    f1 = open('C:/Users/buehrer/Documents/multiagentsystems.txt', 'r', encoding="cp1252")
    text = f1.read()
    f1.close()
    wordset = set(text.split())



    def samplewords(wordset, n):
        import random
        return random.sample(wordset, n)

    for i in range(0, 10):
        context = "You are a story telling assistant. "
        context += "You are skilled in explaining complex programming concepts with creative flair. "
        context += "Use the following words to create your 1000 word story: " + " ".join(samplewords(wordset, 20))
        #make MaaS call
        res = maas_call(context)
        print(str(res))
    '''

    '''  
    # THIS CODE IS GOOFING AROUND WITH ASSISTANTS
    if True:
        client = OpenAI()
        openai.api_type = "azure"
        openai.api_base = "https://aoai.azure.com/"
        openai.api_version = "2023-07-15-preview"
        openai.api_key = ""#os.getenv("OPENAI_API_KEY")

    completion = client.chat.completions.create(
    model="gpt-4-32k,
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]
    )

    print(completion.choices[0].message)


    #try assistants
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-1106-preview"
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )
    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )    
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )        '''
        
        
                                                                                                             