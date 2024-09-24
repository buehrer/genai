

# word vector visualizations
if False:
    import os
    import gensim.downloader
    model = gensim.downloader.load('glove-wiki-gigaword-50')
    w=model['tower']
    z=model['building']
    x=5


# test evaluation for aml using single calls
# see https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/flow-evaluate-sdk
if True:
    import os
    from promptflow.core import AzureOpenAIModelConfiguration
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize Azure OpenAI Connection with your environment variables
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ.get("GPT4O_ENDPOINT2"),
        api_key=os.environ.get("GPT4O_KEY2"),
        azure_deployment="gpt-4o-greg",
        api_version="2024-02-15-preview",
    )

    from promptflow.evals.evaluators import * #RelevanceEvaluator

    # relwevance
    relevance_eval = RelevanceEvaluator(model_config)
    score1 = relevance_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list,"
        " the alpine explorer tent is the most waterproof."
        " The Adventure Dining Table has higher weight.",
        question="Which tent is the most waterproof?",
    )
    print(score1)
    
    #groundedness
    grounded_eval = GroundednessEvaluator(model_config)
    score2 = grounded_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list,"
        " the alpine explorer tent is the most waterproof."
        " The Adventure Dining Table has higher weight.",
        question="Which tent is the most waterproof?",
    )
    print(score2)
    
    #coherence
    coherence_eval = CoherenceEvaluator(model_config)
    score3 = coherence_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list,"
        " the alpine explorer tent is the most waterproof."
        " The Adventure Dining Table has higher weight.",
        question="Which tent is the most waterproof?",
    )
    print(score3)
    
    class AnswerLengthEvaluator:
        def __init__(self):
            pass

        def __call__(self, *, answer: str, **kwargs):
            return {"answer_length": len(answer)}
    
    custom_eval = AnswerLengthEvaluator()
    score4 = custom_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list,"
        " the alpine explorer tent is the most waterproof."
        " The Adventure Dining Table has higher weight.",
        question="Which tent is the most waterproof?",
    )
    print(score4)
    
    

x=5