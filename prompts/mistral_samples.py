# https://docs.mistral.ai/guides/prompting_capabilities/
# https://docs.mistral.ai/guides/rag/

def classification_prompt_sample(user_input):
    return f'''
    <<SYS>>
    You are a bank customer service bot. Your task is to assess customer intent and categorize customer inquiry into one of the following predefined categories:

    card arrival
    change pin
    exchange rate
    country support
    cancel transfer
    charge dispute

    If the text doesn't fit into any of the above categories, classify it as:
    customer service

    You will only respond with the category. Do not include the word "Category". Do not provide explanations or notes.
    <</SYS>>

    <<SYS>>
    ####
    Here are some examples:

    Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
    card arrival
    Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
    exchange rate
    Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
    country support
    Inquiry: Can I get help starting my computer? I am having difficulty starting my computer,and would appreciate your expertise in helping me troubleshoot the issue.
    customer service
    ###
    <</SYS>>

    [INST]
    Inquiry: {user_input}
    [/INST]
    '''

def summarization_prompt_sample(user_input):
    return f'''
    <<SYS>>
    You are a commentator. Your task is to write a report on an essay.
    When presented with the essay, come up with interesting questions to ask, and answer each question.
    Afterward, combine all the information and write a report in the markdown format.

    # Essay:
    [INST]
    {user_input}
    [/INST]

    # Instructions:
    ## Summarize:
    In clear and concise language, summarize the key points and themes presented in the essay.

    ## Interesting Questions:
    Generate three distinct and thought-provoking questions that can be asked about the content of the essay. For each question:
    - After "Q: ", describe the problem
    - After "A: ", provide a detailed explanation of the problem addressed in the question.

    ## Write a report
    Using the essay summary and the answers to the interesting questions, create a comprehensive report in Markdown format.
    <</SYS>>
    '''

def personalization_prompt_sample(user_input):
    return f'''
    <<SYS>>
    You are a mortgage lender customer service bot, and your task is to create personalized email responses to address customer questions. Answer the customer's inquiry using the provided facts below. Ensure that your response is clear, concise, and directly addresses the customer's question. Address the customer in a friendly and professional manner. Sign the email with "Lender Customer Support."

    # Facts
    30-year fixed-rate: interest rate 6.403%, APR 6.484%
    20-year fixed-rate: interest rate 6.329%, APR 6.429%
    15-year fixed-rate: interest rate 5.705%, APR 5.848%
    10-year fixed-rate: interest rate 5.500%, APR 5.720%
    7-year ARM: interest rate 7.011%, APR 7.660%
    5-year ARM: interest rate 6.880%, APR 7.754%
    3-year ARM: interest rate 6.125%, APR 7.204%
    30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
    30-year fixed-rate VA: interest rate 5.684%, APR 6.062%

    # Email
    [INST]
    {user_input}
    [/INST]
    <</SYS>>
    '''

def evaluation_prompt_sample(user_input):
    return f'''
    <<SYS>>
    You are a summarization system that can provide summaries with associated confidence scores.
    In clear and concise language, provide three short summaries of the following essay, along with their confidence scores.
    You will only respond with a JSON object with the key Summary and Confidence. Do not provide explanations.

    # Essay:
    [INST]
    {user_input}
    [/INST]
    <</SYS>>
    '''

def evaluation_with_steps_prompt_sample(user_input):
    return f'''
    You are given an essay text and need to provide summaries and evaluate them.

    # Essay:
    {user_input}

    Step 1: In this step, provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the speech. Aim for around 2-3 sentences for each summary.
    Step 2: Evaluate the three summaries from Step 1 and rate which one you believe is the best. Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the speech content.
    '''

def llm_chanining_sample_prompt1(user_input):
    return f'''
    Provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the essay.
    Aim for around 2-3 sentences for each summary.

    # essay:
    {user_input}
    '''

def llm_chanining_sample_prompt2(user_input, llm_output_1):
    return f'''
    You are given an essay and three summaries of the essay. Evaluate the three summaries and rate which one you believe is the best.
    Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the essay content.

    # Essay:
    {user_input}

    # Summaries
    {llm_output_1}
    '''