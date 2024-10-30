import os.path as osp 
import pandas as pd
import tiktoken 

def handleMissingFiles(dataDir, evalDF):
    drop_idxs = []
    for idx, row in evalDF.iterrows():
        if not osp.exists(osp.join(dataDir, f"{row['ID']}.csv")):
            print(f"skipping participant {row['ID']}, as no transcript exists :(")
            drop_idxs.append(idx)
    evalDF.drop(drop_idxs, inplace=True)
    evalDF.reset_index(drop=True, inplace=True)


def processTranscript(df:pd.DataFrame):
    processedText = "Interviewer: "
    previousSpeaker = 'I'
    for idx, row in df.iterrows():
        currentSpeaker = row['Speaker']
        if previousSpeaker == currentSpeaker:
            processedText += " " + row['Transcript']
        else: 
            speaker = 'Interviewer:' if currentSpeaker == "I" else 'Participant:'
            processedText += "; " + speaker + row['Transcript']
        previousSpeaker = currentSpeaker
    return processedText


    
def manage_tokens(start_prompt, input_text, model, context_length):
    # Initialize the tokenizer for GPT-4
    tokenizer = tiktoken.encoding_for_model(model_name=model)
    
    # Tokenize the input text
    input_tokens = tokenizer.encode(input_text)
    prompt_tokens = tokenizer.encode(start_prompt)

    # Check if the token length exceeds the context length
    adjustment = 15
    if len(prompt_tokens) + len(input_tokens) + adjustment > context_length:
        print(f"Token length exceeded. using the first {context_length} tokens")
        # Truncate the tokens to fit within the context length
        tokens_left = context_length - len(prompt_tokens) - adjustment 
        input_tokens = input_tokens[:tokens_left]
        # Decode the tokens back to text
        truncated_text = tokenizer.decode(input_tokens)
        return truncated_text
    
    return input_text

