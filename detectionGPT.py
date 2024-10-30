import argparse
import os
import os.path as osp 
from openai import OpenAI 
import pandas as pd

from utils import handleMissingFiles, processTranscript, manage_tokens
from calculate_metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve

# Set API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


startPrompt = "Analyse the following transcript from an interview with regards to symptoms of social anxiety disorder. \
Based on this interview, infer the score this person has most likely attained on the Social Phobia Inventory, SPIN. \
This score is in the range of 0 to 68 and if the score is greater than 25 then the person has social anxiety disorder. \
Only give me the score as an INTEGER WITHOUT EXPLANATION!"


def main():
    parser = argparse.ArgumentParser(description='Process interview transcriptions to get social anxiety scores using GPT.')
    parser.add_argument('--base_dir', type=str, required=False, 
                        default= r"/home/deep/MentalHealthProject/Data/Social Anxiety/PBNS_10.07.24/", 
                        help='Base directory with subdirectory Transcripts containing transcript csv files.',
                        )
    parser.add_argument('--scores', type=str, default=r"SPIN scores PBNS.xlsx", required=False, help='Name of the scores file',)
    parser.add_argument('--model', type=str, default=r"gpt-4", help='Name of the model to be used',)
    parser.add_argument('--context_length', type=int, default=8192, help='Length of the context window',)
    parser.add_argument('--temp', type=float, default=0.3, help='Temperature',)
    parser.add_argument('--trials', type=int, default=3, help='Number of trials',)
    parser.add_argument('--true_threshold', type=int, default=25, help='Threshold to be used for True Scores',)
    parser.add_argument('--threshold', type=int, default=25, help='Threshold to be used for Predicted Scores',)

    args = parser.parse_args()

    baseDir = args.base_dir
    evalDF = pd.read_excel(osp.join(baseDir, args.scores))

    dataDir = osp.join(baseDir,"Transcripts")
    handleMissingFiles(dataDir, evalDF)
    
    
    saveDir = osp.join(baseDir, "Results")
    if not osp.exists(saveDir):
        os.makedirs(saveDir)

    #**********variabels*************
    model = args.model
    context_length = args.context_length
    temperature = args.temp
    num_trials = args.trials
    
    ids = evalDF["ID"].tolist()
    evalScores = evalDF["SPIN SCORE"].tolist()


    print("model: ", model)
    print("temp: ", temperature)
    print("ids:", ids)
    print("Scores:", evalScores)


    for i in range(num_trials):
        print(f"Trial {i+1}")
        predictedScore = []
        for idx, participant_id in enumerate(ids):
            # filePath = osp.join(dataDir, f"{participant_id}_Transcript.csv")
            filePath = osp.join(dataDir, f"{participant_id}.csv")

            df = pd.read_csv(filePath, sep=',')
            
            # covert into one string
            interviewPrompt = processTranscript(df)
            interviewPrompt = manage_tokens(startPrompt, interviewPrompt, 
                                            model=model, context_length=context_length)
            
            messages = [{"role": "system", "content": startPrompt}, 
                        {"role": "user", "content": interviewPrompt}]

            # generate a response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature = temperature,
            )

            if ":" in response.choices[0].message.content:
                score = int(response.choices[0].message.content.split(":")[1])
            else:
                score = int(response.choices[0].message.content)
                
            print(f"{participant_id}: {score}")
            predictedScore.append(score)

        evalDF[f'{model}-trial{i+1}'] = predictedScore
        
    evalDF.to_csv(osp.join(saveDir, "Spin_Scores.csv"), index=False)
    print("#### Finished Predictions #####")

    ############# Calculate Metrics #########################
    print("#### Starting Metrics Calculation #####")
    actualThreshold = args.true_threshold
    threshold = args.threshold
    
    evalLabels = evalDF['SPIN SCORE'] > actualThreshold
    predictedLabels = evalDF[[f'{model}-trial{i+1}' for i in range(num_trials)]].mean(axis=1).astype(int)


    calculate_metrics(evalLabels, predictedLabels, saveDir=saveDir, threshold=threshold, 
                      model=model, temperature=temperature, num_trials=num_trials)
    plot_confusion_matrix(evalLabels, predictedLabels, saveDir=saveDir, threshold=threshold)
    plot_roc_curve(evalLabels, predictedLabels, saveDir=saveDir)



if __name__ == "__main__":
    main()

