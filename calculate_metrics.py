import os.path as osp
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(evalLabels, predictedLabels, saveDir, threshold=25, model='gpt-4', temperature=0.3, num_trials=3):
    predictedLabels = (predictedLabels > threshold).astype(int)

    report = classification_report(evalLabels, predictedLabels)
    with open(osp.join(saveDir, "result.txt"), 'a') as f:
        f.write(f"model: {model}, temp: {temperature} \n")
        f.write(f"Threshold: {threshold}, Trial: Average of {num_trials} trials \n \n")
        f.write(f"Classification Report: \n")
        f.write(report)
        f.write("###################################################################")
    
    
def plot_confusion_matrix(evalLabels, predictedLabels, saveDir, threshold=25):
    predictedLabels = (predictedLabels > threshold).astype(int)
    cm = confusion_matrix(evalLabels, predictedLabels)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.savefig(osp.join(saveDir, 'confusion_matrix.png'))
    plt.close()


def plot_roc_curve(evalLabels, predictedLabels, saveDir):
    fpr, tpr, thresholds = roc_curve(evalLabels, predictedLabels)

    roc_auc = auc(fpr, tpr)

    # Step 6: Plot the ROC curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='GPT-4 (AUC = %0.2f)' % roc_auc)
    plt.scatter(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random guessing (AUC = 0.50)')
    for i in range(0, len(thresholds), 2):
        if thresholds[i] == 19:
            plt.annotate(f'{18:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(osp.join(saveDir, 'roc.png'))
    plt.close()

if __name__ == '__main__':
    baseDir = r"/home/parvez/Projects/MentalHealthProject/Data/Social Anxiety/ADEFGH/"
    saveDir = osp.join(baseDir, "Results2")

    #**********variabels*************
    actualThreshold = 25
    threshold = 18
    model = 'gpt-4'
    temperature = 0.3
    num_trials = 3


    df = pd.read_csv(osp.join(saveDir, "SPIN_Scores.csv"), sep=',')
    df = df.dropna(axis=0)



    evalLabels = df['SPIN SCORE'] > actualThreshold
    predictedLabels = df[['gpt4-trial1', 'gpt4-trial2','gpt4-trial3']].mean(axis=1).astype(int)


    calculate_metrics(evalLabels, predictedLabels, saveDir=saveDir, threshold=threshold)
    # plot_confusion_matrix(evalLabels, predictedLabels, saveDir=saveDir, threshold=threshold)
    # plot_roc_curve(evalLabels, predictedLabels, saveDir=saveDir)