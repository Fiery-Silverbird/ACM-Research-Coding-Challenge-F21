from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import tokenize


def classifyscore(score):
    if score <= -0.05:
        translatedScore = 'negative'
    elif -0.05 < score < 0.05:
        translatedScore = 'neutral'
    else:
        translatedScore = 'positive'
    return translatedScore


def main():
    with open('input.txt', 'r') as file:
        data = file.read().replace('\n', ' ')

    sentimentAnalyzer = SentimentIntensityAnalyzer()

    splitText = tokenize.sent_tokenize(data)

    # set to False for visibility, set to true if you need to see the sentences that the nltk library considers sentences
    displayEverySingleLine = False

    avgSentimentOfTextLines = 0
    numOfLines = 0
    for line in splitText:
        score = sentimentAnalyzer.polarity_scores(line)
        avgSentimentOfTextLines += score['compound']
        numOfLines += 1
        if displayEverySingleLine:
            print("Sentimental analysis of the sentence: " + line + str(score) + ", Overall it is " + classifyscore(score['compound']) + ".")
        else:
            print("Sentimental analysis of the sentence: " + str(
                sentimentAnalyzer.polarity_scores(line)) + ", Overall it is " + classifyscore(score['compound']) + ".")

    print()

    print("The average score of all the lines is " + str(avgSentimentOfTextLines/numOfLines) + " which means the whole text is " + classifyscore(avgSentimentOfTextLines/numOfLines) + ".")

    sentimentOfDataFile = sentimentAnalyzer.polarity_scores(data)

    print("Sentimental analysis of the whole text(input.txt): " + str(sentimentOfDataFile) + ", Overall the whole text is " + classifyscore(sentimentOfDataFile['compound']) + " even when passed to the sentiment analyzer as a block of text.")


if __name__ == '__main__':
    main()
