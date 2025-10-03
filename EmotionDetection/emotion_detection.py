import requests
import json


def emotion_detector(text_to_analyze):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'

    myobj = {"raw_document": {"text": text_to_analyze}}

    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    response = requests.post(url, json=myobj, headers=header)

    #Prints status code to server logs
    print(f"status_code: {response.status_code}")

    # Returns a 400 if the entry is blank
    if response.status_code == 400:
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    formatted_response = json.loads(response.text)

    # Obtaining emotion scores in the response
    emotions = get_emotions(formatted_response)
    
    # Extracting emotion scores
    anger_score = emotions.get('anger', 0)
    disgust_score = emotions.get('disgust', 0)
    fear_score = emotions.get('fear', 0)
    joy_score = emotions.get('joy', 0)
    sadness_score = emotions.get('sadness', 0)

    # Creating a dictionary to store the results
    emotion_dict = {
        'anger': anger_score,
        'disgust': disgust_score,
        'fear': fear_score,
        'joy': joy_score,
        'sadness': sadness_score
    }

    # Locating dominant emotion
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)

    emotion_dict['dominant_emotion'] = dominant_emotion

    return emotion_dict


def get_emotions(data):
    """
    Recursive lookup on all emotion keys
    """
    emotion_keys = {'anger', 'disgust', 'fear', 'joy', 'sadness'}

    if isinstance(data, dict):
        # Checking if dict has emotion keys
        if emotion_keys.issubset(data.keys()):
            return data

        for value in data.values():
            result = get_emotions(value)
            if result:
                return result

    elif isinstance(data, list):
        # Watson NLP returns its data as a list,
        # so same as above but for lists
        for item in data:
            result = get_emotions(item)
            if result:
                return result

    return {}
