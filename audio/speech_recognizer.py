# download stopwords corpus, you need to run it once
import nltk
nltk.download('stopwords')
nltk.download('punkt')
#--------#

import os
import codecs
import speech_recognition as sr
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from audio_process import TMP_AUDIO_FILE_NAME

SPEECH_FILE_NAME = 'texts/speech.txt'
WORDS_FILE_NAME = 'texts/words.txt'
QUESTION_FILE_NAME = 'texts/questions.txt'

if os.path.exists(SPEECH_FILE_NAME):
    os.remove(SPEECH_FILE_NAME)
if (os.path.exists(WORDS_FILE_NAME)):
    os.remove(WORDS_FILE_NAME)

def common_words(a, b):     
    a_set = set(a) 
    b_set = set(b) 
      
    # check length  
    if len(a_set.intersection(b_set)) > 0: 
        return(a_set.intersection(b_set))   
    else: 
        return([]) 

r = sr.Recognizer()
speech = ''
with sr.AudioFile(TMP_AUDIO_FILE_NAME) as source:
    r.adjust_for_ambient_noise(source)
    audio = r.record(source)
try:
    value = r.recognize_google(audio, language='ru-RU') # API call to google speech recognition
    os.remove(TMP_AUDIO_FILE_NAME)
    if str is bytes: 
        result = u'{}'.format(value).encode('utf-8')
    else: 
        result = u'{}'.format(value)
    
    speech = result
    with codecs.open(SPEECH_FILE_NAME, 'a', 'utf-8') as f:
        f.write(result)
        f.write(' ')
        
except sr.UnknownValueError:
    print('UnknownValueError')
except sr.RequestError as e:
    print('{0}'.format(e))


stop_words = set(stopwords.words('russian'))
word_tokens = word_tokenize(speech) # tokenizing sentence
filtered_sentence = [w for w in word_tokens if w not in stop_words] # remove stop words

with codecs.open(WORDS_FILE_NAME, 'w', 'utf-8') as f:
    for w in filtered_sentence:
        f.write(w + ' ')
    
# checking proctor needs to be alerted or not
file = codecs.open(QUESTION_FILE_NAME, 'r', 'utf-8')
data = file.read()
file.close()
stop_words = set(stopwords.words('russian'))   
word_tokens = word_tokenize(data) # tokenizing sentence
filtered_questions = [w for w in word_tokens if w not in stop_words] # remove stop words

cw = common_words(filtered_questions, filtered_sentence)
print('Common words:', len(cw))
print(cw)
