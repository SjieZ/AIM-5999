from flask import Flask, render_template, request, jsonify, make_response
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import speech_recognition as sr
import uuid
import time
import threading
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer
from gtts import gTTS
import json
import os
import pandas as pd

base_model = "/mnt/c/Users/shengjie zhao/Desktop/UI1/model/Llama2_model1"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
    # llm_int8_enable_fp32_cpu_offload=False
)
llama_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1
for i in llama_model.named_parameters():
    print(f"{i[0]} -> {i[1].device}")


llama_tokenizer = AutoTokenizer.from_pretrained(base_model)
llama_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
llama_tokenizer.padding_side = "right"
llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_pipe = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
pronouns = ["it", "they", "he", "she", "this", "these", "those", "him", "her", "them"]
#llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


#llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

try:
    print("Loading tokenizer...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("/mnt/c/Users/shengjie zhao/Desktop/UI1/model/GPT2_model_new")
    print("Tokenizer loaded.")

    print("Loading model...")
    gpt2_model = GPT2LMHeadModel.from_pretrained("/mnt/c/Users/shengjie zhao/Desktop/UI1/model/GPT2_model_new").to(device)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading the model: {e}")

app = Flask(__name__)

sessions = {}

def extract_keywords(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in tokens]
    filtered_tokens = [i for i in lemmatized_tokens if i not in stop_words]
    return ' '.join(filtered_tokens)
class QA_Model:
    def __init__(self, data_path=None, gpt2_tokenizer=None, gpt2_model=None, llama_tokenizer=None, llama_model=None, device=None):
        if data_path:
            self.dataset = pd.read_json(data_path, encoding='utf-8')
            self.dataset_keywords = [extract_keywords(row['Question']) for _, row in self.dataset.iterrows()]
            self.vectorizer = TfidfVectorizer()
            self.dataset_vectors = self.vectorizer.fit_transform(self.dataset_keywords)
        else:
            self.dataset = None
        
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.llama_tokenizer = llama_tokenizer
        self.llama_model = llama_model
        self.device = device
    def is_new_context(self, current_input, previous_input):
        contains_pronoun = any(pronoun in current_input.lower() for pronoun in pronouns)
        current_keywords = extract_keywords(current_input)
        previous_keywords = extract_keywords(previous_input)
        vectorized_current = self.vectorizer.transform([current_keywords])
        vectorized_previous = self.vectorizer.transform([previous_keywords])
        similarity = cosine_similarity(vectorized_current, vectorized_previous)[0][0]
        
        # print(f"Current Input: {current_input}")
        # print(f"Previous Input: {previous_input}")
        # print(f"Contains pronoun: {contains_pronoun}")
        # print(f"Similarity: {similarity}")
        
        if contains_pronoun and similarity > 0.5:
            return False 
        elif not contains_pronoun and similarity < 0.2:
            return True  
        else:
            return similarity < 0.5
    def find_max_similar(self, user_input, session_id=None):
        if session_id and sessions.get(session_id):
            history = sessions[session_id]['history']
            if self.is_new_context(user_input, history[-1]):
                combined_input = user_input
            else:
                combined_input = " | ".join(history) + " " + user_input   
        else:
            combined_input = user_input
        
        print(f"Processed input: {combined_input}")

        combined_keywords = extract_keywords(combined_input)
        all_keywords = self.dataset_keywords + [combined_keywords]
        vectorizer = TfidfVectorizer()
        all_vectors = vectorizer.fit_transform(all_keywords)

        vectorized_input = all_vectors[-1]
        dataset_vectors = all_vectors[:-1]
        similarities = cosine_similarity(vectorized_input, dataset_vectors)  
        max_similarity = max(similarities[0])
        similar_index = similarities[0].argmax()
        similarity_threshold = 0.5
        print(similarities)
        print(max_similarity)
        is_code_response = "python" in combined_input.lower() or "code" in combined_input.lower()
        if max_similarity < similarity_threshold:
            return ("Sorry, we cannot provide a definitive answer to this question. However, if you could provide<br>more details or further information about the issue, we'll do our best to assist you.", False)
        else:
            response = self.dataset.iloc[similar_index]['Answer']
            return (response, is_code_response)

    def generate_response(self,user_input):
        try:
            print("Generating response...")
            input_ids = gpt2_tokenizer.encode(user_input, return_tensors="pt").to(device)
            attention_mask = torch.tensor([1] * len(input_ids[0]), dtype=torch.long).unsqueeze(0).to(device)
            gpt2_model.eval()

            outputs = gpt2_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                no_repeat_ngram_size=2,  
                pad_token_id=gpt2_tokenizer.eos_token_id
            )

            response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if response.startswith(user_input):
                response = response[len(user_input):].lstrip()

            eos_index = response.find("</s>")
            if eos_index != -1:
                response = response[:eos_index].rstrip()

            print(f"Generated response: {response}")
            return response
        except Exception as e:
            print(f"Error generating the response: {e}")
            return "I'm sorry, I can't process that right now."
    def generate_response_llama(self, user_input):
            try:
                print("Generating LLaMA response...")
                formatted_input = f"<s>[INST]{user_input}[/INST]"
                result = llama_pipe(formatted_input, max_length=100, pad_token_id=llama_tokenizer.eos_token_id)
                response = result[0]['generated_text']

                response = response.replace(formatted_input, "").strip()

                last_period_index = response.rfind('. ')
                if last_period_index != -1:
                    response = response[:last_period_index+1]


                print(f"LLaMA Generated response: {response}")
                return response
            except Exception as e:
                print(f"Error generating the LLaMA response: {e}")
                return "I'm sorry, I can't process that right now."
        
    def answer_question(self, user_input, session_id=None):
        qa_response, is_code_response = None, False
        gpt2_response = None
        
        if self.dataset is not None:
            qa_response, is_code_response = self.find_max_similar(user_input, session_id)
        
        gpt2_response = self.generate_response(user_input)
        
        return {
            'qa_response': qa_response,
            'gpt2_response': gpt2_response,
            'is_code_response': is_code_response
        }


bot = QA_Model(
    data_path='/mnt/c/Users/shengjie zhao/Desktop/UI1/json/t.json')

@app.route('/get_voice', methods=['POST'])
def get_voice():
    session_id = request.cookies.get('session_id')
    print('session_id',session_id)
    if not session_id:
        session_id = str(uuid.uuid4())

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("please speak...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="en-US")
            response, is_code = bot.find_max_similar(text)
            response = response.replace("\n", "<br>")
        except sr.UnknownValueError:
            text = "Sorry, can't understand the words"
            response = text
        except sr.RequestError as e:
            text = f"Sorry, can't request Google Web Speech API；{e}"
            response = text

    if session_id not in sessions:
        sessions[session_id] = {'history': []}
    sessions[session_id]['history'].append(text)
    sessions[session_id]['last_interaction'] = time.time()  # Update the timestamp for session cleanup

    if text == "Sorry, can't understand the words" or ("Sorry, can't request Google Web Speech API" in text):
        flask_response = jsonify({'botresponse': response})
    else:
        response_content = {'userinput': text, 'botresponse': response, 'is_code': is_code}
        flask_response = make_response(jsonify(response_content))
        flask_response.set_cookie('session_id', session_id)  # Send the session_id back to the client

    return flask_response
def clear_old_sessions():
    current_time = time.time()
    timeout = 3600  # 1 hour
    for session_id, session_data in list(sessions.items()):
        if current_time - session_data['last_interaction'] > timeout:
            del sessions[session_id]
    # Call the function again after your desired timeout
    threading.Timer(timeout, clear_old_sessions).start()    
    

@app.route('/text_to_voice', methods=['POST'])
def text_to_voice_route():
    text = request.form.get('text')
    tts = gTTS(text)
    tts.save("static/output.mp3")
    return jsonify({'status': 'done'})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('userinput')
        session_id = request.cookies.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())

        responses = bot.answer_question(user_input, session_id)
        qa_response = responses['qa_response'].replace("\n", "<br>") if responses['qa_response'] else "No matching entry found in QA database."
        gpt2_response = responses['gpt2_response'].replace("\n", "<br>")
        llama_response = bot.generate_response_llama(user_input)

        response_content = {
            'qa_response': qa_response,
            'gpt2_response': gpt2_response,
            'llama_response': llama_response.replace("\n", "<br>"),
            'is_code': responses['is_code_response']
        }

        flask_response = make_response(jsonify(response_content))
        flask_response.set_cookie('session_id', session_id)
        return flask_response

    return render_template('index.html')


if __name__ == "__main__":
    clear_old_sessions()
    app.run(debug=True)


