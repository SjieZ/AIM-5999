
# Machine Learning Educational Chatbot
This research introduces an innovative question-answering chatbot, merging advanced natural language processing models (Llama2, Mistal, Yi, GPT-2) and NLP techniques. The project aims to enhance the understanding of complex machine learning concepts through an AI-driven, user-friendly platform. The system, built on a Flask backend architecture, seamlessly integrates models for efficient processing. The front-end interface, developed with HTML, CSS, and JavaScript, supports both text and voice inputs, enhancing accessibility. NLTK is employed for text preprocessing, enabling accurate responses to user queries. Preliminary results showcase the chatbot's high proficiency, generating context-sensitive responses with low validation loss. The dual-model approach ensures response diversity, effectively addressing a wide range of queries. This study not only bridges a gap in machine learning education but also lays the groundwork for future AI-driven educational resources, potentially revolutionizing AI education and enhancing students' comprehension of machine learning principles.
# Data
THe whole dataset contains about 300,000 questions and answers pairs about machine learning, and we splitted it randomly into training dataset(70%), validation dataset(10%) and test dataset(20%). 3,000 qa pairs are picked from dataset [[1]](#1), and 5,000 rows of question and answer pairs from SQuAD 2 [[2]](#2). The others are scraped websites, such as Wiki and Geeksforgeeks.
# Model & Finetuning
We used 5 different LLM, which is Llama2-7b-chat, Mistral 7b, Yi-6b, GPT2 and our customized model. The llama2.ipynb (and other files named by models) is the code of finetuning.

## References
<a id="1">[1]</a> 
evang Kulshreshtha, Robert Belfer, Iulian Vlad Serban, and
Siva Reddy. Back-training excels self-training at unsuper-
vised domain adaptation of question generation and passage
retrieval, 2021
<a id="2">[2]</a> 
ranav Rajpurkar, Robin Jia, and Percy Liang. Know what
you don’t know: Unanswerable questions for squad, 2018.
