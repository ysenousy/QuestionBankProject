import os
import gc
import re
import torch
import random
import numpy as np
import pandas as pd
from pptx import Presentation
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
import nltk
import string
import pke
import requests
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Load SpaCy model
import spacy
spacy_model = spacy.load('en_core_web_sm')

# Datamuse API URL
DATAMUSE_API_URL = "https://api.datamuse.com/words"

# Fallback distractors in case Sense2Vec and ConceptNet fail
fallback_distractors = [
    "Data analytics helps organizations make decisions.",
    "Business intelligence is key to strategic planning.",
    "Dashboards provide a visual overview of key metrics.",
    "Reports are used for detailed data analysis.",
    "KPIs are used to measure business performance."
]

# Initialize models
def initialize_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Sense2Vec model
    s2v = Sense2Vec().from_disk(r'C:\Python\Lib\site-packages\sense2vec\tests\data')

    # Load Sentence Transformer model
    sentence_transformer_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v2')

    return s2v, sentence_transformer_model, device

# Create necessary folders
def create_folders(base_directory):
    os.makedirs(os.path.join(base_directory, "Quizzes"), exist_ok=True)
    os.makedirs(os.path.join(base_directory, "Question Bank"), exist_ok=True)
    os.makedirs(os.path.join(base_directory, "Weekly Assignments"), exist_ok=True)

# Read all text from a PowerPoint file
def read_pptx(file_path):
    presentation = Presentation(file_path)
    all_text = []

    for slide in presentation.slides:
        slide_text = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                slide_text.append(paragraph.text)
        all_text.append("\n".join(slide_text))

    return all_text

# Clean input text
def clean_text(text):
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7F]', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Enhanced keyword extraction with stricter filtering
def get_keywords(content):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=content, language='en', spacy_model=spacy_model)
    pos = {'NOUN', 'PROPN'}  # Only focus on meaningful nouns and proper nouns
    stoplist = list(string.punctuation) + stopwords.words('english')
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
    keyphrases = extractor.get_n_best(n=10)
    
    # Filter keywords to remove short, irrelevant, or malformed ones
    filtered_keywords = [
        val[0] for val in keyphrases
        if len(val[0].split()) >= 2 and len(val[0]) > 2 and not any(char.isdigit() for char in val[0])
    ]
    
    return filtered_keywords

# Fetch related terms from Datamuse API for generating distractors
def get_datamuse_related_words(word, max_results=5):
    params = {
        'ml': word,  # means like this word
        'max': max_results
    }
    response = requests.get(DATAMUSE_API_URL, params=params)
    if response.status_code != 200:
        return []
    data = response.json()
    related_words = [item['word'] for item in data]
    return related_words

# Generate distractors using Sense2Vec, ConceptNet, and Datamuse
def sense2vec_get_words(word, s2v, top_n=10):
    word = word.lower()
    word_with_pos = f"{word}|NOUN"
    if word_with_pos in s2v:
        most_similar = s2v.most_similar(word_with_pos, n=top_n)
        return [w[0].split("|")[0] for w in most_similar]
    else:
        return []

# Improved distractor generation with fallback for missing terms
def get_distractors(word, sentence, s2v, sentence_transformer_model, top_n=40):
    # Get distractors from Sense2Vec
    distractors_sense2vec = sense2vec_get_words(word, s2v, top_n)
    
    # Get distractors from Datamuse API
    distractors_datamuse = get_datamuse_related_words(word)

    # Combine all sets of distractors
    distractors = list(set(distractors_sense2vec + distractors_datamuse))
    distractors = [d for d in distractors if d.lower() != word.lower()]

    # Convert distractors to full sentences if necessary
    sentence_distractors = [f"{distractor} plays a role in {word} systems." for distractor in distractors]

    # If no valid distractors, use fallback distractors specific to the domain
    if len(sentence_distractors) < 3:
        sentence_distractors = random.sample(fallback_distractors, 3)
    
    return sentence_distractors[:3]

# Dynamic question templates with flexibility
def generate_dynamic_question_templates(keyword, context=None):
    # Fetch related terms to add variability
    related_terms = get_datamuse_related_words(keyword)
    related_term = random.choice(related_terms) if related_terms else '[related term]'

    # Context-based dynamic templates (if context is provided)
    if context and 'technology' in context:
        templates = [
            f"How is {keyword} used in modern {context} systems?",
            f"What impact does {keyword} have on {context} advancements?",
            f"In what situations is {keyword} applied in the field of {context}?"
        ]
    else:
        # General dynamic templates
        templates = [
            f"What is the role of {keyword} in {related_term}?",
            f"How does {keyword} contribute to achieving {related_term}?",
            f"What key features of {keyword} are important in {related_term}?",
            f"What challenges arise from using {keyword} in {related_term}?",
            f"Why is {keyword} considered essential for {related_term}?"
        ]

    return random.choice(templates)

# Process a single slide (parallel processing)
def process_slide(slide_text, s2v, sentence_transformer_model, question_count, max_questions=20, context=None):
    cleaned_text = clean_text(slide_text)
    imp_keywords = get_keywords(cleaned_text)
    
    slide_questions = []
    for answer in imp_keywords:
        if question_count[0] >= max_questions:  # Stop when 20 questions are reached
            break
        question = generate_dynamic_question_templates(answer, context)
        distractors = get_distractors(answer.capitalize(), question, s2v, sentence_transformer_model)
        if len(distractors) == 0:
            continue

        random.shuffle(distractors)
        options = distractors + [answer]
        random.shuffle(options)
        correct_option = options.index(answer)
        slide_questions.append({
            "Question": question,
            "Option A": options[0],
            "Option B": options[1],
            "Option C": options[2],
            "Option D": options[3],
            "Answer": chr(65 + correct_option)  # Converts to 'A', 'B', 'C', or 'D'
        })
        question_count[0] += 1  # Increment the global question counter

    return slide_questions

# Generate all questions for a PPTX
def generate_questions_for_pptx(pptx_path, output_directory, s2v, sentence_transformer_model, max_questions=20):
    all_questions = []
    question_count = [0]  # Use list to allow modification inside process_slide
    slide_texts = read_pptx(pptx_path)

    # Parallel processing of slides using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda slide_text: process_slide(slide_text, s2v, sentence_transformer_model, question_count, max_questions), slide_texts)
        for slide_questions in results:
            all_questions.extend(slide_questions)
            if question_count[0] >= max_questions:  # Stop once we have enough questions
                break

    # Save to Excel sheet
    if all_questions:
        save_to_excel(all_questions, pptx_path, output_directory)

# Save questions to Excel
def save_to_excel(questions, pptx_path, output_directory):
    df = pd.DataFrame(questions)
    excel_filename = os.path.basename(pptx_path).replace('.pptx', '.xlsx')
    excel_path = os.path.join(output_directory, "Quizzes", excel_filename)
    df.to_excel(excel_path, index=False)
    print(f"Processed {pptx_path} -> {excel_path}")

def merge_all_quizzes(output_directory):
    quizzes_folder = os.path.join(output_directory, "Quizzes")
    question_bank_folder = os.path.join(output_directory, "Question Bank")
    
    # Get all Excel files in the "Quizzes" folder
    all_files = [os.path.join(quizzes_folder, f) for f in os.listdir(quizzes_folder) if f.endswith('.xlsx')]
    
    # Read and concatenate all Excel files into one DataFrame
    merged_df = pd.concat([pd.read_excel(f) for f in all_files], ignore_index=True)
    
    # Create the "Question Bank" folder if it doesn't exist
    os.makedirs(question_bank_folder, exist_ok=True)
    
    # Save the merged DataFrame to the "Question Bank" folder
    merged_file_path = os.path.join(question_bank_folder, 'Merged_Quiz_Question_Bank.xlsx')
    merged_df.to_excel(merged_file_path, index=False)
    
    print(f"Merged quiz saved at: {merged_file_path}")
    return merged_file_path

# Process PowerPoint files and generate quizzes
def process_pptx_files(directory, output_directory, max_questions=20):
    s2v, sentence_transformer_model, _ = initialize_models()

    pptx_files = [os.path.join(root, file) 
                  for root, _, files in os.walk(directory) 
                  for file in files if file.endswith(".pptx")]

    # Create the necessary folders
    os.makedirs(os.path.join(output_directory, "Quizzes"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "Question Bank"), exist_ok=True)

    # Use ThreadPoolExecutor to process each PPTX file in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(lambda pptx_path: generate_questions_for_pptx(pptx_path, output_directory, s2v, sentence_transformer_model, max_questions), pptx_files)

    # Merge all generated quizzes into one file
    merge_all_quizzes(output_directory)

    gc.collect()

# Directory setup
base_directory = r"E:\DPI - Project\Questiones Bank\Data Analytics\PowerBI Engineer\PowerBI Engineer (Revised)"

# Create necessary folders and process files
create_folders(base_directory)
process_pptx_files(base_directory, base_directory, max_questions=20)
