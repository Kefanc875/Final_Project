import os
import re
import random
from collections import Counter

def read_files(data_path):
            file_paths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.txt')]

            modified_contents = []

            for file_path in file_paths: 
                with open(file_path, 'r', encoding='utf-16') as file:
                    lines = file.readlines()

                index_to_remove = None
                for idx, line in enumerate(lines):
                    if line.startswith("<s n=\"001\">"):
                        index_to_remove = idx
                        break

                if index_to_remove is not None:
                    lines = lines[index_to_remove:]
                    
                for line in lines:
                    line = re.sub(r'</?p>|</body>|</cesDoc>', '', line)
                    if line.strip():
                        modified_contents.append(line)
            return modified_contents

def chunk_sentences(sentences, max_chunk_words=2000):
    chunked_sentences = []
    current_chunk = []
    current_chunk_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(re.findall(r'[\u4e00-\u9fa5]+', sentence))
        
        if current_chunk_word_count + sentence_word_count > max_chunk_words:
            chunked_sentences.append(current_chunk)
            current_chunk = []
            current_chunk_word_count = 0
        
        current_chunk.append(sentence)
        current_chunk_word_count += sentence_word_count
    
    if current_chunk:
        chunked_sentences.append(current_chunk)
    
    return chunked_sentences

file_pattern = "F:\Corpus\LCMCv2 WordSmith edition\LCMCv2 WordSmith edition"  
data_set_LCMC = read_files(file_pattern)

file_pattern = "F:\Corpus\ZCTC WordSmith edition\ZCTC WordSmith edition"  
data_set_ZCTC = read_files(file_pattern)

chunked_sentences_LCMC = chunk_sentences(data_set_LCMC)
chunked_sentences_ZCTC = chunk_sentences(data_set_ZCTC)

native_chinese_blocks = chunked_sentences_LCMC
translated_chinese_blocks = chunked_sentences_ZCTC

all_corpus = native_chinese_blocks + translated_chinese_blocks

# add label for each chunk
native_chinese_data = [(block, 0) for block in native_chinese_blocks]
translated_chinese_data = [(block, 1) for block in translated_chinese_blocks]

# merge two list
all_data = native_chinese_data + translated_chinese_data
random.shuffle(all_data)  

def generate_all_sentence(data_set_LCMC, data_set_ZCTC):
    data_set_LCMC_more_than_5 = []
    data_set_ZCTC_more_than_5 = []
    for sentence in data_set_LCMC:
        chinese_words_with_pos = re.findall(r'([\u4e00-\u9fa5]+_[a-zA-Z]+)', sentence)
        chinese_words = [word.split('_')[0] for word in chinese_words_with_pos]
        if len(chinese_words) >= 5:
             data_set_LCMC_more_than_5.append(sentence)

    for sentence in data_set_ZCTC:
        chinese_words_with_pos = re.findall(r'([\u4e00-\u9fa5]+_[a-zA-Z]+)', sentence)
        chinese_words = [word.split('_')[0] for word in chinese_words_with_pos]
        if len(chinese_words) >= 5:
             data_set_ZCTC_more_than_5.append(sentence)
    return data_set_LCMC_more_than_5, data_set_ZCTC_more_than_5
             
data_set_LCMC_more_than_5, data_set_ZCTC_more_than_5 = generate_all_sentence(data_set_LCMC, data_set_ZCTC)
native_chinese_sentence = [(sentence, 0) for sentence in data_set_LCMC_more_than_5]
translated_chinese_sentence = [(sentence, 1) for sentence in data_set_ZCTC_more_than_5]
all_sentence = native_chinese_sentence + translated_chinese_sentence
random.shuffle(all_sentence)

def caculate_words_len_and_words_counts(all_sentence):
    word_counts = Counter()
    sentences = []
    words_len = 0

    for sentence in all_sentence:
        chinese_words_with_pos = re.findall(r'([\u4e00-\u9fa5]+_[a-zA-Z]+)', sentence[0])
        sentences.append(sentence[0])

        chinese_words = [word.split('_')[0] for word in chinese_words_with_pos]
        words_len += len(chinese_words)

        word_counts.update(chinese_words)
    return words_len, word_counts


