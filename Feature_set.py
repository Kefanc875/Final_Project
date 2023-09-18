from collections import Counter
import re
import Clean_text

def calculate_lexical_density_with_pos(corpus):
    total_words = 0
    total_content_words = 0
    
    for sentence in corpus:
        words_with_pos = sentence.split() 
        words = []
        pos_tags = []
        
        for word_with_pos in words_with_pos:
            parts = word_with_pos.split('_')
            if len(parts) == 2: 
                words.append(parts[0])
                pos_tags.append(parts[1])
                
        for i in range(len(words)):
            if re.match(r'[\u4e00-\u9fa5]', words[i]):
                total_words += 1
    
        for i in range(len(words)):
            if pos_tags[i].startswith('a') or pos_tags[i].startswith('v') or pos_tags[i].startswith('n'): 
                if re.match(r'[\u4e00-\u9fa5]', words[i]):
                    total_content_words += 1
    
    lexical_density = total_content_words / total_words if total_words > 0 else 0
    return lexical_density

def calculate_sttr(text):
    total_tokens = 0
    total_types = 0

    for sentence in text:
        words = [word.split('_')[0] for word in sentence.split() if bool(re.fullmatch(r'[\u4e00-\u9fa5]+', word.split('_')[0]))]

        total_tokens += len(words)
        total_types += len(set(words))

    sttr = total_types / total_tokens
    return sttr

def calculate_passive_voice_frequency_with_pos(corpus):
    total_verbs = 0
    passive_verbs = 0
    
    for sentence in corpus:
        words_with_pos = sentence.split()  
        verbs = []
        
        for word_with_pos in words_with_pos:
            parts = word_with_pos.split('_')
            if len(parts) == 2 and parts[1].startswith('v'):  
                verbs.append(parts[0])
            if "被" in parts[0]:
                passive_verbs += 1
                
        total_verbs += len(verbs)
    
    passive_voice_frequency = passive_verbs / total_verbs if total_verbs > 0 else 0
    passive_voice_frequency = 6 * passive_voice_frequency
    return passive_voice_frequency

def calculate_top_chinese_words_with_pos(text_blocks, top_n):
   
    all_sentences = [sentences for text_block in text_blocks for sentences in text_block]
    
    words_with_pos = [word_pos.split('_')[0] for sentence in all_sentences for word_pos in sentence.split()]
    
    chinese_words = [word for word in words_with_pos if re.match(r'^[\u4e00-\u9fa5]+$', word)]
    
    word_counts = Counter(chinese_words)
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_words = [word for word, count in sorted_words[:top_n]]
    
    return top_words

top_words = calculate_top_chinese_words_with_pos(Clean_text.all_corpus, 5)

def calculate_top_words_frequency(corpus_block, top_words):
    merged_text = " ".join(corpus_block)
    words_with_pos = merged_text.split()

    words = [word_pos.split('_')[0] for word_pos in words_with_pos]

    chinese_words = [word for word in words if re.match(r'^[\u4e00-\u9fa5]+$', word)]

    word_frequency_vector = [chinese_words.count(word) / len(chinese_words) for word in top_words]

    min_frequency = min(word_frequency_vector)
    max_frequency = max(word_frequency_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in word_frequency_vector]
    
    return normalized_frequency_vector

cohesive_markers = ["因此", "然而", "此外", "另外", "与此同时", "首先", "其次", "最后", "其中", "反之",
                    "换句话说", "那么", "因为", "所以", "但是", "虽然", "即使", "同样", "尽管", "而且",
                    "不仅如此", "例如", "因为此", "总之", "而是", "那么", "通常", "一方面", "另一方面",
                    "与其说", "由于", "至于", "另一方面", "此外", "鉴于", "比如", "具体来说", "无论如何",
                    "也就是说", "甚至", "然后", "然后", "当然", "突然", "最初", "综上所述", "首先", "最后但并非最不重要", "大致上", "随着"]

def calculate_cohesive_markers_frequency(corpus_block, cohesive_markers):
    merged_text = " ".join(corpus_block)
    words_with_pos = merged_text.split()

    words = [word_pos.split('_')[0] for word_pos in words_with_pos]

    chinese_words = [word for word in words if re.match(r'^[\u4e00-\u9fa5]+$', word)]

    word_frequency_vector = [chinese_words.count(word) / len(chinese_words) for word in cohesive_markers]

    min_frequency = min(word_frequency_vector)
    max_frequency = max(word_frequency_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in word_frequency_vector]
    
    return normalized_frequency_vector

function_words = ["和", "或", "但是", "因为", "所以", "虽然", "也", "就", "在", "与","又", "而", "只是", "即使", "才", "非常", "已经", "刚刚", "曾经", "现在",
    "一直", "这样", "那么", "一定", "不过", "例如", "比如", "由于", "至于", "另外","此外", "于是", "还是", "不仅", "甚至", "亦即", "不管", "无论", "如果", "所谓",
    "另一方面", "当然", "只要", "无论如何", "每当", "为了", "对于", "除了", "其中", "以及","一样", "每个", "自己", "一些", "每天", "几乎", "一般", "其实", "这个", "那个",
    "这些", "那些", "不同", "同时", "当时", "然后", "后来", "以前", "随着", "例如","这里", "那里", "哪里", "什么", "怎么", "为什么", "总是", "差不多", "只有", "不必",
    "然而", "并且", "一起", "否则", "原来", "似乎", "即将", "极了", "其实", "哪怕","顺便", "简直", "看来", "想来", "正是", "此时", "那时", "看见", "快要", "几乎",
    "最后", "未来", "自从", "近年", "刚才", "一直", "何时", "几时", "到底", "即便","如此", "然而", "再者", "并且", "此外", "此时", "而且", "或者", "如果", "假如",
    "要是", "由于", "而言", "鉴于", "这么", "那么", "不如", "不妨", "以便", "所以","因此", "总之", "具体", "换句话说", "比方说", "按照", "依照", "除非", "除了", "至于",
    "不论", "也就是说", "另外", "然后", "一般来说", "例如", "总的来看", "此外", "反之", "否则","与此同时", "而且",
]

def calculate_function_words_frequency(corpus_block, function_words):
    merged_text = " ".join(corpus_block)
    words_with_pos = merged_text.split()

    words = [word_pos.split('_')[0] for word_pos in words_with_pos]

    chinese_words = [word for word in words if re.match(r'^[\u4e00-\u9fa5]+$', word)]

    word_frequency_vector = [chinese_words.count(word) / len(chinese_words) for word in function_words]

    min_frequency = min(word_frequency_vector)
    max_frequency = max(word_frequency_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in word_frequency_vector]
    
    return normalized_frequency_vector

def calculate_average_sentence_length(text_block):
    total_chinese_word_count = 0
    total_sentences = len(text_block)
    
    for sentence in text_block:
        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', sentence)
        chinese_word_count = len(chinese_words)
        total_chinese_word_count += chinese_word_count

    if total_sentences > 0:
        average_length = total_chinese_word_count / total_sentences
        return average_length
    else:
        return 0
    
def caculate_lexical_function_word_ratio(text_block):
    lexical_word_count = 0
    function_word_count = 0

    function_word_pos_tags = {"u", "c", "d", "e", "h", "k", "dg", "m", "p", "q", "r", "t", "f"}
    words_with_pos = []

    for sentence in text_block:
        word_pos_pairs = [word.split('_') for word in sentence.split()]
        for word in word_pos_pairs:
            if re.match(r'[\u4e00-\u9fa5]+', word[0]):
                words_with_pos.append(word)
            
    for word in words_with_pos:
        if word[1] in function_word_pos_tags:
            function_word_count += 1
        elif word[1].startswith('a') or word[1].startswith('v') or word[1].startswith('n') or word[1].startswith('j'):
            lexical_word_count += 1
            
    lexical_function_word_ratio = lexical_word_count / function_word_count
    return lexical_function_word_ratio

chinese_pronouns = ["我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们", "自己", "您", "谁", "哪个", "哪些", "这个", "那个", "这些", "那些", "其"]

def caculate_pronouns_frequency(text_block, chinese_pronouns):
    merged_text = " ".join(text_block)
    words_with_pos = merged_text.split()

    words = [word_pos.split('_')[0] for word_pos in words_with_pos]

    chinese_words = [word for word in words if re.match(r'^[\u4e00-\u9fa5]+$', word)]

    word_frequency_vector = [chinese_words.count(word) / len(chinese_words) for word in chinese_pronouns]

    min_frequency = min(word_frequency_vector)
    max_frequency = max(word_frequency_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in word_frequency_vector]
    
    return normalized_frequency_vector

def caculate_positional_token_frequency(sentence):
    words_len, word_counts = Clean_text.caculate_words_len_and_words_counts(Clean_text.all_sentence)
    frequency_vectors = []
    merged_text = " ".join(sentence)
    chinese_words_with_pos = re.findall(r'([\u4e00-\u9fa5]+_[a-zA-Z]+)', merged_text)
    chinese_words = [word.split('_')[0] for word in chinese_words_with_pos]

    first_word_frequency = word_counts[chinese_words[0]] / words_len
    second_word_frequency = word_counts[chinese_words[1]] / words_len
    third_word_frequency = word_counts[chinese_words[2]] / words_len

    last_word_frequency = word_counts[chinese_words[-1]] / words_len
    second_to_last_word_frequency = word_counts[chinese_words[-2]] / words_len 

    frequency_vector = [first_word_frequency, second_word_frequency, third_word_frequency, last_word_frequency, second_to_last_word_frequency]
        
    min_frequency = min(frequency_vector)
    max_frequency = max(frequency_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in frequency_vector]
    return normalized_frequency_vector

def caculate_punctuation_frequency(text_block, chinese_punctuation, target_punctuation):
    merged_text = " ".join(text_block)
    words_with_pos = merged_text.split()

    words = [word_pos.split('_')[0] for word_pos in words_with_pos]

    chinese_punctuations = [word for word in words if word in chinese_punctuation]

    punctuation_frequency_vector = [chinese_punctuations.count(word) / len(chinese_punctuations) for word in target_punctuation]
    return punctuation_frequency_vector

def find_top_pos_bigrams(corpus, top_n):
    word_pos_triples = []
    for text_block in corpus:
        for sentence in text_block:
            triples = sentence.split()  # Split the sentence into word_pos pairs
            for i in range(len(triples) - 1):
                valid_triples = []
                for word_pos in triples[i:i + 2]:
                    if "_PUNCT" not in word_pos:
                        word = word_pos.split("_")[0]  # Extract the word part
                        if re.match(r'^[\u4e00-\u9fa5]+$', word):  # Check if it contains only Chinese characters
                            valid_triples.append(word_pos)
                if len(valid_triples) == 2:
                    word_pos_triple = "_".join(valid_triples)  # Create a POS 3-gram
                    word_pos_triples.append(word_pos_triple)

    triple_freq = Counter(word_pos_triples)

    top_triples = triple_freq.most_common(top_n)
    top_triples = [triple[0] for triple in triple_freq.most_common(top_n)]
    
    return top_triples

top_bigrams = find_top_pos_bigrams(Clean_text.all_corpus, 100)

def calculate_pos_bigrams_feature(text, top_bigrams):
    word_pos_triples = []
    for sentence in text:
        triples = sentence.split()  # Split the sentence into word_pos pairs
        for i in range(len(triples) - 1):
            valid_triples = []
            for word_pos in triples[i : i + 2]:
                if "_PUNCT" not in word_pos:
                    word = word_pos.split("_")[0]  # Extract the word part
                    if re.match(r'^[\u4e00-\u9fa5]+$', word):  # Check if it contains only Chinese characters
                        valid_triples.append(word_pos)
            if len(valid_triples) == 2:
                word_pos_triple = "_".join(valid_triples)  # Create a POS 3-gram
                word_pos_triples.append(word_pos_triple)
            
    # Count the occurrence of each top triple in the text
    top_triple_counts = Counter(word_pos_triples)
    # Create a list to store the counts of each top triple
    top_triple_occurrences = [top_triple_counts[triple] for triple in top_bigrams]
    
    # Count the occurrence of each triple in the text
    triple_counts = Counter(word_pos_triples)
        
    # Calculate the normalized frequency for each top triple
    POS_trigarms_frequencies_vector = [word_pos_triples.count(triple) / len(word_pos_triples) if len(word_pos_triples) > 0 else 0 for triple in top_bigrams]

    min_frequency = min(POS_trigarms_frequencies_vector)
    max_frequency = max(POS_trigarms_frequencies_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in POS_trigarms_frequencies_vector]

    return normalized_frequency_vector


def find_top_pos_triples(corpus, top_n):
    word_pos_triples = []
    for text_block in corpus:
        for sentence in text_block:
            triples = sentence.split()  # Split the sentence into word_pos pairs
            for i in range(len(triples) - 2):
                valid_triples = []
                for word_pos in triples[i:i + 3]:
                    if "_PUNCT" not in word_pos:
                        word = word_pos.split("_")[0]  # Extract the word part
                        if re.match(r'^[\u4e00-\u9fa5]+$', word):  # Check if it contains only Chinese characters
                            valid_triples.append(word_pos)
                if len(valid_triples) == 3:
                    word_pos_triple = "_".join(valid_triples)  # Create a POS 3-gram
                    word_pos_triples.append(word_pos_triple)

    triple_freq = Counter(word_pos_triples)

    top_triples = triple_freq.most_common(top_n)
    top_triples = [triple[0] for triple in triple_freq.most_common(top_n)]
    
    return top_triples

top_triples = find_top_pos_triples(Clean_text.all_corpus, 300)

def calculate_pos_triples_feature(text, top_triples):
    word_pos_triples = []
    for sentence in text:
        triples = sentence.split()  # Split the sentence into word_pos pairs
        for i in range(len(triples) - 2):
            valid_triples = []
            for word_pos in triples[i : i + 3]:
                if "_PUNCT" not in word_pos:
                    word = word_pos.split("_")[0]  # Extract the word part
                    if re.match(r'^[\u4e00-\u9fa5]+$', word):  # Check if it contains only Chinese characters
                        valid_triples.append(word_pos)
            if len(valid_triples) == 3:
                word_pos_triple = "_".join(valid_triples)  # Create a POS 3-gram
                word_pos_triples.append(word_pos_triple)
            
    # Count the occurrence of each top triple in the text
    top_triple_counts = Counter(word_pos_triples)
    # Create a list to store the counts of each top triple
    top_triple_occurrences = [top_triple_counts[triple] for triple in top_triples]
    
    # Count the occurrence of each triple in the text
    triple_counts = Counter(word_pos_triples)
        
    # Calculate the normalized frequency for each top triple
    POS_trigarms_frequencies_vector = [word_pos_triples.count(triple) / len(word_pos_triples) if len(word_pos_triples) > 0 else 0 for triple in top_triples]

    min_frequency = min(POS_trigarms_frequencies_vector)
    max_frequency = max(POS_trigarms_frequencies_vector)
    normalized_frequency_vector = [(frequency - min_frequency) / (max_frequency - min_frequency) if max_frequency != min_frequency else 0 for frequency in POS_trigarms_frequencies_vector]

    return normalized_frequency_vector















