def sort_sentence(sentence: str) -> str:

    import string
    translator = str.maketrans('', '', string.punctuation)
    clean_sentence = sentence.translate(translator)
    
   
    words = clean_sentence.split()
    words.sort(key=str.lower) 
    
 
    return " ".join(words)


sentence = "Python is, powerful and easy to learn!"
print(sort_sentence(sentence))