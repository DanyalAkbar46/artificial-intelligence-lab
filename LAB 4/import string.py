import string

def remove_punctuations(text: str) -> str:
   
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


sentence = "Hello, world! How's everything going?"
print(remove_punctuations(sentence))