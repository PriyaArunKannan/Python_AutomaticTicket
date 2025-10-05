# utils.py
import re

def basic_clean(text: str) -> str:
    """
    Clean and preprocess a given text by converting it to lowercase, 
    removing non-printable characters, and stripping unnecessary punctuation.
    
    This function performs the following operations:
    - Converts the text to lowercase to ensure uniformity.
    - Replaces non-printable characters (such as newlines, tabs, etc.) with a single space.
    - Removes any non-alphanumeric characters except for common punctuation (.,!?/:;'"() and space).
    - Strips extra spaces between words to ensure a single space between each word.

    Args:
        - text (str): The input text string to be cleaned.

    Returns:
        - str: The cleaned text string.
    
    Example:
        >>> basic_clean("Hello, World! \n\t This is an example 1234.")
        'hello world this is an example 1234'
    """
    # Check if the input text is a valid string
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-printable characters like newlines, tabs, etc.
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Remove any non-alphanumeric characters except common punctuation and spaces
    text = re.sub(r'[^0-9a-zA-Z.,!?/:;\'\"()\s-]', '', text)
    
    # Remove extra spaces between words
    text = " ".join(text.split())
    
    return text
