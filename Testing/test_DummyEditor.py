"""
Created 7 May 2023
@author: Dimitris Lymperopoulos
Description: A script to test the functionality of the DummyEditor class

Usage:
1)  Obtain information about what each method of the editor returns, and how the data are
    being represented by giving a sample term to search, using a default sentence from which
    another sentence will be created
    python3 test_DummyEditor.py


Example:
    python3 test_DummyEditor.py
"""

from datetime import datetime
from Editors.DummyEditor import DummyEditor


def main():
    start_time = datetime.now()

    editor = DummyEditor("adjective")
    sentence = "A beautiful movie with great plot and interesting characters!"
    indicative_sentence = "A [BLANK] movie with [BLANK] plot and [BLANK] characters!"

    print("ORIGINAL SENTENCE:")
    print(sentence)

    counter_sentence = editor.generate_counterfactual(sentence, indicative_sentence, word_similarity='wordnet')
    print("\nGENERATED SENTENCE WITH WORDNET_SIMILARITY:")
    print(counter_sentence)

    counter_sentence = editor.generate_counterfactual(sentence, indicative_sentence, word_similarity='spacy')
    print("\nGENERATED SENTENCE WITH SPACY_SIMILARITY:")
    print(counter_sentence)

    counter_sentence = editor.generate_counterfactual(sentence, indicative_sentence, word_similarity='mixed')
    print("\nGENERATED SENTENCE WITH MIXED_SIMILARITY:")
    print(counter_sentence)

    counter_sentence = editor.generate_counterfactual(sentence, indicative_sentence, word_similarity='conceptnet')
    print("\nGENERATED SENTENCE WITH CONCEPTNET_SIMILARITY:")
    print(counter_sentence)

    print("\n\nScript execution time: " + str(datetime.now()-start_time))


if __name__ == "__main__":
    main()
