import unittest
from src.data_preprocessing.data_preprocessing import strip_html, remove_punctuation, lemmatize_text

class TestDataPreprocessing(unittest.TestCase):

    def test_strip_html(self):
        html = "<div>Hello, world!</div>"
        self.assertEqual(strip_html(html), "Hello, world!")

    def test_remove_punctuation(self):
        text = "Hello, world!"
        self.assertEqual(remove_punctuation(text), "Hello world")

    def test_lemmatize_text(self):
        text = "The striped bats are hanging on their feet for best"
        lemmatized_text = lemmatize_text(text)
        # Assuming 'bats' is lemmatized to 'bat'
        self.assertIn("bat", lemmatized_text)

if __name__ == '__main__':
    unittest.main()