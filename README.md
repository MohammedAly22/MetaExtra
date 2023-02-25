# MetaExtra
"MetaExtra" is an NLP application used for extracting meta-data for any file including title, publisher, author, and keywords of a text file.

### Dataset
link: https://data.mendeley.com/datasets/57zpx667y9/2
the dataset is an arabic-based database consists of thousands of text files in 5 categories: (finance, medicine, sports, politics, technology).

### Used packages:
  - re
  - numpy
  - pandas
  - sklearn
  - NLTK
  - streamlit

### Modules
MetaExtra consists of 2 modules: preprocessing module, main module.

### Preprocessing module:
responsible for all preprocessing areas of our file, it is the actual engine of extracting the meta-data.

### Main module:
it's used for deployment purposes and the actual using of the preprocessing class.

### Example usage:
```python
from preprocessing import Preprocessor

preprocessor = Preprocessor(corpus)
```
