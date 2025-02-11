load wordnet:
```shell
unzip -o -d /root/corpora/nltk_data wordnet.zip
```

eval:
```python
import nltk
from nltk.corpus import wordnet
wordnet.synsets('car')
```