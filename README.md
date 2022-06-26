# simple-tf-idf
JavaScript에서 사용할 수 있는 간단한 tf-idf 모듈입니다.   
mecab-ya 라이브러리를 설치 후 사용 가능합니다.   
A simple tf-idf module for text   
use after install mecab-ya   

https://github.com/golbin/node-mecab-ya

## simple-tf-idf-test.js
```JavaScript
let simple_tfidf_test = require('./simple-tf-idf-test')

let document = [];
document.push('정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.');
document.push('소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.');

simple_tfidf_test.similarity_test(document);
```

0번 문서와 다른 문서의 유사도를 구하는 과정을 출력합니다.   
과정을 살펴보기 위한 것이므로 입력은 5개 이하의 짧은 문장들이 좋습니다.   
mecab으로 문서에서 명사만 추출한 후 사용했습니다.   


## simple-tf-idf.js
```JavaScript
let simple_tfidf = require('./simple-tf-idf')

simple_tfidf.similarity_test(document);
```

0번 문서와 다른 문서를 비교해서 유사도가 높은 5개 문서를 출력합니다.   
문서의 크기와 갯수가 커질수록 느려집니다. 또한 하나의 문서를 토큰화하는 시간이 4ms로, 1000개 문서의 경우 4s가 걸립니다.   


```JavaScript
let simple_tfidf = require('./simple-tf-idf')

let tokenized_document = simpleTfidf.load_document_file('your_tokenized_file_path');

simple_tfidf.similarity_test_token(tokenized_document);
```

토큰화한 데이터를 파일에 저장해서 사용한다면 시간이 많이 줄어듭니다.   
토큰화 파일을 사용하면 10k 개의 문장을 비교하는데 200~300ms가 걸립니다.   