# simple-tf-idf
간단한 tf-idf 모듈입니다.   
mecab-ya 라이브러리를 설치 후 사용 가능합니다.   
A simple tf-idf module for text   
use after install mecab-ya   

https://github.com/golbin/node-mecab-ya

## simple-tf-idf.js
```JavaScript
let simple_tfidf = require('./simple-tf-idf')   

let document = [];   
document.push('정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.');   
document.push('소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.');   

simple_tfidf.similarity_test(document);   
```

0번 문서와 다른 문서의 유사도를 구하는 과정을 출력합니다.   
과정을 살펴보기 위한 것이므로 입력은 5개 이하의 짧은 문장들이 좋습니다.   
