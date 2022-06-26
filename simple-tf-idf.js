function tokenizer(document){
    let mecab = require('mecab-ya');
    let tokenized_document = [];
    for(let i in document){
        tokenized_document.push(mecab.nounsSync(document[i]));
    }
    return tokenized_document;
}

function build_bag_of_words(tokenized_document){
    let word_to_index = new Map();
    let total_bow = [];
    let total_document = [];
    let bow = [];
    
    // 하나의 문서로 통합
    for(let index in tokenized_document){
        for(let j in tokenized_document[index]){
            total_document.push(tokenized_document[index][j]);
        }
    }
    console.log('total document : ', total_document);
    
    // 단어에 index 맵핑
    for(let word in total_document){
        if(word_to_index.get(total_document[word]) == null){
            // 처음 등장하는 단어 처리
            word_to_index.set(total_document[word], word_to_index.size);
            total_bow.splice(word_to_index.size - 1, 0, 1);
        }
        else{
            // 재등장하는 단어 처리
            let index = word_to_index.get(total_document[word]);
            total_bow[index] = total_bow[index] + 1;
        }
    }
    
    for(let index in tokenized_document){
        let bow_obj = {};
        let bow_temp = [];
        
        for(let word in tokenized_document[index]){
            let i = word_to_index.get(tokenized_document[index][word]);
            if(bow_temp.length === 0){
                let pair = {
                    index: i,
                    value: 1
                };
                bow_temp.push(pair);
            }
            else{
                let flag = 0;
                for(let k in bow_temp){
                    if(bow_temp[k].index === i){
                        bow_temp[k].value += 1;
                        flag = 1;
                        break;
                    }
                }
                
                if(flag === 0){
                    let pair = {
                        index: i,
                        value: 1
                    };
                    bow_temp.push(pair);
                }
            }
        }
        
        
        bow_obj = {
            id: index,
            bow: bow_temp
        };
        bow.push(bow_obj);
    }
    
    console.log('vocabulary : ', word_to_index);
    console.log('bag of words vectors(term frequency) : ', bow);
    
    return [word_to_index, bow];
}

function get_idf(bow, vocab){
    let df = [];
    df.length = vocab.size;
    df.fill(0);
    
    // df 구하기
    for(let i in bow){// 문서 개수만큼
        for(let j in bow[i].bow){// 문서당 bow의 객체 개수만큼
            df[bow[i].bow[j].index] += 1;
        }
    }
    console.log('document frequency : ', df);

    let idf = [];
    let N = bow.length; // 전체 문서의 수
    idf.length = vocab.size;
    idf.fill(0);
    
    // idf 구하기
    for(let i in idf){
        idf[i] = 1 + Math.log(N / (1 + df[i])); // 자연로그
    }
    console.log('inverse document frequency : ',idf);
    
    return idf;
}

function get_tfidf(bow, idf){
    // tfidf 구하기
    let tfidf = [];
    let tfidf_obj = {};
    let sum = 0;
    
    for(let i in bow){// 문서 개수만큼
        let tfidf_temp = [];
        
        for(let j in bow[i].bow){
            let t = bow[i].bow[j].value * idf[bow[i].bow[j].index];
            let pair ={
                index: bow[i].bow[j].index,
                value: t
            };
            tfidf_temp.push(pair);
        }
        
        // tfidf index 내림차순 정렬    
        tfidf_temp.sort(function(a, b) {
            return a.index - b.index;
        });
        
        tfidf_obj = {
            id: bow[i].id,
            tfidf: tfidf_temp
        };
        sum += tfidf_temp.length;
        tfidf.push(tfidf_obj);
    }
    
    console.log('TF-IDF : ', tfidf);
    
    return tfidf;
}

function cosine_similarity(tfidf){
    //0번 문서와 다른 모든 문서를 비교해서 코사인 유사도를 구함
    let cos_sim = [];
    let normalized_zero = normalize(tfidf[0].tfidf);
    
    for(let i in tfidf){// 전체 문서에 대해
        let scalar_product = 0;
        for(let j in tfidf[0].tfidf){// 0번 문서의 tfidf 개수만큼
            for(let k in tfidf[i].tfidf){// i번 문서의 tfidf 개수만큼

                // 0번 벡터와 i번 벡터의 스칼라곱
                if(tfidf[0].tfidf[j].index === tfidf[i].tfidf[k].index){
                    scalar_product += tfidf[0].tfidf[j].value * tfidf[i].tfidf[k].value;
                    break;
                }
            }
        }
        
        let cos_sim_temp = 0;
        if(scalar_product === 0){
            // 분자가 0이면 코사인 유사도 = 0
            cos_sim_temp = 0;
        }
        else{
            // 분자가 0이 아니면 코사인 유사도 공식 사용
            cos_sim_temp = scalar_product / (normalized_zero * normalize(tfidf[i].tfidf));
            cos_sim_temp = Number(cos_sim_temp.toFixed(5));
        }
        
        let cos_sim_obj = {
            id: tfidf[i].id,
            similarity: cos_sim_temp
        };
        cos_sim.push(cos_sim_obj);
    }

    // 유사도 오름차순 정렬    
    cos_sim.sort(function(a, b) {
        return b.similarity - a.similarity;
    });
    
    // 상위 5개의 bid만 추출
    let top5_cos_sim_id = [];
    for(let i=1; i<6; i++){
        console.log(cos_sim[i]);
        top5_cos_sim_id.push(cos_sim[i].id);
    }
    
    return top5_cos_sim_id;
}

function similarity_test(document){
    console.time('time');
    // 문서 토큰화
    let tokenized_document = tokenizer(document);
    console.log('tokenized_document : ', tokenized_document);
    
    // 모든 단어에 index 맵핑
    let result = build_bag_of_words(tokenized_document);
    let vocab = result[0];
    let bow = result[1];
    
    // 모든 단어의 idf 구하기
    let idf = get_idf(bow, vocab);
    
    // 모든 문서의 tfidf 구하기
    let tfidf = get_tfidf(bow, idf);
    
    // 0번 문서와 나머지 문서의 유사도 검사
    let cos_sim = cosine_similarity(tfidf);
    
    // 유사한 5개 문서 출력
    for(let i in cos_sim){
        console.log('id : ', cos_sim[i], ', ', document[cos_sim[i]]);
    }
    
    console.timeEnd('time');
}

function similarity_test_token(tokenized_document){
    console.time('time');
    // 모든 단어에 index 맵핑
    let result = build_bag_of_words(tokenized_document);
    let vocab = result[0];
    let bow = result[1];
    
    // 모든 단어의 idf 구하기
    let idf = get_idf(bow, vocab);
    
    // 모든 문서의 tfidf 구하기
    let tfidf = get_tfidf(bow, idf);
    
    // 0번 문서와 나머지 문서의 유사도 검사
    let cos_sim = cosine_similarity(tfidf);
    console.timeEnd('time');
}

function normalize(vector){
    // 벡터 정규화 공식
    let sum_square = 0;
    for(let i in vector){
        sum_square += vector[i].value * vector[i].value;
    }
    
    return Math.sqrt(sum_square);
}

function save_document_file(path, document){
    const fs = require('fs');
    fs.writeFile(path, JSON.stringify(document), err => {
        if(err){
            console.error(err);
            return;
        }
        else{
            console.log('document saved in : ', path);
        }
    });
}

function load_document_file(path){
    const fs = require('fs');
    let readData = fs.readFileSync(path);
    return JSON.parse(readData.toString());
}

module.exports = {
    tokenizer,
    build_bag_of_words,
    get_idf,
    get_tfidf,
    cosine_similarity,
    similarity_test,
    similarity_test_token,
    save_document_file,
    load_document_file,
};