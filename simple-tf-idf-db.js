// DB데이터 사용할 때
function tokenizer_DB(document){
    let mecab = require('mecab-ya');
    if(document.title){
        document.title = mecab.nounsSync(document.title);
    }
    return document;
}

function build_bag_of_words_DB(tokenized_document){
    let word_to_index = new Map();
    let total_bow = [];
    let total_document = [];
    let bow = [];
    
    // 하나의 문서로 통합
    console.time('make total document');
    for(let index in tokenized_document){
        for(let j in tokenized_document[index].title){
            total_document.push(tokenized_document[index].title[j]);
        }
    }
    //console.log('total document : ', total_document);
    console.timeEnd('make total document');
    
    // 단어에 index 맵핑
    console.time('make vocabulary dictionary');
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
    //console.log('vocabulary : ', word_to_index);
    console.timeEnd('make vocabulary dictionary');
    
    // 개별 문서의 BOW 구하기(tf 구하기)
    console.time('make bow');
    for(let index in tokenized_document){
        let bow_obj = {};
        let bow_temp = [];
        
        for(let word in tokenized_document[index].title){
            let i = word_to_index.get(tokenized_document[index].title[word]);
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
            bid: tokenized_document[index].bid,
            bow: bow_temp
        };
        bow.push(bow_obj);
    }
    console.timeEnd('make bow');
    
    return [word_to_index, bow];
}


function get_idf_DB(bow, vocab){
    let df = [];
    df.length = vocab.size;
    df.fill(0);
    
    // df 구하기
    console.time('make df');
    for(let i in bow){// 문서 개수만큼
        for(let j in bow[i].bow){// 문서당 bow의 객체 개수만큼
            df[bow[i].bow[j].index] += 1;
        }
    }
    //console.log('document frequency : ', df);
    console.timeEnd('make df');

    let idf = [];
    let N = bow.length; // 전체 문서의 수
    idf.length = vocab.size;
    idf.fill(0);
    
    // idf 구하기
    console.time('make idf');
    for(let i in idf){
        idf[i] = 1 + Math.log(N / (1 + df[i])); // 자연로그
    }
    //console.log('inverse document frequency : ',idf);
    console.timeEnd('make idf');
    
    return idf;
}

function get_tfidf_DB(bow, idf){
    // tfidf 구하기
    console.time('make tfidf');
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
            bid: bow[i].bid,
            tfidf: tfidf_temp
        };
        sum += tfidf_temp.length;
        tfidf.push(tfidf_obj);
    }
    console.timeEnd('make tfidf');
    
    return tfidf;
}

function cosine_similarity_DB(tfidf){
    //0번 문서와 다른 모든 문서를 비교해서 코사인 유사도를 구함
    console.time('cosine similarity test');
    let cos_sim = [];
    let normalized_zero = normalize_DB(tfidf[0].tfidf);
    
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
            cos_sim_temp = scalar_product / (normalized_zero * normalize_DB(tfidf[i].tfidf));
            cos_sim_temp = Number(cos_sim_temp.toFixed(5));
        }
        
        let cos_sim_obj = {
            bid: tfidf[i].bid,
            similarity: cos_sim_temp
        };
        cos_sim.push(cos_sim_obj);
    }

    // 유사도 오름차순 정렬    
    cos_sim.sort(function(a, b) {
        return b.similarity - a.similarity;
    });
    
    // 상위 5개의 bid만 추출
    let top5_cos_sim_bid = [];
    //console.log('cosine_similarity : ', cos_sim[0]);
    for(let i=1; i<6; i++){
        console.log(cos_sim[i]);
        top5_cos_sim_bid.push(cos_sim[i].bid);
    }
    //console.log('cosine_similarity : ', cos_sim);
    
    console.timeEnd('cosine similarity test');
    return top5_cos_sim_bid;
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


function normalize_DB(vector){
    // 벡터 정규화 공식
    let sum_square = 0;
    for(let i in vector){
        sum_square += vector[i].value * vector[i].value;
    }
    
    return Math.sqrt(sum_square);
}


module.exports = {
    tokenizer_DB,
    build_bag_of_words_DB,
    get_idf_DB,
    get_tfidf_DB,
    cosine_similarity_DB,
    save_document_file,
    load_document_file,
};