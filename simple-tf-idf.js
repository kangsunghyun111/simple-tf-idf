function tokenizer(document){
    let mecab = require('mecab-ya');
    let tokenized_document = mecab.nounsSync(document);
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
        let bow_temp = [];
        bow_temp.length = word_to_index.size;
        bow_temp.fill(0);
        
        // 개별 문서의 BOW 구하기(tf 구하기)
        for(let word in bow_temp){
            let i = word_to_index.get(tokenized_document[index][word]);
            bow_temp[i] = bow_temp[i] + 1;
        }
        
        // NaN 제거
        bow_temp = bow_temp.filter(function(item){
          return item !== null && item !== undefined && item !== '';
        });
        
        bow.push(bow_temp);
    }
    
    console.log('vocabulary : ', word_to_index);
    console.log('bag of words vectors(term frequency) : ', bow);
    
    return [word_to_index, bow];
}

function get_idf(bow){
    let df = [];
    df.length = bow[0].length;
    df.fill(0);
    
    // df 구하기
    for(let i in df){
        for(let index in bow){
            if(bow[index][i] !== 0){
                df[i] += 1;
            }
        }
    }
    console.log('document frequency : ', df);

    let idf = [];
    let N = bow.length; // 전체 문서의 수
    idf.length = bow[0].length;
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
    
    for(let i in bow){
        let tfidf_temp = [];
        tfidf_temp.length = bow[0].length;
        tfidf_temp.fill(0);
        
        for(let j in bow[0]){
            tfidf_temp[j] = bow[i][j] * idf[j];
        }
        
        tfidf.push(tfidf_temp);
    }
    console.log('TF-IDF : ', tfidf);
    
    return tfidf;
}

function cosine_similarity(tfidf){
    //0번 문서와 다른 모든 문서를 비교해서 코사인 유사도를 구함
    let cos_sim = [];
    let normalized_zero = normalize(tfidf[0]);
    
    for(let i in tfidf){
        let scalar_product = 0;
        for(let j in tfidf[i]){
            // 0번 벡터와 i번 벡터의 스칼라곱
            scalar_product += tfidf[0][j] * tfidf[i][j];
        }
        
        let cos_sim_temp = 0;
        if(scalar_product === 0){
            // 분자가 0이면 코사인 유사도 = 0
            cos_sim_temp = 0;
        }
        else{
            // 분자가 0이 아니면 코사인 유사도 공식 사용
            cos_sim_temp = scalar_product / (normalized_zero * normalize(tfidf[i]));
            cos_sim_temp = Number(cos_sim_temp.toFixed(5));
        }
        
        let cos_sim_obj = {
            index: i,
            similarity: cos_sim_temp
        };
        cos_sim.push(cos_sim_obj);
    }

    // 유사도 오름차순 정렬    
    cos_sim.sort(function(a, b) {
        return b.similarity - a.similarity;
    });
    
    console.log('cosine_similarity : ', cos_sim);
    
    return cos_sim;
}

function normalize(vector){
    // 벡터 정규화 공식
    let sum_square = 0;
    for(let i in vector){
        sum_square += vector[i] * vector[i];
    }
    
    return Math.sqrt(sum_square);
}

module.exports = {
    tokenizer,
    build_bag_of_words,
    get_idf,
    get_tfidf,
    cosine_similarity,
};