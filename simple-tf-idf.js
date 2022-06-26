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
    
    // �ϳ��� ������ ����
    for(let index in tokenized_document){
        for(let j in tokenized_document[index]){
            total_document.push(tokenized_document[index][j]);
        }
    }
    console.log('total document : ', total_document);
    
    // �ܾ index ����
    for(let word in total_document){
        if(word_to_index.get(total_document[word]) == null){
            // ó�� �����ϴ� �ܾ� ó��
            word_to_index.set(total_document[word], word_to_index.size);
            total_bow.splice(word_to_index.size - 1, 0, 1);
        }
        else{
            // ������ϴ� �ܾ� ó��
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
    
    // df ���ϱ�
    for(let i in bow){// ���� ������ŭ
        for(let j in bow[i].bow){// ������ bow�� ��ü ������ŭ
            df[bow[i].bow[j].index] += 1;
        }
    }
    console.log('document frequency : ', df);

    let idf = [];
    let N = bow.length; // ��ü ������ ��
    idf.length = vocab.size;
    idf.fill(0);
    
    // idf ���ϱ�
    for(let i in idf){
        idf[i] = 1 + Math.log(N / (1 + df[i])); // �ڿ��α�
    }
    console.log('inverse document frequency : ',idf);
    
    return idf;
}

function get_tfidf(bow, idf){
    // tfidf ���ϱ�
    let tfidf = [];
    let tfidf_obj = {};
    let sum = 0;
    
    for(let i in bow){// ���� ������ŭ
        let tfidf_temp = [];
        
        for(let j in bow[i].bow){
            let t = bow[i].bow[j].value * idf[bow[i].bow[j].index];
            let pair ={
                index: bow[i].bow[j].index,
                value: t
            };
            tfidf_temp.push(pair);
        }
        
        // tfidf index �������� ����    
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
    //0�� ������ �ٸ� ��� ������ ���ؼ� �ڻ��� ���絵�� ����
    let cos_sim = [];
    let normalized_zero = normalize(tfidf[0].tfidf);
    
    for(let i in tfidf){// ��ü ������ ����
        let scalar_product = 0;
        for(let j in tfidf[0].tfidf){// 0�� ������ tfidf ������ŭ
            for(let k in tfidf[i].tfidf){// i�� ������ tfidf ������ŭ

                // 0�� ���Ϳ� i�� ������ ��Į���
                if(tfidf[0].tfidf[j].index === tfidf[i].tfidf[k].index){
                    scalar_product += tfidf[0].tfidf[j].value * tfidf[i].tfidf[k].value;
                    break;
                }
            }
        }
        
        let cos_sim_temp = 0;
        if(scalar_product === 0){
            // ���ڰ� 0�̸� �ڻ��� ���絵 = 0
            cos_sim_temp = 0;
        }
        else{
            // ���ڰ� 0�� �ƴϸ� �ڻ��� ���絵 ���� ���
            cos_sim_temp = scalar_product / (normalized_zero * normalize(tfidf[i].tfidf));
            cos_sim_temp = Number(cos_sim_temp.toFixed(5));
        }
        
        let cos_sim_obj = {
            id: tfidf[i].id,
            similarity: cos_sim_temp
        };
        cos_sim.push(cos_sim_obj);
    }

    // ���絵 �������� ����    
    cos_sim.sort(function(a, b) {
        return b.similarity - a.similarity;
    });
    
    // ���� 5���� bid�� ����
    let top5_cos_sim_id = [];
    for(let i=1; i<6; i++){
        console.log(cos_sim[i]);
        top5_cos_sim_id.push(cos_sim[i].id);
    }
    
    return top5_cos_sim_id;
}

function similarity_test(document){
    console.time('time');
    // ���� ��ūȭ
    let tokenized_document = tokenizer(document);
    console.log('tokenized_document : ', tokenized_document);
    
    // ��� �ܾ index ����
    let result = build_bag_of_words(tokenized_document);
    let vocab = result[0];
    let bow = result[1];
    
    // ��� �ܾ��� idf ���ϱ�
    let idf = get_idf(bow, vocab);
    
    // ��� ������ tfidf ���ϱ�
    let tfidf = get_tfidf(bow, idf);
    
    // 0�� ������ ������ ������ ���絵 �˻�
    let cos_sim = cosine_similarity(tfidf);
    
    // ������ 5�� ���� ���
    for(let i in cos_sim){
        console.log('id : ', cos_sim[i], ', ', document[cos_sim[i]]);
    }
    
    console.timeEnd('time');
}

function similarity_test_token(tokenized_document){
    console.time('time');
    // ��� �ܾ index ����
    let result = build_bag_of_words(tokenized_document);
    let vocab = result[0];
    let bow = result[1];
    
    // ��� �ܾ��� idf ���ϱ�
    let idf = get_idf(bow, vocab);
    
    // ��� ������ tfidf ���ϱ�
    let tfidf = get_tfidf(bow, idf);
    
    // 0�� ������ ������ ������ ���絵 �˻�
    let cos_sim = cosine_similarity(tfidf);
    console.timeEnd('time');
}

function normalize(vector){
    // ���� ����ȭ ����
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