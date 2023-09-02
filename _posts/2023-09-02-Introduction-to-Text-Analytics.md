---
title : 1. Text Mining Overview
date : 2023-09-02 22:10:00 +09:00
categories : [Unstrucutred Data Analysis Lecture]
tags : [NLP, 고려대학교 강필성 교수, Text Analytics] #소문자만 가능
description: 고려대학교 강필성 교수님 Unstructured Data Analysis 강의 정리
toc: true
toc_sticky: true
toc_label: 목차
math: true
mermaid: true
image: /assets/img/post/output3.png
---

> 고려대학교 강필성 교수님의 Unstructured Data Analysis 강의를 듣고 필기했습니다.
> 혹시 저작권 등의 문제가 발생하면 지우겠습니다 😀

### 1. 한국어 분석의 어려움

![한국어 분석의 어려움](/assets/img/post/1.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

- BOW 방식에 의하면 하나의 문서는 해당하는 문서에 존재하는 단어의 빈도로 표현하게 되어있다. 즉, 특정 임의의 언어의 문서를 표현하기 위해서는 해당하는 언어가 가지고 있는 단어의 총 개수만큼 차원의 수가 필요하다
- 그런데 이 단어의 수가 한국어가 압도적으로 많다. 즉, 어떠한 문서가 들어왔을 때 한국어의 경우 1,100,373 차원이 필요하다는 것

### 2. Text Mining의 어려움

❕**언어 자체가 '모호성'을 띈다**

예를 들면, 컴퓨터는 `APPLE`이 들어왔을 때 `먹는 사과`인지 회사 `애플`인지 구분하기 어렵다

### 3. TM Process

- **Types of Text Analytics**

    1. Document Classification

        a. Grouping and categorizing snippets, paragraphs, or document using data mining classification methods, based on models trained on labeled examples
    2. Document Clustering

        a. Grouping and categorizing terms, snippets, paragraphs or documents using data mining clustering methods
    3. Concept Extraction

        a. Grouping or words and phrases into semantically similar groups
    4. Search and Information Retrieval(IR)

        a. Storage and retrieval of text documents, including search engines and keyword search
    5. Information Extraction(IE)

        a. Identification and extraction of relevant facts and relationships from unstructured texts, the process of making structured  data from unstructured and semi-structured texts
    6. Web Mining

        a. Data and text mining on the internet with a specific focus on the scale and interconnectedness of the web
    7. Natural Language Processing(NLP)

        a. Low-level language processing and understanding tasks
      
        b. Often used synonymously with computational linguistics

- **Text Preprocessing Level**
  
  * Level 0 --> Text : 문서 전체
  * Level 1 --> Sentence
  * Level 2 --> Token

      - 단어, 숫자, 공백 ...
      - 빈번하게 사용되는 단어일수록 중요하지 않은 단어일 확률이 높음
      - <span style="color:violet"> stop-words로 처리하고 나면 부자연스러운 문장 또는 문서로 변하지만 핵심적인 어휘는 살아있기 때문에 Text Analytics 관점에서는 군집화 하는데 문제 없는 데이터가 된다.</span>

      - **Stemming**
      ![Stemming](/assets/img/post/2.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

      - **Lemmmatization**
      ![Lemmatization](/assets/img/post/3.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

      <span style="color:violet">Stemming은 차원의 축소 관점에서 효과적이다(단어의 수가 줄어들기 때문). 반면 Lemmatization은 단어의 품소를 보존하는 관점에서 효과적이다.</span>


- **Text Transformation**

  단어의 나열을 $R^d$의 차원의 관점으로 벡터화 하는 것. 즉, 글자를 숫자화함

  - Bag-of-Words

    ![Bag-of-words](/assets/img/post/4.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

  - TF-IDF

    ![TF-IDF](/assets/img/post/5.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }  

    `the`를 예로들면, `the`는 웬만하면 모든 문서에 등장한다. 따라서 $df(w) = N$이 될 것이고, $log 1 = 0$이 된다. 따라서 `the`의 중요도는 `0`에 가깝게 된다.

    <span style="color:violet">즉, 중요 단어는 특정 문서에만 등장해야하고 전체 문서 기준으로는 적게 등장해야 중요도가 올라가는 것이다</span>

  - one-hot-vector representation

    원-핫 벡터의 단점은 두 단어 사이의 유사성이 보존될 수 없다는 것이다.

    ![one-hot-vector](/assets/img/post/6.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

    `hotel`과 `motel` 간의 유사성이 높음에도 원-핫 벡터 기준으로 내적을 하면 `0`이 나오게 된다.

 - **Word vectors : distributed representation**   

   ![one-hot-vector](/assets/img/post/7.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

    ✅ 벡터화 과정

    1. 단어를 n차원의 실수공간에 맵핑한다
    2. n : 우리가 원하는 vocabulary 사이즈보다 작아야한다. `n < |v|`
    3. 단어를 벡터화 했을 때 `king`과 `queen` 의 차이가 `man`과 `woman`의 차이와 유사하다는 것을 발견.

    여기서 차이는 성별인데, 성별이라는 개념이 방향으로서 표현이 된 것.

    즉, 만약 `woman`이 없더라도, `king`과 `man` 그리고 `queen`의 관계를 통해 `woman`을 구할 수 있다. 

### 4. Dimensionality Reduction

- **Feature subset selection**

  ![one-hot-vector](/assets/img/post/8.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

  - 특정한 목적에 걸맞는 최적의 변수 집합을 선택하는 것
  - 주어져 있는 변수들이 무엇이 중요하고 아닐지를 판단을 해서 down stream 태스크에 알맞는 토큰을 선택하는 것
  - 예를들어, 긍/부정 태스크에서 특정 단어는 긍정에 최적화 되어있음

- **Feature subset extraction**

  ![one-hot-vector](/assets/img/post/9.png){:style="border:1px solid #eaeaea; border-radius: 7px; padding: 0px;" }

  - 주어진 데이터로부터 새로운 데이터를 구축하는 것
  - 이 때, 주어진 데이터 차원이 $d$라고 했을 때, 새로운 데이터의 차원 $d'$은 $d$보다 작아야함. $d > d'$ 
  - <span style="color:violet">즉, 차원이 축소되는 것</span>
  - LSA가 대표적인 방법론

### 참조
---
* [강필성 교수님 강의](https://www.youtube.com/watch?v=Y0zrFVZqnl4&list=PLetSlH8YjIfVzHuSXtG4jAC2zbEAErXWm&index=2/){:target="_blank"}