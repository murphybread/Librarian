## 프로젝트 개요
보유하고 있는 데이터를 이용하여 LLM과 결합하여 개인 사서 RAG시스템을 목표로 시작한 프로젝트입니다. 나중에 스스로에게 물어볼때나 블로그 글을 전부 읽지 않더라고 나에대해 궁금한 사람들을 대신 대답해주기 위한 목적

설명을 덧 붙이자면 Chatgpt와 같은  LLM은 기존의 학습된 내용이외에 대답을 못하고 웹서치나, 파인튜닝등과 같은 여러 기술을 요구하는데 그중에서도 RAG라는 기술은 구현하기 쉬우면서도 개인의 정보를 바탕으로 개인화된 기능을 제공해주는 방식입니다.

예를들어 Chatgpt와 같은 LLM에 블로그주인의 정보를 묻는다면 모른다 라고하겠지만, RAG를 통해 정보를 주어지면 제 블로그 글을 바탕으로 대답이 가능하게 됩니다.


코드
https://github.com/murphybread/Librarian

아키텍쳐
![Screenshot 2024-04-04 at 1 38 51 PM 1](https://github.com/murphybread/Librarian/assets/50486329/5a52ac17-1b65-472e-a07f-e075a1e2e333)





## 프레임워크 및 라이브러리 선정

- **Langchain**
RAG구현을 위한 LLM프레임워크.
지금도 그렇지만 막 해당 프로젝트를 시작할 당시인 2024년 초반에 몇 안되는 유명한 기에 선택하였습니다. 파이썬 기반에 레퍼런스도 많고, RAG기능도 지원하는 것을 확인했기에 선택지가 별로 없어서 해당 프레임워크 선택하였습니다.

- **Streamlit**
메인 웹-백엔드 프레임워크.
해당 프레임워크는 파이썬 기반의 특히 LLM관련기능을 쉽게 서비스하는데 도움을 주는 프레임워크입니다. 여러 LLM 어플리케이션을 찾아봤을때 Streamlit으로 만든 것이 많았고, 이 당시 개발지식이 많이 부족했던 터라 이 부분에 관해서는 프레임워크에 의지해야했습니다.(여기에 시간을 쓰면 프로젝트 완성을 못하던 상황)



## 구현된  주요 기능

### RAG기능
앞서 언급한 ChatGPT와 같은 LLM이 학습된 데이터 이외에 외부데이터를 활용하여 대답하는 기능입니다.
가지고 있는 데이터와 임베딩 된 벡터DB를 활용하는 부분이 섞여있습니다.
가장 먼저 로컬파일의 `base_template.txt`라는 txt의 값이 기본적으로 들어가 해당 내용기반으로 답변합니다. 이 후 유저가 특정 글을 더욱 상세하게 요구한다면 Mlivus로 구성된 벡터DB에 쿼리 요청을 통해 id가 같은 값을 찾아 해당 내용을 기반으로 답변합니다

> 고민 포인트
- 어떤 임베딩을 할 것인가?-> 현재 임베딩의 성능 차이는 크지 않기에 사용성을 생각한 OPENAI 임베딩 활용
- 어떤 벡터 DB를 사용할 것인가?-> 장단점이 많고, 아직 명확히 체계화된게 없는 시장이기때문에 적당히 레퍼런스많고 무료로 사용가능한 것 찾아봄.그래서 Milvus 선택
- Langchain의 추상화 * Milvus의 추상화 * Zilliz의 추상화 * Streamlit의 추상화를 어떻게 해결할 것인가?-> 구현하면서 가장 많이 문제를 겪었던부분입니다. 추상화가 너무 많다보니 문제가 발생하였을떄 원인과 해결포인트를 찾기가 힘들었습니다. 특히 데이터 입출력이 안맞다거나, 버전이 틀려졌을때 Zilliz에서는 버전이 바뀐거 기준이지만 Langchain에서는 최신버전 기준이 아니다던가 하는 문제가 많았스빈다.


## Admin 페이지기능
파일일괄 관리와 같은 기능을 수행하기 위한 Admin페이지 기능입니다. 
해당 페이지가 없던 상황에서는 일일히 코드 수정하고 `python feature.py`와 같은 동작을 해야했었던 상황이였기에 구현하였습니다. 암호화 관련 지원 요소가 거의 없어서 단순히 빌민번호가 일치하면 페이지가 보이는 형식으로 구현. 이후 관리자가 사용하기 쉬운 단일 파일 업데이트, 모든 파일 업데이트, 삭제, 생성 버튼을 구현하였습니다. 

>고민 포인트
- Admin페이지를 위한 보안은?-> jwt같은 개념을 사용하려하였는데, 프레임워크에서 지원을 안한다는 결과를 얻었습니다. reddit과 같은 커뮤니티에서도 이런식일 수 밖에 없다고하길래 Streamlit 프레임워크의 공식 설정을 보면서 페이지 렌더링시 조건 분기를 통하여 구현하였습니다.
- 어떤 기능을 Admin으로?-> 처음부터 Admin에 어떤 기능을 만들자 라기보다는 수십번 반복하면서 번거로운 작업, 그리고 단순하지만 잘못하면 무조건 에러가 발생하는 작업들 위주로 선별하였습니다. 지금와서 돌이켜보면 이런것들을 하나의 객체나 모듈로 만들 수 있지 않을까 싶습니다.

## 메모리
uuid를 통해 사용자가 관리하는 메모리 기능입니다.
해당 기능을 구현하기위해서 서버에 어떻게 메모리를 구현해야하는 고민이 많았습니다. 하지만 결론적으로는 세션의 대화데이터를 내가 기록하고, uuid값을 사용자에게 제공하여 필요시 사용자가 해당 uuid를 입력하여 데이터를 찾는 방식으로 구현하였습니다. 물론 보안면이나 편의면이나 부끄러운 수준의 기능이지만, 결국 제한된 환경 특히 기술이 부족한 상황에서 임시로나마 기능을 구현하기 위한 방법이었습니다.

> 고민 포인트
- 메모리 기능 구현을 어떻게?-> 메모리 기능구현에만 며칠을 생각해보았는데 그당시 개발 지식인 세션이라던가 쿠키도 모르던 상황이었습니다. 그래서 일단 기록을 남기고 그것을 어떻게든 불러오자는 생각으로 구현하였습니다. 처음에는 유저관리 시스템을 생각하였는데, 개발지식이 부족하거니와 streamlit이 해당 기능을 지원해주는지도 몰랐기 떄문에 실현되지는 못하였습니다. 그래서 고유성을 가진 형태로 일단 uuid를 만들고 이를 불러오자고 하였습니다. 하지만 해당 기능의 구현에 난이도가 너무높아서 단순하게 나중에 사용자가 입력받는 형태로 구현하였습니다. 

## 메인 프롬프트
https://smith.langchain.com/hub/murphy/librarian_guide
LLM 애플리케이션의 특성상 시스템 프롬프트 조작을 통해 성능을 크게 개선할 수 있었습니다. 하지만 이를 위해선 적절한 예시, 명확한 목적 설명, 퓨샷 러닝을통한 일부 예제 학습등의 절차가 필요했습니다. 특히 대답할때의 answer link가 실제 웹 링크와 연동되게 하기위해서 코드를 수정하는 것보다, 시스템 프롬프트를 수정하는 경우가 많았습니다.
```
human

Library_base_knowledge
{Library_base_knowledge}
history_conversation
{history_conversation}
Current conversation:
Human: {input}
```



## 배포와 운영 관리
운영관리를 위해서는 24시간 서비스되며, 최대한 비용과 저의 시간과 노력을 줄이고자 하였습니다. 그래서 24시간 무료 호스팅 서비스들을 사용하였습니다.
사용된 툴
- **Stramlit Cloud**: 배포된 AI어플리케이션의 호스팅 환경으로 유지비없이, 관리할 필요 없이 무료로 Streamlit기반의 애플리케이션을 활용할 수 있었습니다.
- **Zilliz cloud**: 무로 벡터DB를 사용하기 위한 호스팅 환경입니다.
- **LangSmith**: 정형화된 프롬프트를 관리하기위한 저장소입니다. 매번 RAG시 활용되기에 업데이트가 잦아서 이런 서비스를 호스팅하는 사이트를 사용하기로하였습니다.


