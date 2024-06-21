![Screenshot 2024-04-04 at 1 38 51 PM 1](https://github.com/murphybread/Librarian/assets/50486329/5a52ac17-1b65-472e-a07f-e075a1e2e333)
전체 아키텍쳐

데이터 준비
- 작성자가 마크다운 파일을 블로그에 작성합니다.
해당 마크다운파일을 임베딩한 벡터를 클라우드에 업로드합니다

서비스 사용
- 이용자가 Librarian에 쿼리시, Chatgpt는 VectorDB의 정보를 참고한 RAG방식으로 답변합니다.

Langchain 통해 AI 기능 구현
Streamlit을 통해 프론트 및 백엔드 구현됐습니다.
VectorDB는 Milvus기반이며, 무료 클라우드 형태로 제공하는 Zilliz Cloud를 사용


