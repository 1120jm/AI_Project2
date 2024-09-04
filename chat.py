import torch
from unsloth import FastLanguageModel
from langchain.memory import ConversationBufferMemory
import streamlit as st

# GPU 0번 설정
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Memory 초기화
memory = ConversationBufferMemory()

# 모델과 토크나이저를 세션 상태에 저장 (이미 로드된 경우 다시 로드하지 않음)
if "model" not in st.session_state:
    save_directory = "./trained_model"
    model, tokenizer = FastLanguageModel.from_pretrained(save_directory)
    model = FastLanguageModel.for_inference(model)
    
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer

def main():
    st.title("저자와의 가상 인터뷰 챗봇")

    # 페이지 선택
    if 'page' not in st.session_state:
        st.session_state.page = "select_author"

    if st.session_state.page == "select_author":
        select_author_page()
    elif st.session_state.page == "chat_page":
        chat_page()

def select_author_page():
    st.header("저자 선택")

    author = st.selectbox("대화하고 싶은 작가를 선택하세요.", ("김승호", "유시민"))

    if st.button("선택하기"):
        st.session_state.selected_author = author
        st.session_state.page = "chat_page"

def chat_page():
    st.header(f"{st.session_state.selected_author} 작가님과의 대화")

    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = []

    # 대화 기록을 말풍선 형태로 출력
    for msg in st.session_state.chat_memory:
        if msg['role'] == '나':
            st.markdown(
                f"<div style='text-align: right; color: white; background-color: #6c757d; "
                f"padding: 10px; border-radius: 10px; max-width: 60%; margin-left: auto; word-wrap: break-word;'>"
                f"<strong>나:</strong> {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align: left; color: black; background-color: #f8f9fa; "
                f"padding: 10px; border-radius: 10px; max-width: 60%; word-wrap: break-word;'>"
                f"<strong>{st.session_state.selected_author}:</strong> {msg['content']}</div>",
                unsafe_allow_html=True,
            )

    user_input = st.text_input("질문을 입력하세요:")

    if st.button("보내기") and user_input:
        # 지시사항 추가
        persona_prompt = (
            f"당신은 {st.session_state.selected_author} 작가입니다. 대화는 {st.session_state.selected_author}의 사상과 저서를 바탕으로 진행됩니다.\n\n"
            f"지시사항:\n"
            f"1. 절대 거짓말하지 마세요. AI의 개인적 의견을 넣어서 말하지 마세요.\n"
            f"2. 모르는 내용은 솔직하게 모른다고 말하세요.\n"
            f"3. 넌 지금 인터뷰를 하고 있습니다. 남의 말을 인용하듯이 하지 말고, 직접 말하듯이 존댓말로 답변하세요.\n"
            f"4. {st.session_state.selected_author} 말고 다른 작가에 대한 질문은 모른다고 답변하세요.\n\n"
            f"사용자의 질문: {user_input}\n\n"
            f"{st.session_state.selected_author}의 답변:"
        )

        inputs = st.session_state.tokenizer(
            [persona_prompt], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512,
        ).to("cuda")

        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **inputs, 
                max_new_tokens=100,  
                use_cache=True,  
                num_return_sequences=1,  
                repetition_penalty=1.2,  
                no_repeat_ngram_size=2, 
                eos_token_id=st.session_state.tokenizer.eos_token_id,  
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                temperature=0.4
            )

        generated_text_list = st.session_state.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        full_text = generated_text_list[0]

        # 저자 이름 이후에 응답을 분리하여 처리
        if f"답변:" in full_text:
            generated_text = full_text.split("답변:")[1].strip()
        else:
            generated_text = full_text.strip()

        # 메모리에 대화 내용 저장
        st.session_state.chat_memory.append({"role": "나", "content": user_input})
        st.session_state.chat_memory.append({"role": f"{st.session_state.selected_author}", "content": generated_text})

        st.write(f"**{st.session_state.selected_author}**: {generated_text}")

if __name__ == "__main__":
    main()
