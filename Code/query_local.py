import os
import pandas as pd
import torch
import h5py
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from pyngrok import ngrok


@st.cache_resource
def load_model():
    """Tải mô hình chỉ một lần duy nhất."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("/content/jin-AI/checkpoint-68425", trust_remote_code=True).to(device)
    return model


@st.cache_data
def load_corpus(csv_path):
    """Tải dữ liệu văn bản chỉ một lần duy nhất."""
    return pd.read_csv(csv_path)


@st.cache_data
def load_embeddings(h5_path):
    """Tải dữ liệu nhúng chỉ một lần duy nhất."""
    with h5py.File(h5_path, 'r') as f:
        cids = f['cids'][:]
        embeddings = f['embeddings'][:]
    return cids, torch.tensor(embeddings)


def process_query(query_text, model, embeddings, corpus, cids):
    """Xử lý truy vấn và hiển thị kết quả."""
    if query_text.lower() == "no":
        st.write("Dừng tìm kiếm.")
        return

    query_embedding = model.encode(query_text)
    hits = util.semantic_search(query_embedding, embeddings, top_k=10)[0]

    if hits:
        data = [{
            "CID": str(cids[hit['corpus_id']].decode()),
            "Text": corpus.iloc[hit['corpus_id']]['text']
        } for hit in hits]
        st.dataframe(pd.DataFrame(data))
    else:
        st.write("Không tìm thấy kết quả.")


def main():
    """Chương trình chính."""
    # Đường dẫn dữ liệu
    csv_path = "/content/drive/MyDrive/BKAI_2/DATA/combined_output.csv"
    h5_path = '/content/drive/MyDrive/BKAI_2/DATA/encode /merged_output.h5'

    # Tải dữ liệu và mô hình
    corpus = load_corpus(csv_path)
    cids, embeddings = load_embeddings(h5_path)
    model = load_model()

    # Giao diện Streamlit
    st.title("Truy vấn tài liệu pháp luật")

    if "previous_query" not in st.session_state:
        st.session_state.previous_query = ""

    query_text = st.text_input(
        label="Tìm kiếm:",
        placeholder="Nhập nội dung pháp luật"
    ).strip()

    if query_text and query_text != st.session_state.previous_query:
        st.session_state.previous_query = query_text
        process_query(query_text, model, embeddings, corpus, cids)


def start_ngrok():
    """Khởi động Ngrok."""
    public_url = "ok"
    try:
        os.system("pip install pyngrok")
        from pyngrok import ngrok
        public_url = ngrok.connect(8501)
        
    except Exception as e:
        public_url = "ok"
    print("Streamlit URL:", public_url)


if __name__ == '__main__':
    start_ngrok()  # Khởi động ngrok
    main()         # Khởi chạy Streamlit
