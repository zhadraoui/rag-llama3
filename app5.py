from typing import List
import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger
from sqlalchemy.exc import ProgrammingError
from assistant import get_groq_assistant  # type: ignore
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Groq RAG",
    page_icon=":orange_heart:",
)
st.title("RAG avec Llama3 sur Groq")
st.markdown("##### :orange_heart: construit avec [llama]")

def restart_assistant():
    """Redémarre l'assistant en réinitialisant l'état de la session."""
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def get_assistant(llm_model: str, embeddings_model: str) -> Assistant:
    """Retourne une instance de l'assistant avec les modèles spécifiés."""
    logger.info(f"---*--- Création de l'assistant {llm_model} ---*---")
    return get_groq_assistant(llm_model=llm_model, embeddings_model=embeddings_model)

def setup_sidebar() -> tuple:
    """Configure la barre latérale pour la sélection des modèles LLM et des embeddings."""
    llm_model = st.sidebar.selectbox("Sélectionnez LLM", options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
    embeddings_model = st.sidebar.selectbox(
        "Sélectionnez les embeddings",
        options=["nomic-embed-text", "text-embedding-3-small"],
        help="Lorsque vous changez le modèle d'embeddings, les documents devront être ajoutés à nouveau.",
    )
    return llm_model, embeddings_model

def main() -> None:
    llm_model, embeddings_model = setup_sidebar()
    
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        st.session_state["rag_assistant"] = get_assistant(llm_model, embeddings_model)
    rag_assistant = st.session_state["rag_assistant"]

    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("Impossible de créer l'assistant, la base de données fonctionne-t-elle ?")
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Téléchargez un document et posez-moi des questions..."}]

    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if assistant_chat_history:
        st.session_state["messages"] = assistant_chat_history

    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    for message in st.session_state["messages"]:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    if st.session_state["messages"][-1].get("role") == "user":
        question = st.session_state["messages"][-1]["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            # Ajouter une instruction pour répondre en français
            french_prompt = f"Veuillez répondre en français : {question}"
            for delta in rag_assistant.run(french_prompt):
                response += delta  # type: ignore
                resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    if rag_assistant.knowledge_base:
        manage_knowledge_base(rag_assistant)

    if "embeddings_model_updated" in st.session_state:
        st.sidebar.info("Veuillez ajouter à nouveau les documents car le modèle d'embeddings a changé.")
        st.session_state["embeddings_model_updated"] = False

def manage_knowledge_base(rag_assistant: Assistant) -> None:
    """Gère l'ajout de sites web et de PDF à la base de connaissances."""
    if "url_scrape_key" not in st.session_state:
        st.session_state["url_scrape_key"] = 0

    input_url = st.sidebar.text_input("Ajouter une URL à la base de connaissances", key=st.session_state["url_scrape_key"])
    add_url_button = st.sidebar.button("Ajouter l'URL")
    if add_url_button and input_url:
        process_url(input_url, rag_assistant)

    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 100

    uploaded_file = st.sidebar.file_uploader("Ajouter un PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"])
    if uploaded_file:
        process_pdf(uploaded_file, rag_assistant)

    if st.sidebar.button("Vider la base de connaissances"):
        try:
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Base de connaissances vidée")
        except ProgrammingError as e:
            st.sidebar.error(f"Erreur : {e}")
            logger.error(f"Erreur lors du vidage de la base de connaissances : {e}")

    if rag_assistant.storage:
        rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
        new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
        if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
            st.session_state["rag_assistant"] = get_assistant(
                st.session_state["llm_model"], st.session_state["embeddings_model"]
            )
            st.rerun()

    if st.sidebar.button("Nouveau Run"):
        restart_assistant()

def process_url(input_url: str, rag_assistant: Assistant) -> None:
    """Traite l'ajout d'une URL à la base de connaissances."""
    alert = st.sidebar.info("Traitement des URLs...", icon="ℹ️")
    scraper = WebsiteReader(max_links=2, max_depth=1)
    web_documents: List[Document] = scraper.read(input_url)
    if web_documents:
        rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
    else:
        st.sidebar.error("Impossible de lire le site web")
    st.session_state[f"{input_url}_uploaded"] = True
    alert.empty()

def process_pdf(uploaded_file, rag_assistant: Assistant) -> None:
    """Traite l'ajout d'un PDF à la base de connaissances."""
    alert = st.sidebar.info("Traitement du PDF...", icon="🧠")
    reader = PDFReader()
    rag_documents: List[Document] = reader.read(uploaded_file)
    if rag_documents:
        rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
    else:
        st.sidebar.error("Impossible de lire le PDF")
    st.session_state[f"{uploaded_file.name.split('.')[0]}_uploaded"] = True
    alert.empty()

if __name__ == "__main__":
    main()
