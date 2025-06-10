import streamlit as st
import os
import yaml
import time
import uuid
import base64
import logging
from datetime import datetime, timedelta
from io import BytesIO

# Import utility modules
from utils.database import (
    init_database,
    get_conversations,
    create_conversation,
    get_messages,
    add_message,
    get_sources,
    update_conversation_title,
    delete_conversation,
    get_active_knowledge_base,
    set_active_knowledge_base
)

from utils.document_processing import (
    process_and_chunk_file,
    get_all_knowledge_base_names,
    kb_exists,
    auto_create_knowledge_base_if_needed
)

from utils.chat import (
    load_settings,
    process_query,
    get_suggested_prompts,
    direct_openai_query
)

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag-chatbot")

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# from styling import add_custom_css

# Custom CSS
def add_custom_css():
    st.markdown("""
    <style>
    /* =====================================================================
       rac‑branded stylesheet (inline version of assets/style.css)
       ===================================================================== */

    /* ----  rac brand fonts  ---- */
    @font-face {
      font-family: "racForward";
      src: url("../fonts/racForward-Regular.otf") format("opentype");
      font-weight: 400;
    }
    @font-face {
      font-family: "racForward";
      src: url("../fonts/racForward-Medium.otf") format("opentype");
      font-weight: 500;
    }
    @font-face {
      font-family: "racForward";
      src: url("../fonts/racForward-Bold.otf") format("opentype");
      font-weight: 700;
    }

    :root {
      --rac-purple-dark: #43206f;
      --rac-purple:       #5b2d91;
      --rac-pink:         #ff0066;
      --rac-pink-dark:    #e0005b;
      --rac-lavender:     #f3f0f9;
      --rac-light-gray:   #f7f7f8;
      --rac-gray:         #4b5563;
      --rac-green:        #00c389;
    }

    * {
      font-family: "racForward", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ----  Layout  ---- */
    #MainMenu, footer, header { display: none !important; }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
      background: var(--rac-purple-dark) !important;
      color: #fff !important;
    }

    /* Force white text in sidebar for all elements */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiselect span,
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="tag"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: white !important;
    }

    /* Make checkboxes more visible in sidebar */
    section[data-testid="stSidebar"] .racheckbox label span p {
      color: white !important;
      font-weight: 500 !important;
    }

    /* Ensure white text in sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
      background-color: rgba(255, 255, 255, 0.1) !important;
      color: white !important;
      border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
      background-color: rgba(255, 255, 255, 0.2) !important;
      border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Knowledge base selection highlighting */
    .kb-selection-active {
      background-color: rgba(255, 255, 255, 0.2) !important;
      border-left: 3px solid var(--rac-pink) !important;
      border-radius: 4px !important;
      padding-left: 10px !important;
    }

    /* Chat history items */
    section[data-testid="stSidebar"] .chat-history-item {
      background-color: rgba(255, 255, 255, 0.1) !important;
      border-radius: 6px !important;
      padding: 8px 12px !important;
      margin-bottom: 8px !important;
      display: flex !important;
      justify-content: space-between !important;
      align-items: center !important;
      color: white !important;
    }

    section[data-testid="stSidebar"] .chat-history-item:hover {
      background-color: rgba(255, 255, 255, 0.2) !important;
    }

    section[data-testid="stSidebar"] .chat-history-item.active {
      background-color: var(--rac-pink) !important;
    }

    /* Sidebar section headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
      color: white !important;
      font-weight: 600 !important;
      text-shadow: 0 1px 2px rgba(0,0,0,.3) !important;
    }

    /* Sidebar section titles */
    .sidebar-section-title {
      display: flex !important;
      align-items: center !important;
      margin: 20px 0 10px 0 !important;
      color: white !important;
      font-weight: 600 !important;
    }

    .sidebar-section-title svg {
      margin-right: 8px !important;
    }

    /* Main content KB selection styling */
    .kb-card {
      padding: 15px !important;
      border-radius: 8px !important;
      border: 1px solid #e5e7eb !important;
      margin-bottom: 10px !important;
      background: white !important;
      transition: all 0.2s ease !important;
    }

    .kb-card:hover {
      box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    .kb-card.active {
      border-color: var(--rac-purple) !important;
      background: var(--rac-lavender) !important;
    }

    /* Mode indicator styling */
    .mode-indicator {
      padding: 4px 8px !important;
      border-radius: 4px !important;
      font-size: 12px !important;
      margin-left: 10px !important;
    }

    .mode-indicator.rag {
      background: var(--rac-purple) !important;
      color: white !important;
    }

    .mode-indicator.openai {
      background: var(--rac-gray) !important;
      color: white !important;
    }

    .toggle-container {
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
    }

    .main .block-container { padding-top:30px !important; max-width:1200px !important; }

    /* ----  Chat bubbles  ---- */
    [data-testid="rachatMessage"] { background:transparent !important; padding:12px !important; }
    [data-testid="rachatMessage"] [data-testid="rachatMessageContent"]{
      background:var(--rac-light-gray) !important; color:#000 !important;
      padding:12px 16px !important; border-radius:12px !important; max-width:85% !important;
    }
    [data-testid="rachatMessage"][data-testid="rachatMessageUser"]
      [data-testid="rachatMessageContent"]{
      background:var(--rac-purple) !important; color:#fff !important; border-bottom-right-radius:4px !important;
    }

    /* ----  Buttons  ---- */
    .stButton > button{
      width:100% !important; background:var(--rac-lavender) !important;
      color:var(--rac-purple-dark) !important; border:1px solid #d1d5db !important;
      border-radius:6px !important; font-weight:600 !important; transition:all .2s ease;
    }
    .stButton > button:hover{
      background:#e5e7eb !important; border-color:#9ca3af !important; transform:translateY(-1px);
    }

    .stButton > button[data-testid="baseButton-primary"]{
      background:var(--rac-pink) !important; color:#fff !important; border:none !important;
      box-shadow:0 2px 4px rgba(255,0,102,.3) !important;
    }
    .stButton > button[data-testid="baseButton-primary"]:hover{
      background:var(--rac-pink-dark) !important; box-shadow:0 4px 6px rgba(224,0,91,.4) !important;
    }

    .stButton > button[data-testid="baseButton-secondary"]{
      background:var(--rac-gray) !important; color:#fff !important; border:none !important;
      box-shadow:0 1px 2px rgba(0,0,0,.1) !important;
    }
    .stButton > button[data-testid="baseButton-secondary"]:hover{
      background:var(--rac-purple) !important; transform:translateY(-1px) !important;
    }

    /* ----  Progress bar  ---- */
    .stProgress > div > div > div{ background:var(--rac-pink) !important; }

    /* ----  Upload area  ---- */
    .upload-area{
      border:2px dashed var(--rac-purple); background:var(--rac-lavender);
      border-radius:8px; padding:25px; text-align:center;
      transition:all .2s ease; box-shadow:0 1px 3px rgba(0,0,0,.05);
    }
    .upload-area:hover{
      border-color:var(--rac-purple); background:#ebe6f3;
      box-shadow:0 3px 6px rgba(0,0,0,.08); transform:translateY(-1px);
    }

    /* ----  Select & tags  ---- */
    div[data-baseweb="select"]{
      background:#fff !important; color:var(--rac-purple-dark) !important;
      border-radius:6px !important; border:1px solid #d1d5db !important;
    }
    div[data-baseweb="select"]:hover{ border-color:var(--rac-purple) !important; }
    div[data-baseweb="select"] > div{ background:transparent !important; color:inherit !important; }
    div[data-baseweb="tag"]{ background:var(--rac-purple) !important; color:#fff !important; border-radius:4px !important; }
    div[data-baseweb="tag"] svg{ color:#fff !important; }

    /* ----  Thinking dots ---- */
    @keyframes pulse{0%{opacity:.4;transform:scale(1);}50%{opacity:1;transform:scale(1.3);}100%{opacity:.4;transform:scale(1);}}
    .thinking-dot{background:var(--rac-purple);width:8px;height:8px;border-radius:50%;margin-right:4px;animation:pulse 1.5s infinite;}
    .thinking-dot:nth-child(2){animation-delay:.3s;}
    .thinking-dot:nth-child(3){animation-delay:.6s;margin-right:10px;}
    </style>
    """, unsafe_allow_html=True)


# Ensure the database is initialized
init_database()


# Initialize session state variables
def init_session_state():
    logger.info("Initializing session state")

    if "conversations" not in st.session_state:
        st.session_state.conversations = get_conversations()
        logger.debug(f"Loaded {len(st.session_state.conversations)} conversations")

    if "current_conversation_id" not in st.session_state:
        # Create a new conversation if none exists
        if not st.session_state.conversations:
            logger.info("No conversations found, creating a new one")
            new_id = create_conversation()
            st.session_state.conversations = get_conversations()
            st.session_state.current_conversation_id = new_id
        else:
            logger.info(f"Setting current conversation to first in list: {st.session_state.conversations[0][0]}")
            st.session_state.current_conversation_id = st.session_state.conversations[0][0]

    if "messages" not in st.session_state:
        st.session_state.messages = get_messages(st.session_state.current_conversation_id)
        logger.debug(
            f"Loaded {len(st.session_state.messages)} messages for conversation {st.session_state.current_conversation_id}")

    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
        logger.debug(f"Loaded settings: {st.session_state.settings}")



    if "active_kb" not in st.session_state:
        # Get active knowledge base from database
        active_kb = get_active_knowledge_base()
        if active_kb:
            st.session_state.active_kb = active_kb["name"]
            logger.info(f"Active knowledge base set to: {st.session_state.active_kb}")
        else:
            # Auto-create a knowledge base if none exists
            logger.info("No active knowledge base, creating default")
            kb_name = auto_create_knowledge_base_if_needed()
            st.session_state.active_kb = kb_name
            set_active_knowledge_base(kb_name)

    # For multiple KB selection
    if "selected_kbs" not in st.session_state:
        st.session_state.selected_kbs = [st.session_state.active_kb] if st.session_state.active_kb else []

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True

    if "is_thinking" not in st.session_state:
        st.session_state.is_thinking = False

    if "processing_file" not in st.session_state:
        st.session_state.processing_file = False

    if "direct_chat_mode" not in st.session_state:
        st.session_state.direct_chat_mode = False
        logger.info("Direct chat mode initially set to: False")

    if "upload_status" not in st.session_state:
        st.session_state.upload_status = None

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "kb_names" not in st.session_state:
        st.session_state.kb_names = get_all_knowledge_base_names()


def process_uploaded_files(uploaded_files):
    """Process uploaded files automatically with improved handling"""
    if not uploaded_files:
        return

    # Set processing flag
    st.session_state.processing_file = True

    # Keep track of processed file names to avoid reprocessing
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Get list of files to process (filter already processed ones)
    files_to_process = []
    for file in uploaded_files:
        # Create a unique identifier for the file based on name and size
        file_id = f"{file.name}_{len(file.getvalue())}"
        if file_id not in st.session_state.processed_files:
            files_to_process.append((file, file_id))

    if not files_to_process:
        st.session_state.processing_file = False
        return

    logger.info(f"Processing {len(files_to_process)} new uploaded files")

    # Fixed embedding model and chunking strategy
    embedding_model = "text-embedding-3-small"
    chunking_strategy = "semantic_percentile"

    success_count = 0
    error_count = 0
    new_kb_created = False
    newly_created_kbs = []  # Track newly created KBs

    # Process each file
    for file, file_id in files_to_process:
        try:
            logger.info(f"Processing file: {file.name}")

            # Mark as processed
            st.session_state.processed_files.add(file_id)

            # Generate KB name from filename (avoid odd-length string issues)
            file_name = file.name
            base_name = os.path.splitext(file_name)[0].lower().replace(' ', '_')
            kb_name = f"kb_{base_name}"

            # Ensure file is not corrupted by reading content
            file_content = file.read()
            file.seek(0)  # Reset file pointer after reading

            logger.debug(f"File size: {len(file_content)} bytes")
            logger.debug(f"Knowledge base name: {kb_name}")

            # Process the file
            result = process_and_chunk_file(
                file=file,
                kb_name=kb_name,
                embedding_model_name=embedding_model,
                chunking_strategy_name=chunking_strategy
            )

            if result["status"] == "success":
                logger.info(f"Successfully processed {file_name}: {result['chunk_count']} chunks")

                # Add the new KB to the list of selected KBs
                if kb_name not in st.session_state.selected_kbs:
                    st.session_state.selected_kbs.append(kb_name)

                # Track newly created KBs
                newly_created_kbs.append(kb_name)

                # Set as active KB if needed
                if not st.session_state.active_kb or st.session_state.active_kb == "default_knowledge_base":
                    logger.info(f"Setting active knowledge base to: {kb_name}")
                    st.session_state.active_kb = kb_name
                    set_active_knowledge_base(kb_name)

                new_kb_created = True
                success_count += 1

                # Update the knowledge base list immediately
                if kb_name not in st.session_state.kb_names:
                    st.session_state.kb_names.append(kb_name)

                st.session_state.upload_status = {
                    "type": "success",
                    "message": f"✅ {file_name} processed with {result['chunk_count']} chunks"
                }
            else:
                logger.error(f"Error processing {file_name}: {result['message']}")
                error_count += 1
                st.session_state.upload_status = {
                    "type": "error",
                    "message": f"❌ Error processing {file_name}: {result['message']}"
                }
        except Exception as e:
            logger.exception(f"Exception while processing {file.name}: {str(e)}")
            error_count += 1
            st.session_state.upload_status = {
                "type": "error",
                "message": f"❌ Error: {str(e)}"
            }

    logger.info(f"Processing complete: {success_count} successful, {error_count} failed")

    # Update knowledge base list if new KBs were created
    if new_kb_created:
        st.session_state.kb_names = get_all_knowledge_base_names()

        # Add a new status message about knowledge bases
        if newly_created_kbs:
            kb_list = ", ".join(newly_created_kbs)
            st.session_state.upload_status = {
                "type": "success",
                "message": f"✅ Created knowledge base(s): {kb_list}. You can select them in the 'Select Knowledge Base' section below."
            }

    # Turn off direct chat mode if we have successful uploads
    if success_count > 0 and st.session_state.direct_chat_mode:
        logger.info("Turning off direct chat mode due to successful KB creation")
        st.session_state.direct_chat_mode = False

    # Clear processing flag
    st.session_state.processing_file = False


def sidebar():
    st.sidebar.title("RAG Chatbot")

    # New Chat Button with better styling
    if st.sidebar.button("✨ New Chat", use_container_width=True,
                         help="Start a new conversation"):
        logger.info("Creating new chat")
        new_id = create_conversation()
        st.session_state.current_conversation_id = new_id
        st.session_state.conversations = get_conversations()
        st.session_state.messages = []
        st.rerun()

        # Direct Chat Mode Option - Better UI
    direct_chat_container = st.sidebar.container()
    direct_chat_container.markdown("<div style='margin-top:15px; margin-bottom:5px;'></div>",
                                   unsafe_allow_html=True)

    direct_chat = direct_chat_container.checkbox(
        "💬 Direct Chat Mode (No KB)",
        value=st.session_state.direct_chat_mode,
        help="Chat directly with the AI without using a knowledge base"
    )
    if direct_chat != st.session_state.direct_chat_mode:
        logger.info(f"Changing direct chat mode to: {direct_chat}")
        st.session_state.direct_chat_mode = direct_chat
        st.rerun()

    # Chat History - Better styling
    st.sidebar.write("📥Chat History")

    for conv_id, title, created_at in st.session_state.conversations:
        col1, col2 = st.sidebar.columns([5, 1])

        with col1:
            is_active = conv_id == st.session_state.current_conversation_id
            button_type = "primary" if is_active else "secondary"
            display_title = f"📝 {title}" if is_active else title

            if st.button(
                    display_title,
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type=button_type
            ):
                logger.info(f"Switching to conversation: {conv_id}")
                st.session_state.current_conversation_id = conv_id
                st.session_state.messages = get_messages(conv_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{conv_id}", help="Delete this conversation"):
                logger.info(f"Deleting conversation: {conv_id}")
                delete_conversation(conv_id)
                st.session_state.conversations = get_conversations()

                if st.session_state.conversations:
                    if conv_id == st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = st.session_state.conversations[0][0]
                        st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                else:
                    new_id = create_conversation()
                    st.session_state.current_conversation_id = new_id
                    st.session_state.conversations = get_conversations()
                    st.session_state.messages = []

                st.rerun()

    # Professional plan indicator with better styling
    st.sidebar.markdown("<hr style='margin-top:20px; margin-bottom:15px; border-color:rgba(255,255,255,0.1);'>",
                        unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='display:flex; align-items:center;'><span style='color:#a78bfa;margin-right:8px;'>👤</span> <span style='font-weight:500;'>Professional Plan</span></div>",
        unsafe_allow_html=True)

    # Knowledge Bases section with better styling
    st.sidebar.markdown("""
    <div class="sidebar-section-title">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
        </svg>
        <span>Knowledge Bases</span>
    </div>
    """, unsafe_allow_html=True)

    # Update knowledge base names
    kb_names = get_all_knowledge_base_names()
    st.session_state.kb_names = kb_names

    # Multiple knowledge base selection
    if kb_names:
        # Create a multiselect for knowledge bases
        default_selections = st.session_state.selected_kbs if "selected_kbs" in st.session_state else [
            st.session_state.active_kb]

        # Create a placeholder for the multiselect
        kb_select_container = st.sidebar.container()

        # Add a select all/none option
        col1, col2 = kb_select_container.columns([1, 1])

        with col1:
            if st.button("Select All", key="select_all_kbs", use_container_width=True, type="primary"):
                st.session_state.selected_kbs = kb_names.copy()
                st.rerun()

        with col2:
            if st.button("Clear All", key="clear_all_kbs", use_container_width=True):
                st.session_state.selected_kbs = []
                st.rerun()

        # Display the multiselect
        selected_kbs = kb_select_container.multiselect(
            "Select knowledge bases to use",
            options=kb_names,
            default=default_selections,
            key="kb_multiselect",
            label_visibility="collapsed"
        )

        # Update the selected KBs in session state
        if selected_kbs != st.session_state.selected_kbs:
            st.session_state.selected_kbs = selected_kbs

            # If there are selected KBs, set the first one as active
            if selected_kbs:
                st.session_state.active_kb = selected_kbs[0]
                set_active_knowledge_base(selected_kbs[0])
                st.session_state.direct_chat_mode = False
            else:
                # If no KBs selected, switch to direct chat mode
                st.session_state.direct_chat_mode = True

            st.rerun()

        # Display individual KB options with better UI
        # In the sidebar() function:

        # Display individual KB options with better UI
        if st.session_state.selected_kbs:
            st.sidebar.markdown("### Active Knowledge Bases")
            for i, kb_name in enumerate(st.session_state.selected_kbs):
                kb_active = kb_name == st.session_state.active_kb

                # Create a cleaner UI for each KB
                col1, col2 = st.sidebar.columns([4, 1])

                with col1:
                    # Use a button instead of markdown to make it clickable - add index to key
                    button_label = f"{'📌' if kb_active else '📄'} {kb_name}"
                    if st.button(button_label, key=f"sidebar_activate_{kb_name}_{i}",
                                 use_container_width=True,
                                 type="primary" if kb_active else "secondary"):
                        st.session_state.active_kb = kb_name
                        set_active_knowledge_base(kb_name)
                        st.session_state.direct_chat_mode = False
                        st.rerun()

                with col2:
                    # Button to remove KB from selection - add index to key
                    if st.button("❌", key=f"sidebar_remove_{kb_name}_{i}",
                                 help=f"Remove {kb_name} from selection"):
                        st.session_state.selected_kbs.remove(kb_name)
                        # If this was the active KB, change active KB
                        if kb_name == st.session_state.active_kb and st.session_state.selected_kbs:
                            st.session_state.active_kb = st.session_state.selected_kbs[0]
                            set_active_knowledge_base(st.session_state.selected_kbs[0])
                        elif not st.session_state.selected_kbs:
                            st.session_state.direct_chat_mode = True
                        st.rerun()
    else:
        st.sidebar.info("No knowledge bases found. Upload documents to create one.")


def chat_interface():
    # Mode indicator with better styling
    if st.session_state.direct_chat_mode:
        mode_text = "OpenAI Direct Chat"
        mode_class = "openai"
    else:
        # Show multiple KBs if selected
        if len(st.session_state.selected_kbs) > 1:
            mode_text = f"Multiple Knowledge Bases ({len(st.session_state.selected_kbs)})"
        else:
            mode_text = f"Knowledge Base: {st.session_state.active_kb}"
        mode_class = "rag"

    st.markdown(f"""
    <div class='toggle-container' style='margin-bottom:20px;'>
        <h3 style='margin:0;'>{mode_text}</h3>
        <span class='mode-indicator {mode_class}' style='font-weight:600;'>{mode_class.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    # If multiple KBs are selected, show them
    if not st.session_state.direct_chat_mode and len(st.session_state.selected_kbs) > 1:
        kb_list = ", ".join([f"'{kb}'" for kb in st.session_state.selected_kbs])
        st.markdown(
            f"<p style='font-size:14px; color:#6b7280; margin-bottom:20px;'>Using knowledge bases: {kb_list}</p>",
            unsafe_allow_html=True)

    # Option to show sources (only in RAG mode) - better styling
    if not st.session_state.direct_chat_mode:
        show_sources = st.checkbox("📄 Show Sources", value=st.session_state.show_sources,
                                   help="Show the source documents for each response")
        if show_sources != st.session_state.show_sources:
            st.session_state.show_sources = show_sources
            st.rerun()

    # Upload documents section (in the main chat window) - improved UI
    with st.expander("📁 Upload Documents", expanded=False):
        # Upload area with better styling
        st.markdown("<div class='upload-area' style='padding:25px;'>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag & drop PDF or DOCX files here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="main_file_uploader",
            label_visibility="collapsed"
        )
        st.markdown("""
        <div style='text-align:center; margin-top:10px; font-size:14px; color:#6b7280;'>
            Supported formats: PDF, DOCX
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Display upload status if any - better styling
        if st.session_state.upload_status:
            status_type = st.session_state.upload_status["type"]
            status_msg = st.session_state.upload_status["message"]
            icon = "✅" if status_type == "success" else "❌"
            st.markdown(
                f"<div class='upload-status {status_type}' style='display:flex; align-items:center;'><span style='margin-right:5px;'>{icon}</span> {status_msg}</div>",
                unsafe_allow_html=True)

        # Auto-process files when uploaded
        if uploaded_files and not st.session_state.processing_file:
            with st.spinner("Processing files..."):
                process_uploaded_files(uploaded_files)

    # Knowledge Base Selector in main area - FIXED VERSION (using proper Streamlit components)
    # Replace the KB card grid with a dropdown
    if st.session_state.kb_names and len(st.session_state.kb_names) > 0:
        with st.expander("📚 Select Knowledge Base", expanded=True):
            st.write("### Available Knowledge Bases")
            st.write("Select and activate knowledge bases for your questions:")

            # Use a dropdown for selecting the active KB
            selected_kb = st.selectbox(
                "Set Active Knowledge Base:",
                options=st.session_state.kb_names,
                index=st.session_state.kb_names.index(
                    st.session_state.active_kb) if st.session_state.active_kb in st.session_state.kb_names else 0,
                key="kb_dropdown"
            )

            if st.button("🎯 Activate Selected KB", key="activate_from_dropdown"):
                st.session_state.active_kb = selected_kb
                st.session_state.direct_chat_mode = False
                if selected_kb not in st.session_state.selected_kbs:
                    st.session_state.selected_kbs.append(selected_kb)
                set_active_knowledge_base(selected_kb)
                st.rerun()

            # Multi-select for choosing which KBs to use
            multi_select = st.multiselect(
                "Select Knowledge Bases to Use:",
                options=st.session_state.kb_names,
                default=st.session_state.selected_kbs,
                key="kb_multiselect_main"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save Selection", key="save_kb_selection"):
                    st.session_state.selected_kbs = multi_select
                    if not multi_select:
                        st.session_state.direct_chat_mode = True
                    elif st.session_state.active_kb not in multi_select and multi_select:
                        st.session_state.active_kb = multi_select[0]
                        set_active_knowledge_base(multi_select[0])
                    st.rerun()

            with col2:
                if st.button("🔄 Select All KBs", key="select_all_kbs_main"):
                    # Use session_state.kb_names instead of kb_names
                    st.session_state.selected_kbs = st.session_state.kb_names.copy()
                    if not st.session_state.active_kb and st.session_state.selected_kbs:
                        st.session_state.active_kb = st.session_state.selected_kbs[0]
                        set_active_knowledge_base(st.session_state.selected_kbs[0])
                    st.session_state.direct_chat_mode = False
                    st.rerun()
        # Suggested prompts (only show when no messages yet)
        if not st.session_state.messages:
            st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
            st.markdown("### Example Questions to Get Started:")

            # Show suggestions in columns
            col1, col2 = st.columns(2)
            suggestions = get_suggested_prompts()
            # if suggestions:
            #     content = "\n".join(suggestions)  # or format it however you prefer
            #     add_message(st.session_state.current_conversation_id, "user", content)

            with col1:
                for i in range(0, len(suggestions), 2):
                    if i < len(suggestions):
                        if st.button(f"💡 {suggestions[i]}", key=f"sugg_{i}", use_container_width=True):
                            # Add the suggestion to the chat
                            message_id = add_message(st.session_state.current_conversation_id, "user", suggestions[i])
                            st.session_state.messages = get_messages(st.session_state.current_conversation_id)

                            title = suggestions[i][:30] + ('...' if len(suggestions[i]) > 30 else '')
                            update_conversation_title(st.session_state.current_conversation_id, title)
                            st.session_state.conversations = get_conversations()
                            logger.info(f"Updated conversation title to: {title}")
                            st.session_state.is_thinking = True
                            st.rerun()

            with col2:
                for i in range(1, len(suggestions), 2):
                    if i < len(suggestions):
                        if st.button(f"💡 {suggestions[i]}", key=f"sugg_{i}", use_container_width=True):
                            # Add the suggestion to the chat
                            message_id = add_message(st.session_state.current_conversation_id, "user", suggestions[i])
                            st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                            title = suggestions[i][:30] + ('...' if len(suggestions[i]) > 30 else '')
                            update_conversation_title(st.session_state.current_conversation_id, title)
                            st.session_state.conversations = get_conversations()
                            logger.info(f"Updated conversation title to: {title}")
                            st.session_state.is_thinking = True
                            st.rerun()
        #
        # Chat messages - with improved styling
        message_container = st.container()

        with message_container:
            # Display existing messages
            for msg_id, role, content, timestamp in st.session_state.messages:
                if role == "system":
                    continue  # Skip system messages

                # Better styling for timestamp
                try:
                    formatted_time = (datetime.fromisoformat(timestamp.replace('Z', '+00:00')) + timedelta(hours=5, minutes=30)).strftime("%I:%M %p")
                except:
                    formatted_time = timestamp

                with st.chat_message(role):
                    st.write(content)
                    st.markdown(
                        f"<div style='font-size:11px; color:#6b7280; text-align:right; margin-top:5px;'>{formatted_time}</div>",
                        unsafe_allow_html=True)

                    # Show sources for assistant messages (only in RAG mode) - improved styling
                    # In chat_interface() function where it displays sources:

                    # Show sources for assistant messages (only in RAG mode) - improved styling with tabs
                    if role == "assistant" and st.session_state.show_sources and not st.session_state.direct_chat_mode:
                        sources = get_sources(msg_id)
                        if sources:
                            with st.expander("📄 Sources", expanded=False):
                                # Create tabs for different source views
                                source_tabs = st.tabs(["📚 Query-Relevant Sources", "🗂️ Selected Knowledge Bases"])

                                # First tab: Query-relevant sources (sorted by relevance)
                                with source_tabs[0]:
                                    # Group sources by unique document/page to avoid duplicates
                                    unique_sources = {}
                                    for source in sources:
                                        key = (source.get('source', 'Unknown'), source.get('page', 0))
                                        # Only keep the source with the highest score
                                        if key not in unique_sources or source.get('score', 0) > unique_sources[
                                            key].get('score', 0):
                                            unique_sources[key] = source

                                    # Sort by relevance score
                                    sorted_sources = sorted(unique_sources.values(), key=lambda x: x.get('score', 0),
                                                            reverse=True)

                                    for i, source in enumerate(sorted_sources):
                                        source_name = source.get('source', 'Unknown')
                                        page_number = source.get('page', 0)
                                        score = source.get('score', 0)
                                        kb_name = source.get('kb_name', 'Unknown KB')
                                        score_percentage = int(score * 100)

                                        # Better source display with color-coded relevance
                                        relevance_color = "#10b981"  # Green for high relevance
                                        if score_percentage < 70:
                                            relevance_color = "#f59e0b"  # Yellow for medium relevance
                                        if score_percentage < 50:
                                            relevance_color = "#ef4444"  # Red for low relevance

                                        st.markdown(f"""
                                        <div style='margin-bottom:12px; padding:10px; border-radius:6px; background-color:#f9fafb; border-left:4px solid {relevance_color};'>
                                            <div style='font-weight:600; font-size:15px;'>{i + 1}. {source_name}</div>
                                            <div style='display:flex; flex-wrap:wrap; justify-content:space-between; margin-top:5px;'>
                                                <span style='font-weight:500; margin-right:8px;'>KB: {kb_name}</span>
                                                <span style='background:#5b2d91; color:white; padding:2px 8px; border-radius:12px; font-size:12px; margin-right:8px;'>Page {page_number}</span>
                                                <span style='color:{relevance_color}; font-weight:500;'>Relevance: {score_percentage}%</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                # Second tab: Group by knowledge base
                                with source_tabs[1]:
                                    # Group sources by knowledge base
                                    kb_groups = {}
                                    for source in sources:
                                        kb_name = source.get('kb_name', 'Unknown KB')
                                        if kb_name not in kb_groups:
                                            kb_groups[kb_name] = []
                                        kb_groups[kb_name].append(source)

                                    # Sort knowledge bases alphabetically for consistent display
                                    for kb_name in sorted(kb_groups.keys()):
                                        kb_sources = kb_groups[kb_name]

                                        # Deduplicate sources within this KB
                                        unique_kb_sources = {}
                                        for source in kb_sources:
                                            key = (source.get('source', 'Unknown'), source.get('page', 0))
                                            if key not in unique_kb_sources or source.get('score', 0) > \
                                                    unique_kb_sources[key].get('score', 0):
                                                unique_kb_sources[key] = source

                                        # Sort by source name and page
                                        sorted_kb_sources = sorted(unique_kb_sources.values(),
                                                                   key=lambda x: (
                                                                   x.get('source', 'Unknown'), x.get('page', 0)))

                                        # Only display KB section if it has sources
                                        if sorted_kb_sources:
                                            st.markdown(f"### {kb_name}")

                                            for i, source in enumerate(sorted_kb_sources):
                                                source_name = source.get('source', 'Unknown')
                                                page_number = source.get('page', 0)
                                                score = source.get('score', 0)
                                                score_percentage = int(score * 100)

                                                # Color code relevance
                                                relevance_color = "#10b981"  # Green for high relevance
                                                if score_percentage < 70:
                                                    relevance_color = "#f59e0b"  # Yellow for medium relevance
                                                if score_percentage < 50:
                                                    relevance_color = "#ef4444"  # Red for low relevance

                                                st.markdown(f"""
                                                <div style='margin-bottom:8px; padding:8px; border-radius:4px; background-color:#f3f0f9;'>
                                                    <div style='font-weight:500;'>{i + 1}. {source_name}</div>
                                                    <div style='display:flex; justify-content:space-between; font-size:13px;'>
                                                        <span>Page {page_number}</span>
                                                        <span style='color:{relevance_color}; font-weight:500;'>Relevance: {score_percentage}%</span>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)

                                            st.markdown("---")
            # Display thinking indicator with animation
            if st.session_state.is_thinking:
                with st.chat_message("assistant"):
                    st.markdown("""
                    <div style='display:flex; align-items:center;'>
                        <div class='thinking-dot'></div>
                        <div class='thinking-dot'></div>
                        <div class='thinking-dot'></div>
                        <span>Thinking...</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input with better styling
        user_input = st.chat_input("Type your message here...")

        if user_input:
            logger.info(f"Received user input: {user_input[:50]}...")

            # Add user message to database
            message_id = add_message(st.session_state.current_conversation_id, "user", user_input)

            # Update conversation title if this is the first message
            if len(st.session_state.messages) == 0:
                # Use first few words (up to 30 chars) as the title
                title = user_input[:30] + ('...' if len(user_input) > 30 else '')
                update_conversation_title(st.session_state.current_conversation_id, title)
                st.session_state.conversations = get_conversations()
                logger.info(f"Updated conversation title to: {title}")

            # Set thinking state and refresh
            st.session_state.messages = get_messages(st.session_state.current_conversation_id)
            st.session_state.is_thinking = True
            st.rerun()


def handle_ai_response():
    """Process the AI response when in thinking state"""
    if st.session_state.is_thinking and st.session_state.messages:
        # Get the last user message
        last_message = None
        for msg_id, role, content, timestamp in st.session_state.messages:
            if role == "user":
                last_message = content

        if not last_message:
            logger.warning("No user message found to respond to")
            st.session_state.is_thinking = False
            return

        logger.info(f"Processing response to: {last_message[:50]}...")

        # Check if we're in direct chat mode
        if st.session_state.direct_chat_mode:
            logger.info("Using direct OpenAI chat mode")
            try:
                # Get settings
                settings = st.session_state.settings
                chat_model = settings.get("chat_model", "gpt-4o-mini")

                # Query OpenAI directly
                response = direct_openai_query(
                    conversation_id=st.session_state.current_conversation_id,
                    query=last_message,
                    model_name=chat_model
                )

                logger.info("Direct OpenAI response received")

                # Update messages
                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()
            except Exception as e:
                # Handle error
                logger.exception(f"Error in direct chat mode: {str(e)}")
                error_message = f"Error processing query: {str(e)}"
                add_message(st.session_state.current_conversation_id, "assistant", error_message)
                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()
        # RAG mode with knowledge base(s)
        elif st.session_state.selected_kbs:
            try:
                # Load settings
                settings = st.session_state.settings
                chat_model = settings.get("chat_model", "gpt-4o-mini")
                embedding_model = "text-embedding-3-small"
                retrieval_k = settings.get("retrieval", {}).get("top_k", 4)

                # Always use all selected knowledge bases for more comprehensive results
                if len(st.session_state.selected_kbs) > 0:
                    logger.info(f"Using selected knowledge bases: {', '.join(st.session_state.selected_kbs)}")
                    response = process_query(
                        conversation_id=st.session_state.current_conversation_id,
                        query=last_message,
                        kb_names=st.session_state.selected_kbs,
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        retrieval_k=retrieval_k
                    )
                else:
                    # Fallback to active KB if no selection (shouldn't happen normally)
                    logger.info(f"Using active knowledge base: {st.session_state.active_kb}")
                    response = process_query(
                        conversation_id=st.session_state.current_conversation_id,
                        query=last_message,
                        kb_name=st.session_state.active_kb,
                        embedding_model=embedding_model,
                        chat_model=chat_model,
                        retrieval_k=retrieval_k
                    )

                logger.info("RAG response received")

                # Update messages
                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()
            except Exception as e:
                # Handle error with appropriate fallback
                logger.exception(f"Error in RAG mode: {str(e)}")

                # Fallback to direct chat if KB error
                if "Knowledge base" in str(e) and "does not exist" in str(e):
                    try:
                        logger.info("Falling back to direct chat mode due to KB error")
                        st.session_state.direct_chat_mode = True
                        direct_response = direct_openai_query(
                            conversation_id=st.session_state.current_conversation_id,
                            query=last_message,
                            model_name=chat_model
                        )
                        st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                        st.session_state.is_thinking = False
                        st.rerun()
                        return
                    except Exception as fallback_error:
                        logger.exception(f"Error in fallback mode: {str(fallback_error)}")

                error_message = f"Error processing query: {str(e)}"
                add_message(st.session_state.current_conversation_id, "assistant", error_message)
                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()
        else:
            # No knowledge base selected - switch to direct chat mode
            logger.info("No knowledge base selected, switching to direct chat mode")
            st.session_state.direct_chat_mode = True

            try:
                # Get settings and query OpenAI directly
                settings = st.session_state.settings
                chat_model = settings.get("chat_model", "gpt-4o-mini")

                response = direct_openai_query(
                    conversation_id=st.session_state.current_conversation_id,
                    query=last_message,
                    model_name=chat_model
                )

                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()
            except Exception as e:
                logger.exception(f"Error in direct fallback mode: {str(e)}")
                error_message = f"Error processing query: {str(e)}"
                add_message(st.session_state.current_conversation_id, "assistant", error_message)
                st.session_state.messages = get_messages(st.session_state.current_conversation_id)
                st.session_state.is_thinking = False
                st.rerun()


def api_key_required_screen():
    st.error("⚠️ OpenAI API Key Required")
    st.markdown("""
    Please configure your OpenAI API key in the settings.yml file:

    1. Get your API key from [OpenAI's platform](https://platform.openai.com/api-keys)
    2. Open the `settings.yml` file in the root directory
    3. Replace the placeholder with your actual API key:
       ```yaml
       openai_key: "your-api-key-here"
       ```
    4. Restart the application
    """)


def main():
    logger.info("Application starting")

    # Apply custom CSS
    add_custom_css()

    # Initialize session state
    init_session_state()

    # Render sidebar
    sidebar()

    # Check if OpenAI API key is configured
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OpenAI API key not configured, showing API key screen")
        api_key_required_screen()
    else:
        chat_interface()

        # Process AI response if in thinking state
        if st.session_state.is_thinking:
            handle_ai_response()

    logger.debug("UI rendering complete")


if __name__ == "__main__":
    main()