import json
from pathlib import Path

import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Message Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------------------------
# File paths
# -------------------------------------------------
BASE_DIR = Path(__file__).parent

MESSAGES_FILE = BASE_DIR / "data" / "messages.csv"
CLASSIFICATION_FILE = BASE_DIR / "outputs" / "classification_results.csv"
TASK_EVENT_FILE = BASE_DIR / "outputs" / "task_event_results.json"
SENSITIVE_FILE = BASE_DIR / "outputs" / "sensitive_results.json"
MANDATORY_FILE = BASE_DIR / "data" / "mandatory_message_ids.csv"


# -------------------------------------------------
# Load data
# -------------------------------------------------
@st.cache_data
def load_data():
    messages = pd.read_csv(MESSAGES_FILE)

    classification = pd.read_csv(CLASSIFICATION_FILE)

    with open(TASK_EVENT_FILE, "r", encoding="utf-8") as f:
        task_events = pd.DataFrame(json.load(f))

    with open(SENSITIVE_FILE, "r", encoding="utf-8") as f:
        sensitive = pd.DataFrame(json.load(f))

    return messages, classification, task_events, sensitive


messages, classification, task_events, sensitive = load_data()


# -------------------------------------------------
# Helper
# -------------------------------------------------
def safe_count(df):
    return 0 if df is None else len(df)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🧠 Message Intelligence System")
st.caption(
    "Local AI/NLP pipeline for message classification, "
    "task & event extraction, and sensitive-information protection."
)

st.divider()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Message Classification",
        "Tasks & Events",
        "Sensitive Information",
        "Mandatory Messages"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Dataset")
st.sidebar.metric("Total Messages", len(messages))


# =================================================
# OVERVIEW
# =================================================
if page == "Overview":

    st.subheader("System Overview")

    st.markdown("""
    This system processes messages locally through three main modules:

    **1. Message Classification**
    - Action Required
    - Meeting or Event
    - Personal Information
    - General Information
    - Promotional
    - Sensitive Information

    **2. Task & Event Extraction**
    - Title
    - Description
    - Date / deadline
    - Time
    - Person
    - Priority

    **3. Sensitive Information Detection**
    - Detects sensitive information
    - Assigns risk
    - Masks sensitive values
    - Recommends an action
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Messages", len(messages))

    with col2:
        st.metric("Classified", len(classification))

    with col3:
        st.metric("Tasks / Events", len(task_events))

    with col4:
        st.metric("Sensitive Messages", len(sensitive))

    st.divider()

    st.subheader("Processing Flow")

    st.code(
        """Messages
   ↓
Preprocessing
   ↓
┌─────────────────────────────┐
│ Classification              │
│ Task/Event Extraction       │
│ Sensitive Detection/Masking │
└─────────────────────────────┘
   ↓
Structured Outputs""",
        language="text"
    )

    st.info(
        "Sensitive values are not displayed in the Sensitive Information "
        "table. Only masked text is shown."
    )


# =================================================
# CLASSIFICATION
# =================================================
elif page == "Message Classification":

    st.subheader("Message Classification")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Filter by category",
            ["All"] + sorted(classification["category"].dropna().unique())
        )

    with col2:
        search_id = st.text_input(
            "Search Message ID",
            placeholder="Example: MSG_0007"
        )

    filtered = classification.copy()

    if category != "All":
        filtered = filtered[
            filtered["category"] == category
        ]

    if search_id:
        filtered = filtered[
            filtered["message_id"].str.contains(
                search_id,
                case=False,
                na=False
            )
        ]

    st.write(f"Showing **{len(filtered)}** messages")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Category Distribution")

    distribution = (
        classification["category"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    st.bar_chart(
        distribution.set_index("Category")
    )

    st.caption(
        "Confidence values are heuristic scores produced by the local "
        "rule-based classification system."
    )


# =================================================
# TASKS & EVENTS
# =================================================
elif page == "Tasks & Events":

    st.subheader("Tasks & Events")

    if task_events.empty:
        st.warning("No tasks or events were extracted.")
    else:

        col1, col2 = st.columns(2)

        with col1:
            item_type = st.selectbox(
                "Type",
                ["All", "task", "event"]
            )

        with col2:
            task_search = st.text_input(
                "Search source message ID",
                placeholder="Example: MSG_0019"
            )

        filtered = task_events.copy()

        if item_type != "All":
            filtered = filtered[
                filtered["type"] == item_type
            ]

        if task_search:
            filtered = filtered[
                filtered["source_message_id"].astype(str).str.contains(
                    task_search,
                    case=False,
                    na=False
                )
            ]

        display_columns = [
            "item_id",
            "type",
            "title",
            "description",
            "date",
            "deadline",
            "time",
            "person",
            "priority",
            "source_message_id"
        ]

        display_columns = [
            c for c in display_columns
            if c in filtered.columns
        ]

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Tasks",
                len(task_events[task_events["type"] == "task"])
            )

        with col2:
            st.metric(
                "Events",
                len(task_events[task_events["type"] == "event"])
            )

        st.info(
            "Missing or unclear information is represented as null "
            "instead of being guessed."
        )


# =================================================
# SENSITIVE INFORMATION
# =================================================
elif page == "Sensitive Information":

    st.subheader("Sensitive Information Detection & Masking")

    st.warning(
        "Raw sensitive values are intentionally not displayed."
    )

    if sensitive.empty:
        st.success("No sensitive information detected.")
    else:

        col1, col2 = st.columns(2)

        with col1:
            sensitivity_type = st.selectbox(
                "Sensitivity type",
                ["All"] + sorted(
                    sensitive["sensitivity_type"].dropna().unique()
                )
            )

        with col2:
            risk = st.selectbox(
                "Risk",
                ["All"] + sorted(
                    sensitive["risk"].dropna().unique()
                )
            )

        filtered = sensitive.copy()

        if sensitivity_type != "All":
            filtered = filtered[
                filtered["sensitivity_type"] == sensitivity_type
            ]

        if risk != "All":
            filtered = filtered[
                filtered["risk"] == risk
            ]

        # Only masked information is displayed.
        display_columns = [
            "message_id",
            "sensitivity_type",
            "risk",
            "masked_text",
            "recommended_action"
        ]

        st.dataframe(
            filtered[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Sensitive Messages Detected",
            len(sensitive)
        )


# =================================================
# MANDATORY MESSAGE IDs
# =================================================
elif page == "Mandatory Messages":

    st.subheader("Mandatory Message Demonstration")

    st.write(
        "Use this section during the Loom recording to demonstrate "
        "all 15 required message IDs."
    )

    if MANDATORY_FILE.exists():

        mandatory = pd.read_csv(MANDATORY_FILE)

        # Accept either message_id or the first column.
        if "message_id" not in mandatory.columns:
            mandatory.columns = ["message_id"] + list(
                mandatory.columns[1:]
            )

        mandatory_ids = mandatory["message_id"].astype(str).tolist()

    else:

        st.info(
            "mandatory_message_ids.csv was not found. "
            "Enter the 15 IDs below, separated by commas."
        )

        raw_ids = st.text_area(
            "Mandatory Message IDs",
            placeholder="MSG_0001, MSG_0012, MSG_0025, ..."
        )

        mandatory_ids = [
            x.strip()
            for x in raw_ids.split(",")
            if x.strip()
        ]

    if mandatory_ids:

        st.write(f"Mandatory IDs loaded: **{len(mandatory_ids)}**")

        selected_id = st.selectbox(
            "Select mandatory message",
            mandatory_ids
        )

        # Classification result
        class_result = classification[
            classification["message_id"].astype(str) == selected_id
        ]

        st.markdown("### Classification")

        if not class_result.empty:
            st.dataframe(
                class_result,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No classification result found.")

        # Task/event result
        event_result = task_events[
            task_events["source_message_id"].astype(str) == selected_id
        ]

        st.markdown("### Task / Event")

        if not event_result.empty:
            st.dataframe(
                event_result,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No task or event extracted for this message.")

        # Sensitive result
        sensitive_result = sensitive[
            sensitive["message_id"].astype(str) == selected_id
        ]

        st.markdown("### Sensitive Information")

        if not sensitive_result.empty:

            st.dataframe(
                sensitive_result[
                    [
                        "message_id",
                        "sensitivity_type",
                        "risk",
                        "masked_text",
                        "recommended_action"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.success(
                "No sensitive information detected for this message."
            )
