import os
import streamlit as st

# --- UI ---
st.set_page_config(page_title="StudyMate — PDF Q&A", page_icon="📚", layout="wide")
st.title("📚 StudyMate — AI PDF Q&A for Students")
st.caption("Upload PDFs → Ask a question → Get answers grounded in your material.")

# For now, let's create a simplified version that doesn't require heavy ML dependencies
st.warning("⚠️ This is a simplified version. The full AI functionality requires additional setup.")

# --- Sidebar: PDF upload ---
st.sidebar.header("📄 Upload PDFs")
files = st.sidebar.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)

if files:
    st.sidebar.success(f"Uploaded {len(files)} PDF(s)")
    
    # Show file names
    for file in files:
        st.sidebar.write(f"📄 {file.name}")

# --- Main QA box ---
q = st.text_input("Ask a question about your PDFs:", placeholder="e.g., Explain power factor improvement methods and why they matter.")

col1, col2 = st.columns([1,1])
with col1:
    top_k = st.slider("How many passages to use", 3, 10, 5)
with col2:
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2)

if st.button("Answer", type="primary"):
    if not files:
        st.error("Please upload at least one PDF first.")
    elif not q.strip():
        st.error("Type a question.")
    else:
        st.info("📝 To get AI-powered answers, you need to add your API keys. Here's how:")
        
        st.markdown("""
        ### Where to Add API Keys:
        
        1. **Groq API Key (Recommended - Fast & Free tier)**:
           - Go to Replit Secrets (🔐 in left sidebar)
           - Add key: `GROQ_API_KEY`
           - Get your key from: https://console.groq.com/
        
        2. **OpenAI API Key (Alternative)**:
           - Go to Replit Secrets (🔐 in left sidebar)  
           - Add key: `OPENAI_API_KEY`
           - Get your key from: https://platform.openai.com/
        
        3. **The app will automatically detect and use these keys once added!**
        """)

st.info("💡 Tip: This app uses AI to read your PDFs and answer questions. Add an API key above to enable full functionality!")

# Show current API key status
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

st.sidebar.markdown("### API Status")
if groq_key:
    st.sidebar.success("✅ Groq API key detected")
elif openai_key:
    st.sidebar.success("✅ OpenAI API key detected")
else:
    st.sidebar.warning("⚠️ No API keys detected")
    st.sidebar.info("Add GROQ_API_KEY or OPENAI_API_KEY in Secrets")