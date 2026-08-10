import os
import subprocess
import sys

# Download and setup git-lfs binary
if os.path.exists('/mount/src'):
    repo_path = "/mount/src/dl-24f2002314-notebook-t22026"
    lfs_dir = "/tmp/git-lfs-3.6.1"
    lfs_bin = os.path.join(lfs_dir, "git-lfs")
    
    # Download git-lfs if not already there
    if not os.path.exists(lfs_bin):
        print("📥 Downloading git-lfs...")
        subprocess.run([
            "curl", "-sL", 
            "https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz",
            "-o", "/tmp/git-lfs.tar.gz"
        ])
        subprocess.run(["tar", "-xzf", "/tmp/git-lfs.tar.gz", "-C", "/tmp/"])
        os.chmod(lfs_bin, 0o755)
    
    # Use absolute path to git-lfs
    git_lfs_path = os.path.join(lfs_dir, "git-lfs")
    
    # Setup git-lfs
    print("🔧 Setting up git-lfs...")
    subprocess.run([git_lfs_path, "install", "--local"], cwd=repo_path)
    
    # Pull LFS files
    print("⬇️ Pulling LFS files...")
    result = subprocess.run(
        [git_lfs_path, "pull"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=300  # 5 min timeout
    )
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    # Verify files are actually pulled
    model_path = os.path.join(repo_path, 'models/Electra-best-model/Electra-best-model/electra-base-checkpoint500')
    model_file = os.path.join(model_path, 'model.safetensors')
    
    if os.path.exists(model_file):
        size = os.path.getsize(model_file)
        print(f"✅ model.safetensors: {size / 1_000_000:.1f} MB")
        if size < 1_000_000:
            print("⚠️ WARNING: Still a pointer file!")
    else:
        print("❌ model.safetensors not found")
    
    sys.stdout.flush()


import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

st.set_page_config(page_title="D&G PROJECT MCQ SOLVER",layout="centered")
MODEL_PATH='models/Electra-best-model/Electra-best-model/electra-base-checkpoint500'
LABELS=["Option A","Option B","Option C","Option D","Option E"]

@st.cache_resource
def load_model():
    model=AutoModelForMultipleChoice.from_pretrained(MODEL_PATH)
    tokenizer=AutoTokenizer.from_pretrained('google/electra-base-discriminator')
    model.eval()
    return tokenizer,model

st.title("MCQ SOLVER")
st.write("Enter a question and its 5 options. The model will predict the correct answer.")

option_labels=["A","B","C","D","E"]
SAMPLE_QUESTION="Determine the correct option: What is the term used in astrophysics to describe light-matter interactions resulting in energy shifts in the radiation field?"
SAMPLE_OPTIONS=["Blueshifting","Redshifting","Reddening","Whitening","Yellowing"]

if "question" not in st.session_state:
    st.session_state.question=""
if "options" not in st.session_state:
    st.session_state.options=["","","","",""]
if st.button("Load Sample Question"):
    st.session_state["question_input"]=SAMPLE_QUESTION
    st.session_state["opt_a_input"]=SAMPLE_OPTIONS[0]
    st.session_state["opt_b_input"]=SAMPLE_OPTIONS[1]
    st.session_state["opt_c_input"]=SAMPLE_OPTIONS[2]
    st.session_state["opt_d_input"]=SAMPLE_OPTIONS[3]
    st.session_state["opt_e_input"]=SAMPLE_OPTIONS[4]
    st.rerun()
question=st.text_area("Question",height=100,key="question_input")

st.subheader("Options")
opt_a=st.text_input("Option A",key="opt_a_input")
opt_b=st.text_input("Option B",key="opt_b_input")
opt_c=st.text_input("Option C",key="opt_c_input")
opt_d=st.text_input("Option D",key="opt_d_input")
opt_e=st.text_input("Option E",key="opt_e_input")

options=[opt_a,opt_b,opt_c,opt_d,opt_e]

tokenizer,model=load_model()
if st.button("Solve"):
    if question.strip()=="" or any(opt.strip()=="" for opt in options):
        st.warning("Please fill in the question and all 5 options.")
    else:
        questions_repeated=[question]*len(options)
        processed=tokenizer(questions_repeated,options,truncation=True,max_length=256,padding='max_length',return_tensors='pt')
        processed={k:v.unsqueeze(0) for k,v in processed.items()}
        with torch.no_grad():
            outputs=model(**processed)
            probs=torch.softmax(outputs.logits,dim=-1).squeeze()
            top3_probs,top3_indices=torch.topk(probs,k=min(3,probs.shape[0]))
            top3_results=[(LABELS[idx] if idx<len(LABELS) else f"Class {idx}",prob) for idx,prob in zip(top3_indices.tolist(),top3_probs.tolist())]
            predicted_label,conf=top3_results[0]

        st.success(f"**Predicted Correct Answer:** {predicted_label} \n**Confidence:** {conf:.2%}")
        st.subheader("Top 3 Predictions")
        for i,(label,confidence) in enumerate(top3_results,start=1):
            st.write(f"{i}. **{label}** — {confidence:.2%}")
            st.progress(confidence)
