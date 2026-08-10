import streamlit as st
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

st.set_page_config(page_title="D&G PROJECT MCQ SOLVER",layout="centered")
MODEL_ID="24f2002314/electra-mcq-solver"
LABELS=["Option A","Option B","Option C","Option D","Option E"]

@st.cache_resource
def load_model():
    model=AutoModelForMultipleChoice.from_pretrained(MODEL_ID)
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
