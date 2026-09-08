import os
import sys
import gradio as gr
import spaces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from screening_engine import screen_candidate


# ---------------------------------------------------------
# ZeroGPU compatibility
# ---------------------------------------------------------
# Hugging Face ZeroGPU requires at least one @spaces.GPU
# function to exist during startup.
#
# This function is NEVER called.
# Our actual CV screening system continues to run on CPU.
# ---------------------------------------------------------

@spaces.GPU
def zerogpu_compatibility():
    return "ZeroGPU enabled"


# ---------------------------------------------------------
# Format screening result
# ---------------------------------------------------------

def format_screening_result(data):

    candidate = data.get("candidate", {})
    job = data.get("job", {})

    score = data.get("score", 0)
    coverage = data.get("required_coverage", 0)

    recommendation = data.get(
        "recommendation",
        "REVIEW"
    )

    reason = data.get("reason", "")

    processing_time = data.get(
        "processing_time",
        0
    )

    candidate_name = candidate.get(
        "candidate_name",
        "Unknown Candidate"
    )

    email = candidate.get(
        "email",
        ""
    )

    phone = candidate.get(
        "phone",
        ""
    )

    job_title = job.get(
        "job_title",
        "Candidate Position"
    )

    matches = data.get(
        "matches",
        []
    )

    output = f"""
# 🤖 Candidate Screening Result

## 👤 Candidate

**Name:** {candidate_name}

"""

    if email:
        output += f"**Email:** {email}\n\n"

    if phone:
        output += f"**Phone:** {phone}\n\n"

    output += f"""
## 💼 Position

**{job_title}**

---

## 📊 Overall Assessment

### Match Score: **{score:.2f}%**

### Required Qualification Coverage: **{coverage:.2f}%**

### Recommendation: **{recommendation}**

**Reason:** {reason}

---

## 🔎 Requirement Analysis

"""

    if not matches:

        output += (
            "No requirements were detected "
            "from the job description.\n"
        )

        return output

    for i, match in enumerate(
        matches,
        start=1
    ):

        requirement = match.get(
            "requirement",
            ""
        )

        category = match.get(
            "category",
            "knowledge"
        )

        match_type = match.get(
            "type",
            "required"
        )

        final_score = match.get(
            "final_score",
            0
        )

        similarity = match.get(
            "similarity",
            0
        )

        keyword_score = match.get(
            "keyword_score",
            0
        )

        status = match.get(
            "status",
            "Unknown"
        )

        evidence = match.get(
            "evidence",
            ""
        )

        matched_skills = match.get(
            "matched_skills",
            []
        )

        output += f"""
### {i}. {requirement}

- **Type:** {match_type.title()}
- **Category:** {category.replace("_", " ").title()}
- **Status:** **{status}**
- **Match Score:** {final_score * 100:.1f}%
- **Semantic Similarity:** {similarity * 100:.1f}%
- **Keyword Match:** {keyword_score * 100:.1f}%
"""

        if matched_skills:

            output += (
                "- **Matched Skills:** "
                + ", ".join(matched_skills)
                + "\n"
            )

        if evidence:

            clean_evidence = evidence.strip()

            if len(clean_evidence) > 500:

                clean_evidence = (
                    clean_evidence[:500]
                    + "..."
                )

            output += f"""
**CV Evidence:**

> {clean_evidence}

"""

        output += "---\n"

    output += f"""
## ⚡ Performance

**Processing Time:** {processing_time:.2f} seconds

> This result is an AI-assisted screening assessment
> and should be followed by human review.
"""

    return output


# ---------------------------------------------------------
# Main screening function
# ---------------------------------------------------------

def analyze_candidate(
    cv_file,
    job_description
):

    if cv_file is None:

        return (
            "⚠️ **Please upload a CV.**"
        )

    if (
        not job_description
        or not job_description.strip()
    ):

        return (
            "⚠️ **Please enter a job description.**"
        )

    try:

        result = screen_candidate(
            cv_file,
            job_description
        )

        return format_screening_result(
            result
        )

    except Exception as e:

        return (
            "❌ **Screening Error**\n\n"
            f"```text\n{str(e)}\n```"
        )


# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------

with gr.Blocks(
    title="AI Candidate Screening Assistant"
) as demo:

    gr.Markdown(
        """
# 🤖 AI Candidate Screening Assistant

Upload a CV and paste a job description to evaluate
the candidate using semantic matching, keyword matching,
and CV evidence analysis.
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            cv_file = gr.File(
                label="📄 Upload CV",
                file_types=[
                    ".pdf",
                    ".docx"
                ],
                type="filepath"
            )

            job_description = gr.Textbox(
                label="💼 Job Description",
                placeholder=(
                    "Paste the complete job "
                    "description here..."
                ),
                lines=18
            )

            analyze_button = gr.Button(
                "🔍 Screen Candidate",
                variant="primary"
            )

        with gr.Column(scale=1):

            screening_result = gr.Markdown(
                value=(
                    "### Screening Result\n\n"
                    "Upload a CV and enter a job "
                    "description to begin."
                )
            )

    analyze_button.click(
        fn=analyze_candidate,
        inputs=[
            cv_file,
            job_description
        ],
        outputs=[
            screening_result
        ]
    )


# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
        show_error=True
    )