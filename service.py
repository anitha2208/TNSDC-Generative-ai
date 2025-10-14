import os
import json
import google.generativeai as genai


# ==============================
# Utility
# ==============================
def load_json_if_exists(path, default={}):
    """Load JSON file if it exists - with better error handling"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
    return default


# ==============================
# Gemini Model Loader
# ==============================
def load_model():
    api_key = "AIzaSyCA-rO_ZqfVGGF1sO5BKVEGClxSA2UTezY"  # ⚡ Replace with your Gemini API key
    if not api_key or api_key.strip() == "":
        raise ValueError("❌ GEMINI_API_KEY is missing.")

    genai.configure(api_key=api_key)

    model_names = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-pro",
        "models/gemini-2.0-pro-exp",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    for model_name in model_names:
        try:
            print(f"🔄 Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            print(f"✅ Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue

    raise ValueError("❌ No compatible Gemini model found. Please check your API key and available models.")


# ==============================
# PROMPT 1 — Base Content Generation
# ==============================
def build_prompt(description, insights, comparison, dashboard, query_out, sentiment_data, insights_charts):
    json_schema = '''
{
  "slides": [
    {
      "slide_index": "integer",
      "placeholders": {
        "title": "string",
        "subtitle": "string (optional)",
        "content": "string (optional)",
        "image_path": "string (optional)"
      }
    }
  ]
}
'''

    prompt = f"""
You are an expert PPT creator and data analyst.

### Dataset Description
{json.dumps(description, indent=2)}

### Insights
{json.dumps(insights, indent=2)}

### Comparisons
{json.dumps(comparison, indent=2)}

### Dashboard
{json.dumps(dashboard, indent=2)}

### Query Results
{json.dumps(query_out, indent=2)}

### Sentiment Analysis
{json.dumps(sentiment_data, indent=2)}

### Insight Charts
{json.dumps(insights_charts, indent=2)}

---

Generate a structured JSON (preview.json) **strictly following this schema: {json_schema}**

Rules:
- Slide 2: Data Description
- Slide 3: Key Comparisons
- Next slides: Feature Insights, Query Results, Dashboard, Business Insights, Summary, Thank You
- Include image paths where relevant
- Only output valid JSON (no markdown)
- Make content professional, concise, and business-ready
"""
    return prompt.strip()


# ==============================
# PROMPT 2 — Enhanced Content Generation
# ==============================
def build_enhanced_content_prompt(json_input):
    """Enhance slide content with Gamma-AI style visuals and insights"""
    prompt = f"""
You are an advanced AI business content creator like Gamma AI.
Enhance the following slide content for a professional presentation.

### Input JSON
{json.dumps(json_input, indent=2)}

### TASK
- Rewrite and enrich content with **business-focused insights**, bullet points, and better narrative flow.
- Add relevant **flowcharts, workflows, or image suggestions** as "(Image: ...)" when useful.
- Use professional tone for corporate reports.
- Preserve JSON structure (title, content, image_path, etc).
- Output only valid JSON in the same structure.

Output:
- JSON only, same schema.
"""
    return prompt.strip()


# ==============================
# PROMPT 3 — PPT Alignment Optimization
# ==============================
def build_alignment_prompt(preview_json, template_summary):
    """Align slide formatting to template style"""
    prompt = f"""
You are a PowerPoint design expert.
Refine the following slides for **perfect alignment and font consistency**.

### Slide JSON
{json.dumps(preview_json, indent=2)}

### Template Summary
{template_summary}

### TASK
- Maintain uniform **font style, size, and spacing** (use template rules).
- Adjust text for readability (line breaks, bullets, titles).
- Ensure every slide fits well visually.
- Do not rewrite meaning — just refine layout text.

Output:
- JSON only, same schema as input.
"""
    return prompt.strip()


# ==============================
# Main Service Orchestrator
# ==============================
def service():
    outputs_dir = getattr(service, 'output_dir', 'output')
    os.makedirs(outputs_dir, exist_ok=True)
    print(f"📂 Output Directory: {outputs_dir}")

    # === Load Data Files ===
    description = load_json_if_exists(os.path.join(outputs_dir, "data_description.json"))
    insights_data = load_json_if_exists(os.path.join(outputs_dir, "insights.json"), [])
    charts_insights_data = load_json_if_exists(os.path.join(outputs_dir, "insights_charts.json"), [])
    comparison = load_json_if_exists(os.path.join(outputs_dir, "comparison.json"))
    dashboard_data = load_json_if_exists(os.path.join(outputs_dir, "dashboard.json"))
    query_out = load_json_if_exists(os.path.join(outputs_dir, "query_output.json"))
    sentiment_data = load_json_if_exists(os.path.join(outputs_dir, "sentiment.json"))

    if not description:
        raise FileNotFoundError(f"❌ data_description.json not found in {outputs_dir}")

    # === Load Gemini Model ===
    model = load_model()

    # ============================================================
    # STEP 1: BASE SLIDE GENERATION
    # ============================================================
    print("\n🚀 Generating Base Preview (raw business slides)...")
    base_prompt = build_prompt(description, insights_data, comparison, dashboard_data, query_out, sentiment_data, charts_insights_data)
    base_response = model.generate_content(base_prompt)

    # --- Safe Gemini Output Extraction ---
    generated_text = ""
    try:
        if hasattr(base_response, "text") and base_response.text:
            generated_text = base_response.text.strip()
        elif hasattr(base_response, "candidates"):
            for cand in base_response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "text"):
                            generated_text += part.text
        generated_text = generated_text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting Gemini output: {e}")
        generated_text = ""

    if not generated_text:
        print("❌ No valid text returned from Gemini. Check API quota or input data.")
        return

    if generated_text.startswith("```"):
        generated_text = generated_text.strip("`")
        if generated_text.lower().startswith("json"):
            generated_text = generated_text[4:].strip()

    try:
        base_json = json.loads(generated_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini output was not valid JSON: {e}")
        print(f"Raw Output (first 500 chars):\n{generated_text[:500]}")
        base_json = {"slides": [], "raw_text": generated_text}

    base_path = os.path.join(outputs_dir, "preview_base.json")
    json.dump(base_json, open(base_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"✅ Base preview generated: {base_path}")

    # ============================================================
    # STEP 2: ENHANCED CONTENT GENERATION
    # ============================================================
    print("\n🎨 Enhancing content with visuals & better storytelling...")
    enhance_prompt = build_enhanced_content_prompt(base_json)
    enhance_response = model.generate_content(enhance_prompt)

    enhanced_text = ""
    try:
        if hasattr(enhance_response, "text") and enhance_response.text:
            enhanced_text = enhance_response.text.strip()
        elif hasattr(enhance_response, "candidates"):
            for cand in enhance_response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "text"):
                            enhanced_text += part.text
        enhanced_text = enhanced_text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting enhanced response: {e}")
        enhanced_text = ""

    if not enhanced_text:
        print("❌ No valid text returned from Gemini for enhancement.")
        return

    if enhanced_text.startswith("```"):
        enhanced_text = enhanced_text.strip("`")
        if enhanced_text.lower().startswith("json"):
            enhanced_text = enhanced_text[4:].strip()

    try:
        enhanced_json = json.loads(enhanced_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ Enhanced output not valid JSON: {e}")
        print(f"Raw Output (first 500 chars):\n{enhanced_text[:500]}")
        enhanced_json = {"slides": [], "raw_text": enhanced_text}

    enhanced_path = os.path.join(outputs_dir, "preview_enhanced.json")
    json.dump(enhanced_json, open(enhanced_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"✅ Enhanced content saved: {enhanced_path}")

    # ============================================================
    # STEP 3: PPT ALIGNMENT OPTIMIZATION
    # ============================================================
    print("\n🧩 Optimizing alignment based on PPT template...")

    template_summary = """
Template Style: Professional
Font: Segoe UI or Calibri
Title Font Size: 36pt
Body Font Size: 24pt
Color Scheme: Blue accent with white background
Layout: Title + Content (image optional)
"""
    alignment_prompt = build_alignment_prompt(enhanced_json, template_summary)
    align_response = model.generate_content(alignment_prompt)

    align_text = ""
    try:
        if hasattr(align_response, "text") and align_response.text:
            align_text = align_response.text.strip()
        elif hasattr(align_response, "candidates"):
            for cand in align_response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "text"):
                            align_text += part.text
        align_text = align_text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting alignment response: {e}")
        align_text = ""

    if not align_text:
        print("❌ No valid text returned from Gemini for alignment.")
        return

    if align_text.startswith("```"):
        align_text = align_text.strip("`")
        if align_text.lower().startswith("json"):
            align_text = align_text[4:].strip()

    try:
        final_json = json.loads(align_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ Alignment output not valid JSON: {e}")
        print(f"Raw Output (first 500 chars):\n{align_text[:500]}")
        final_json = {"slides": [], "raw_text": align_text}

    final_path = os.path.join(outputs_dir, "preview_final.json")
    json.dump(final_json, open(final_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"🎯 Final aligned preview ready: {final_path}")

    print("\n✅ All stages complete! You can now run ppt.py to generate the presentation.")


# ==============================
# Run Example
# ==============================
if __name__ == "__main__":
    service()
