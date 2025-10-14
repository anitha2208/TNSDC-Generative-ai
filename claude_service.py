import os
import json
import google.generativeai as genai

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
    api_key = "AIzaSyB7M7RKvctshe_qVys-iq1mOxm7ZHZ3LxQ"  # ⚡ Replace with your Gemini API key
    if not api_key or api_key.strip() == "":
        raise ValueError("❌ GEMINI_API_KEY is missing.")
    
    genai.configure(api_key=api_key)
    
    model_names = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
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
    
    raise ValueError("❌ No compatible Gemini model found.")


# ==============================
# PROMPT 1: Content Generation
# ==============================
def build_content_generation_prompt(description, insights, insights_charts, comparison, dashboard, query_out, sentiment_data):
    """Generate enhanced professional content"""
    
    json_schema = '''
{
  "slides": [
    {
      "slide_index": "integer",
      "slide_type": "title/content/dashboard/conclusion",
      "placeholders": {
        "title": "string (max 80 characters)",
        "subtitle": "string (optional, max 100 characters)",
        "content": "string or array of strings (NO ** markers)",
        "image_path": "string (optional, exact path)"
      },
      "estimated_char_count": "integer"
    }
  ]
}
'''

    prompt = f"""You are an expert Business Intelligence Analyst and Professional Presentation Designer.

Generate a complete PowerPoint presentation content in JSON format for a PROFESSIONAL BUSINESS AUDIENCE.

# INPUT DATA
## Dataset Description
{json.dumps(description, indent=2)}

## Statistical Insights
{json.dumps(insights, indent=2)}

## Insights Charts
{json.dumps(insights_charts, indent=2)}

## Comparison Analysis
{json.dumps(comparison, indent=2)}

## Dashboard Information
{json.dumps(dashboard, indent=2)}

## Query Results
{json.dumps(query_out, indent=2)}

## Sentiment Analysis
{json.dumps(sentiment_data, indent=2)}

---

# OUTPUT REQUIREMENTS

## JSON Schema: {json_schema}

## Slide Structure (Order matters)
1. **Slide 1**: Title slide with main title and compelling subtitle
2. **Slide 2**: Executive Summary (150-200 words)
3. **Slide 3**: Dataset Overview (200-250 words)
4. **Slide 4**: Dashboard (if available - 50-75 words description)
5. **Slides 5-N**: Key Insights (150-200 words each with chart)
6. **Slides N+1**: Comparison Analysis (150-200 words each)
7. **Slide N+2**: Query Results (if available, 100-word interpretation)
8. **Slide N+3**: Sentiment Analysis (if data exists, 150-200 words)
9. **Slide N+4**: Strategic Business Recommendations (250-300 words total)
10. **Slide N+5**: Conclusion (200-250 words)
11. **Final Slide**: Thank You with call-to-action subtitle

## CRITICAL CONTENT RULES

### 1. NO MARKDOWN FORMATTING
- NEVER use ** for bold
- NEVER use * for italics  
- NEVER use # for headers
- Use plain text only

### 2. Content Length Guidelines (STRICTLY FOLLOW)
- Title slide subtitle: 80-100 characters
- Executive Summary: 150-200 words (900-1200 chars)
- Data Overview: 200-250 words (1200-1500 chars)
- Dashboard description: 50-75 words (300-450 chars)
- Insight slides: 150-200 words each (900-1200 chars)
- Comparison slides: 150-200 words (900-1200 chars)
- Business Recommendations: 250-300 words (1500-1800 chars)
- Conclusion: 200-250 words (1200-1500 chars)
- Thank You subtitle: 40-60 characters

### 3. Content Style
- Professional paragraphs (NOT bullet points except recommendations)
- Business terminology
- Specific numbers and percentages from data
- No generic statements

### 4. Business Recommendations Format
"1. [ACTION VERB] [Recommendation]: [Detailed explanation with data]. [Expected outcome].

2. [ACTION VERB] [Recommendation]: [Detailed explanation with data]. [Expected outcome]."

### 5. Image Path Handling
- Include EXACT image paths from input data in "image_path" field
- Do NOT reference images in content text
- Do NOT use (Chart: path) format
- Keep image paths separate

### 6. Data-Driven Content
Each insight slide:
- Key finding first
- Specific statistics
- Business implications
- Actionable insights

### 7. Paragraph Structure
- 3-5 sentences per paragraph
- Logical flow: finding → analysis → implication
- Transition words
- Forward-looking ending

Output ONLY valid JSON (no markdown, no explanation)."""
    
    return prompt.strip()


# ==============================
# PROMPT 2: Font Size Alignment
# ==============================
def build_font_alignment_prompt(preview_json, template_info):
    """Analyze and fix font sizes for consistency"""
    
    prompt = f"""You are an expert PowerPoint Template Designer and Layout Specialist.

Analyze the generated content and provide FIXED font sizes for optimal consistency.

# INPUT DATA
## Generated Content
{json.dumps(preview_json, indent=2)}

## Template Information
Template: {template_info.get('template_name', 'Professional')}
Slide Size: {template_info.get('slide_width', '10"')} x {template_info.get('slide_height', '7.5"')}
Text Color: {template_info.get('text_color', '#000000')}

---

# FONT SIZE RULES (STRICT)

## Slide Types and Sizes:
1. **Title Slide**: Title 36pt, Subtitle 18pt
2. **Content with Images**: Title 28pt, Content 16pt
3. **Content Full Width**: Title 28pt, Content 18pt
4. **Dashboard**: Title 28pt, Caption 14pt
5. **Recommendations**: Title 28pt, Content 16pt
6. **Conclusion**: Title 28pt, Content 18pt
7. **Thank You**: Title 36pt, Subtitle 20pt

## Character Limits:
- Full width content: Maximum 1400 characters
- Split layout: Maximum 1100 characters
- Dashboard caption: Maximum 450 characters
- Recommendations: Maximum 1600 characters

## OUTPUT FORMAT

{{
  "font_size_mapping": [
    {{
      "slide_index": 1,
      "slide_type": "title",
      "title_font_size": 36,
      "subtitle_font_size": 18,
      "content_font_size": null,
      "layout_type": "centered",
      "has_image": false,
      "content_status": "optimal/too_long/too_short",
      "content_length": 100,
      "recommended_length": "80-100",
      "adjustment_needed": false,
      "adjustment_notes": ""
    }}
  ],
  "overall_recommendations": [
    "List of content adjustments needed"
  ],
  "template_compatibility": {{
    "font_consistency": "100%",
    "layout_optimization": "Complete"
  }}
}}

Analyze each slide:
1. Calculate character count
2. Determine slide type
3. Check for images
4. Assign fixed font size
5. Evaluate content fit
6. Provide adjustment recommendations

Return ONLY valid JSON."""
    
    return prompt.strip()


# ==============================
# LLM Integration Functions
# ==============================
def generate_content_with_llm(model, description, insights, insights_charts, comparison, dashboard, query_out, sentiment_data):
    """First LLM call: Generate enhanced content"""
    
    print("\n" + "="*60)
    print("STEP 1: Generating Enhanced Content")
    print("="*60)
    
    prompt = build_content_generation_prompt(
        description, insights, insights_charts, comparison, 
        dashboard, query_out, sentiment_data
    )
    
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        generated_text = response.text.strip()
        
        # Clean markdown wrapper
        if generated_text.startswith("```"):
            generated_text = generated_text.strip("`")
            if generated_text.lower().startswith("json"):
                generated_text = generated_text[4:].strip()
        
        preview_json = json.loads(generated_text)
        print("✅ Content generation successful")
        return preview_json
        
    except Exception as e:
        print(f"❌ Error generating content: {e}")
        raise


def optimize_font_alignment(model, preview_json, template_info):
    """Second LLM call: Optimize font sizes and alignment"""
    
    print("\n" + "="*60)
    print("STEP 2: Optimizing Font Sizes and Alignment")
    print("="*60)
    
    prompt = build_font_alignment_prompt(preview_json, template_info)
    
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 4096,
    }
    
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        generated_text = response.text.strip()
        
        # Clean markdown wrapper
        if generated_text.startswith("```"):
            generated_text = generated_text.strip("`")
            if generated_text.lower().startswith("json"):
                generated_text = generated_text[4:].strip()
        
        alignment_json = json.loads(generated_text)
        print("✅ Font alignment optimization successful")
        return alignment_json
        
    except Exception as e:
        print(f"❌ Error optimizing alignment: {e}")
        raise


# ==============================
# Main Service Orchestrator
# ==============================
def service():
    outputs_dir = getattr(service, 'output_dir', 'output')
    print(f"\n🔍 Looking for files in: {outputs_dir}")
    os.makedirs(outputs_dir, exist_ok=True)

    # Load all input files
    description = load_json_if_exists(os.path.join(outputs_dir, "data_description.json"))
    insights_data = load_json_if_exists(os.path.join(outputs_dir, "insights.json"), [])
    charts_insights_data = load_json_if_exists(os.path.join(outputs_dir, "insights_charts.json"), [])
    comparison = load_json_if_exists(os.path.join(outputs_dir, "comparison.json"))
    dashboard_data = load_json_if_exists(os.path.join(outputs_dir, "dashboard.json"))
    query_out = load_json_if_exists(os.path.join(outputs_dir, "query_output.json"))
    sentiment_data = load_json_if_exists(os.path.join(outputs_dir, "sentiment.json"))

    # Sanity check
    if not description:
        raise FileNotFoundError(f"❌ data_description.json not found in {outputs_dir}")

    # Load model
    print("\n⚡ Loading Gemini model...")
    try:
        model = load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # STEP 1: Generate enhanced content
    try:
        preview_json = generate_content_with_llm(
            model, description, insights_data, charts_insights_data,
            comparison, dashboard_data, query_out, sentiment_data
        )
        
        # Save initial preview.json
        preview_path = os.path.join(outputs_dir, "preview.json")
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview_json, f, indent=2, ensure_ascii=False)
        print(f"✅ Initial preview.json saved to {preview_path}")
        
    except Exception as e:
        print(f"❌ Content generation failed: {e}")
        return

    # STEP 2: Optimize font sizes and alignment
    try:
        # Prepare template info (you can load from input.json)
        input_json_path = os.path.join(outputs_dir.replace("output", "input"), "csv", "input.json")
        template_info = {}
        
        if os.path.exists(input_json_path):
            with open(input_json_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)
                template_info = {
                    "template_name": input_data.get("template_name", "Professional"),
                    "slide_width": "10 inches",
                    "slide_height": "7.5 inches",
                    "text_color": input_data.get("text_color", "#000000"),
                    "background_style": "Light"
                }
        else:
            # Default template info
            template_info = {
                "template_name": "Professional",
                "slide_width": "10 inches",
                "slide_height": "7.5 inches",
                "text_color": "#000000",
                "background_style": "Light"
            }
        
        alignment_json = optimize_font_alignment(model, preview_json, template_info)
        
        # Save font alignment mapping
        alignment_path = os.path.join(outputs_dir, "font_alignment.json")
        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(alignment_json, f, indent=2, ensure_ascii=False)
        print(f"✅ Font alignment mapping saved to {alignment_path}")
        
        # Print recommendations
        print("\n" + "="*60)
        print("ALIGNMENT RECOMMENDATIONS")
        print("="*60)
        for rec in alignment_json.get("overall_recommendations", []):
            print(f"• {rec}")
        
        # Check for content issues
        font_mapping = alignment_json.get("font_size_mapping", [])
        issues_found = False
        
        print("\n" + "="*60)
        print("CONTENT ANALYSIS")
        print("="*60)
        
        for slide_info in font_mapping:
            if slide_info.get("adjustment_needed", False):
                issues_found = True
                slide_idx = slide_info.get("slide_index")
                status = slide_info.get("content_status")
                length = slide_info.get("content_length")
                recommended = slide_info.get("recommended_length")
                notes = slide_info.get("adjustment_notes", "")
                
                print(f"\n⚠️ Slide {slide_idx}: {status.upper()}")
                print(f"   Current length: {length} chars")
                print(f"   Recommended: {recommended} chars")
                print(f"   Notes: {notes}")
        
        if not issues_found:
            print("✅ All slides have optimal content length")
        
        print("\n" + "="*60)
        print("FONT SIZE SUMMARY")
        print("="*60)
        
        for slide_info in font_mapping:
            slide_idx = slide_info.get("slide_index")
            slide_type = slide_info.get("slide_type")
            title_font = slide_info.get("title_font_size")
            content_font = slide_info.get("content_font_size")
            
            if content_font:
                print(f"Slide {slide_idx} ({slide_type}): Title={title_font}pt, Content={content_font}pt")
            else:
                subtitle_font = slide_info.get("subtitle_font_size")
                print(f"Slide {slide_idx} ({slide_type}): Title={title_font}pt, Subtitle={subtitle_font}pt")
        
    except Exception as e:
        print(f"❌ Font alignment optimization failed: {e}")
        print("⚠️ Proceeding with initial preview.json")

    print("\n" + "="*60)
    print("🎉 LLM Processing Complete")
    print("="*60)
    print(f"📁 Output files:")
    print(f"   - {preview_path}")
    if 'alignment_path' in locals():
        print(f"   - {alignment_path}")
    print("="*60)


# ==============================
# Run Example
# ==============================
if __name__ == "__main__":
    service()