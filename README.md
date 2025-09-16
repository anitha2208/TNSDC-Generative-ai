"""
excel_to_ppt_template.py
Use a PowerPoint template + Excel (or generated text) to create slides
preserving template fonts, colors and styles.
"""

from pptx import Presentation
from pptx.util import Pt
import pandas as pd
import os


def generate_ppt_from_template(template_path, excel_path, output_path,
                               title_column='Title', content_column='Description',
                               layout_index=1, autosize=True):
    """
    Generate a PPT from an uploaded template and Excel file.

    template_path : str   Path to uploaded PowerPoint template (pptx)
    excel_path    : str   Path to Excel file containing content
    output_path   : str   Path to save the generated PowerPoint file
    title_column  : str   Column name in Excel for slide titles
    content_column: str   Column name in Excel for slide content
    layout_index  : int   Which layout from the template to use
    autosize      : bool  Whether to auto-fit long text
    """

    # --- 1. Load template ---
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    prs = Presentation(template_path)

    # --- 2. Load Excel data ---
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    df = pd.read_excel(excel_path)

    # --- 3. Choose layout from template ---
    try:
        slide_layout = prs.slide_layouts[layout_index]
    except IndexError:
        raise ValueError(f"Layout index {layout_index} not found in template")

    # --- 4. Loop through Excel rows and add slides ---
    for i, row in df.iterrows():
        slide = prs.slides.add_slide(slide_layout)

        # TITLE PLACEHOLDER – inherits font & style from template
        if len(slide.placeholders) > 0 and title_column in df.columns:
            slide.placeholders[0].text = str(row[title_column])

        # CONTENT PLACEHOLDER – inherits font & style from template
        if len(slide.placeholders) > 1 and content_column in df.columns:
            placeholder = slide.placeholders[1]
            placeholder.text = str(row[content_column])

            # Optional: wrap and auto-size long text
            if autosize:
                text_frame = placeholder.text_frame
                text_frame.word_wrap = True
                # Enable PowerPoint to auto-resize text to fit
                text_frame.auto_size = True

    # --- 5. Save the new PowerPoint ---
    prs.save(output_path)
    print(f"[INFO] PPT generated successfully at: {output_path}")


# Example usage (backend)
if __name__ == "__main__":
    # Replace these with your actual paths (UID-based)
    uid = "user123"
    template_path = f"storage/uploads/{uid}/template.pptx"
    excel_path = f"storage/uploads/{uid}/input.xlsx"
    output_path = f"storage/downloads/{uid}/output.pptx"

    # Generate PPT
    generate_ppt_from_template(
        template_path=template_path,
        excel_path=excel_path,
        output_path=output_path,
        title_column='Title',        # must match your Excel header
        content_column='Description' # must match your Excel header
    )
