# Create test_pdf_generator.py
from fpdf import FPDF, XPos, YPos
import os

class PDFWithIndianLanguages(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Indian Languages Test Document', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

# Create PDF
pdf = PDFWithIndianLanguages()
pdf.add_page()
pdf.set_font('Helvetica', '', 12)

# Add text in different languages (ASCII representation)
pdf.cell(0, 10, 'English: This is a test document for OCR extraction', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

# Note: For proper rendering, you need proper Unicode fonts
# Using ASCII representation for testing purposes
pdf.cell(0, 10, 'Hindi: Test Hindi text (requires Unicode font)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

pdf.cell(0, 10, 'Gujarati: Test Gujarati text (requires Unicode font)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

pdf.cell(0, 10, 'Urdu: Test Urdu text (requires Unicode font)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save the PDF
pdf.output('test_indian_languages.pdf')
print("Test PDF created: test_indian_languages.pdf")