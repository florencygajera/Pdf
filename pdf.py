import os
import io
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from langdetect import detect
import argparse
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import warnings
import numpy as np
import cv2
from collections import defaultdict
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class IndianLanguagePDFExtractor:
    """
    Enhanced PDF text extractor with comprehensive support for Indian languages
    including Hindi, Gujarati, Urdu, Tamil, Telugu, Bengali, Marathi, and more.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the PDF extractor with Indian language support.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Comprehensive Indian language support with Tesseract codes
        self.indian_languages = {
            # Major Indian languages
            'hin': {
                'name': 'Hindi',
                'script': 'Devanagari',
                'tesseract_code': 'hin',
                'direction': 'ltr'
            },
            'ben': {
                'name': 'Bengali',
                'script': 'Bengali',
                'tesseract_code': 'ben',
                'direction': 'ltr'
            },
            'tel': {
                'name': 'Telugu',
                'script': 'Telugu',
                'tesseract_code': 'tel',
                'direction': 'ltr'
            },
            'tam': {
                'name': 'Tamil',
                'script': 'Tamil',
                'tesseract_code': 'tam',
                'direction': 'ltr'
            },
            'mar': {
                'name': 'Marathi',
                'script': 'Devanagari',
                'tesseract_code': 'mar',
                'direction': 'ltr'
            },
            'guj': {
                'name': 'Gujarati',
                'script': 'Gujarati',
                'tesseract_code': 'guj',
                'direction': 'ltr'
            },
            'kan': {
                'name': 'Kannada',
                'script': 'Kannada',
                'tesseract_code': 'kan',
                'direction': 'ltr'
            },
            'mal': {
                'name': 'Malayalam',
                'script': 'Malayalam',
                'tesseract_code': 'mal',
                'direction': 'ltr'
            },
            'ori': {
                'name': 'Odia',
                'script': 'Odia',
                'tesseract_code': 'ori',
                'direction': 'ltr'
            },
            'pan': {
                'name': 'Punjabi',
                'script': 'Gurmukhi',
                'tesseract_code': 'pan',
                'direction': 'ltr'
            },
            'urd': {
                'name': 'Urdu',
                'script': 'Perso-Arabic',
                'tesseract_code': 'urd',
                'direction': 'rtl'  # Right-to-left script
            },
            'san': {
                'name': 'Sanskrit',
                'script': 'Devanagari',
                'tesseract_code': 'san',
                'direction': 'ltr'
            },
            'kas': {
                'name': 'Kashmiri',
                'script': 'Perso-Arabic',
                'tesseract_code': 'kas',
                'direction': 'rtl'
            },
            'nep': {
                'name': 'Nepali',
                'script': 'Devanagari',
                'tesseract_code': 'nep',
                'direction': 'ltr'
            },
            'bod': {
                'name': 'Tibetan',
                'script': 'Tibetan',
                'tesseract_code': 'bod',
                'direction': 'ltr'
            },
            'mya': {
                'name': 'Burmese',
                'script': 'Myanmar',
                'tesseract_code': 'mya',
                'direction': 'ltr'
            }
        }
        
        # Language groups for better OCR
        self.language_groups = {
            'north_indian': ['hin', 'mar', 'nep', 'san'],
            'south_indian': ['tel', 'tam', 'kan', 'mal'],
            'east_indian': ['ben', 'ori'],
            'west_indian': ['guj', 'pan'],
            'perso_arabic': ['urd', 'kas']
        }
    
    def extract_from_pdf(self, pdf_path: str, languages: List[str] = None, 
                        preprocessing_level: str = 'medium',
                        extract_images: bool = True,
                        save_debug_images: bool = False) -> Dict[str, Any]:
        """
        Extract text from PDF with enhanced Indian language support.
        
        Args:
            pdf_path: Path to the PDF file
            languages: List of language codes (e.g., ['hin', 'guj', 'urd'])
            preprocessing_level: 'low', 'medium', or 'high' - level of image preprocessing
            extract_images: Whether to extract text from images
            save_debug_images: Save preprocessed images for debugging
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if languages is None:
            languages = ['hin', 'eng']  # Default to Hindi + English
            
        result = {
            'filename': os.path.basename(pdf_path),
            'extraction_date': datetime.now().isoformat(),
            'text': '',
            'pages': [],
            'metadata': {},
            'statistics': {
                'total_pages': 0,
                'pages_with_text': 0,
                'images_processed': 0,
                'languages_detected': []
            }
        }
        
        try:
            # Extract metadata using PyMuPDF
            doc = fitz.open(pdf_path)
            result['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': len(doc)
            }
            
            result['statistics']['total_pages'] = len(doc)
            
            # Process each page
            for page_num in range(len(doc)):
                print(f"Processing page {page_num + 1}/{len(doc)}...")
                
                page = doc[page_num]
                
                # Get text directly from PDF
                page_text = page.get_text()
                
                page_data = {
                    'page_number': page_num + 1,
                    'direct_text': page_text,
                    'images_text': [],
                    'has_images': False,
                    'detected_languages': []
                }
                
                # Extract text from images if requested
                if extract_images:
                    image_texts = self._extract_text_from_images(
                        page, languages, preprocessing_level, save_debug_images
                    )
                    
                    if image_texts:
                        page_data['images_text'] = image_texts
                        page_data['has_images'] = True
                        result['statistics']['images_processed'] += len(image_texts)
                
                # Detect languages in the page
                page_languages = self._detect_languages_in_text(
                    page_data['direct_text'] + ' ' + 
                    ' '.join([img['text'] for img in page_data['images_text']])
                )
                page_data['detected_languages'] = page_languages
                
                # Update statistics
                if page_data['direct_text'].strip() or page_data['images_text']:
                    result['statistics']['pages_with_text'] += 1
                
                for lang in page_languages:
                    if lang not in result['statistics']['languages_detected']:
                        result['statistics']['languages_detected'].append(lang)
                
                result['pages'].append(page_data)
            
            doc.close()
            
            # Compile all text
            result['text'] = self._compile_text(result)
            
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        
        return result
    
    def _extract_text_from_images(self, page: fitz.Page, languages: List[str],
                                 preprocessing_level: str,
                                 save_debug_images: bool) -> List[Dict]:
        """
        Extract text from images on a page.
        
        Args:
            page: PyMuPDF page object
            languages: List of language codes
            preprocessing_level: Level of image preprocessing
            save_debug_images: Whether to save debug images
            
        Returns:
            List of dictionaries containing extracted text from images
        """
        image_texts = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # can save as PNG
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Preprocess image based on level
                    processed_image = self._preprocess_image(
                        pil_image, preprocessing_level
                    )
                    
                    if save_debug_images:
                        debug_dir = 'debug_images'
                        os.makedirs(debug_dir, exist_ok=True)
                        processed_image.save(
                            f"{debug_dir}/page_{page.number + 1}_img_{img_index}.png"
                        )
                    
                    # Perform OCR with Indian languages
                    ocr_result = self._perform_multilingual_ocr(
                        processed_image, languages
                    )
                    
                    if ocr_result['text'].strip():
                        image_texts.append({
                            'image_index': img_index,
                            'text': ocr_result['text'],
                            'confidence': ocr_result['confidence'],
                            'detected_languages': ocr_result['detected_languages'],
                            'bbox': img[1:5] if len(img) > 4 else None
                        })
                
                pix = None  # Free memory
                
            except Exception as e:
                print(f"Error processing image {img_index} on page {page.number + 1}: {e}")
                continue
        
        return image_texts
    
    def _preprocess_image(self, image: Image.Image, level: str = 'medium') -> Image.Image:
        """
        Enhanced image preprocessing for Indian language scripts.
        
        Args:
            image: PIL Image object
            level: Preprocessing level ('low', 'medium', 'high')
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if level == 'low':
            # Basic preprocessing
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = Image.fromarray(thresh)
            
        elif level == 'medium':
            # Medium preprocessing - good for most Indian scripts
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Adaptive thresholding (better for varying lighting)
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Remove small noise
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            processed = Image.fromarray(cleaned)
            
        else:  # high level
            # Advanced preprocessing for difficult documents
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Sharpen
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 3
            )
            
            # Advanced morphological operations
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            processed = Image.fromarray(cleaned)
        
        return processed
    
    def _perform_multilingual_ocr(self, image: Image.Image, 
                                 languages: List[str]) -> Dict[str, Any]:
        """
        Perform OCR with support for multiple Indian languages.
        
        Args:
            image: Preprocessed PIL Image
            languages: List of language codes
            
        Returns:
            Dictionary with OCR results
        """
        result = {
            'text': '',
            'confidence': 0.0,
            'detected_languages': []
        }
        
        try:
            # Prepare language string for Tesseract
            # Map our language codes to Tesseract codes
            tesseract_langs = []
            for lang in languages:
                if lang in self.indian_languages:
                    tesseract_langs.append(self.indian_languages[lang]['tesseract_code'])
                else:
                    tesseract_langs.append(lang)  # Assume it's already a Tesseract code
            
            # Add English as fallback if not present
            if 'eng' not in tesseract_langs:
                tesseract_langs.append('eng')
            
            lang_str = '+'.join(tesseract_langs)
            
            # Perform OCR with language-specific configuration
            # Custom config for Indian languages
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist='
            
            # Get OCR data with confidence
            ocr_data = pytesseract.image_to_data(
                image, lang=lang_str, output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    text_parts.append(text)
                    conf = int(ocr_data['conf'][i])
                    if conf > 0:
                        confidences.append(conf)
            
            result['text'] = ' '.join(text_parts)
            
            if confidences:
                result['confidence'] = sum(confidences) / len(confidences)
            
            # Detect languages in the extracted text
            if result['text'].strip():
                result['detected_languages'] = self._detect_languages_in_text(
                    result['text']
                )
            
        except Exception as e:
            print(f"OCR Error: {e}")
        
        return result
    
    def _detect_languages_in_text(self, text: str) -> List[str]:
        """
        Detect Indian languages in text.
        
        Args:
            text: Input text
            
        Returns:
            List of detected language codes
        """
        detected = []
        
        try:
            # Use langdetect for language detection
            # Note: langdetect has limited Indian language support
            lang = detect(text)
            if lang in self.indian_languages or lang == 'en':
                detected.append(lang)
        except:
            pass
        
        # Additional script-based detection for Indian languages
        if not detected:
            # Simple script detection based on Unicode ranges
            scripts = self._detect_script(text)
            for script in scripts:
                for lang_code, lang_info in self.indian_languages.items():
                    if lang_info['script'] == script and lang_code not in detected:
                        detected.append(lang_code)
        
        return detected
    
    def _detect_script(self, text: str) -> List[str]:
        """
        Detect script type based on Unicode ranges.
        
        Args:
            text: Input text
            
        Returns:
            List of detected script names
        """
        scripts = []
        
        # Unicode ranges for Indian scripts
        script_ranges = {
            'Devanagari': (0x0900, 0x097F),
            'Bengali': (0x0980, 0x09FF),
            'Gurmukhi': (0x0A00, 0x0A7F),
            'Gujarati': (0x0A80, 0x0AFF),
            'Oriya': (0x0B00, 0x0B7F),
            'Tamil': (0x0B80, 0x0BFF),
            'Telugu': (0x0C00, 0x0C7F),
            'Kannada': (0x0C80, 0x0CFF),
            'Malayalam': (0x0D00, 0x0D7F),
            'Perso-Arabic': (0x0600, 0x06FF)  # For Urdu
        }
        
        for char in text:
            code = ord(char)
            for script, (start, end) in script_ranges.items():
                if start <= code <= end and script not in scripts:
                    scripts.append(script)
                    break
        
        return scripts
    
    def _compile_text(self, result: Dict[str, Any]) -> str:
        """
        Compile all extracted text into a formatted string.
        
        Args:
            result: Result dictionary from extraction
            
        Returns:
            Formatted text string
        """
        compiled = []
        
        # Add header
        compiled.append("=" * 80)
        compiled.append(f"PDF TEXT EXTRACTION REPORT")
        compiled.append(f"File: {result['filename']}")
        compiled.append(f"Date: {result['extraction_date']}")
        compiled.append("=" * 80)
        compiled.append("")
        
        # Add statistics
        compiled.append("EXTRACTION STATISTICS:")
        compiled.append(f"  Total Pages: {result['statistics']['total_pages']}")
        compiled.append(f"  Pages with Text: {result['statistics']['pages_with_text']}")
        compiled.append(f"  Images Processed: {result['statistics']['images_processed']}")
        compiled.append(f"  Languages Detected: {', '.join(result['statistics']['languages_detected'])}")
        compiled.append("")
        
        # Add metadata
        if result['metadata']:
            compiled.append("DOCUMENT METADATA:")
            for key, value in result['metadata'].items():
                if value:
                    compiled.append(f"  {key}: {value}")
            compiled.append("")
        
        # Add page contents
        for page in result['pages']:
            compiled.append("-" * 60)
            compiled.append(f"PAGE {page['page_number']}")
            compiled.append(f"Languages detected: {', '.join(page['detected_languages'])}")
            compiled.append("-" * 60)
            
            # Add direct text
            if page['direct_text'].strip():
                compiled.append("\n[DIRECT TEXT FROM PDF]:")
                compiled.append(page['direct_text'].strip())
            
            # Add text from images
            if page['images_text']:
                compiled.append("\n[TEXT EXTRACTED FROM IMAGES]:")
                for img_text in page['images_text']:
                    compiled.append(f"\n--- Image {img_text['image_index'] + 1} ---")
                    compiled.append(f"Confidence: {img_text['confidence']:.2f}%")
                    if img_text['detected_languages']:
                        compiled.append(f"Detected languages: {', '.join(img_text['detected_languages'])}")
                    compiled.append(img_text['text'])
            
            compiled.append("\n")
        
        return '\n'.join(compiled)
    
    def save_output(self, result: Dict[str, Any], output_format: str = 'txt',
                   output_path: Optional[str] = None):
        """
        Save extraction results in various formats.
        
        Args:
            result: Result dictionary from extraction
            output_format: 'txt', 'json', or 'both'
            output_path: Output file path (optional)
        """
        if not output_path:
            base_name = os.path.splitext(result['filename'])[0]
            output_path = base_name
        
        if output_format in ['txt', 'both']:
            txt_path = f"{output_path}_extracted.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            print(f"Text saved to: {txt_path}")
        
        if output_format in ['json', 'both']:
            json_path = f"{output_path}_extracted.json"
            # Convert to serializable format
            serializable_result = {
                'filename': result['filename'],
                'extraction_date': result['extraction_date'],
                'metadata': result['metadata'],
                'statistics': result['statistics'],
                'pages': []
            }
            
            for page in result['pages']:
                serializable_page = {
                    'page_number': page['page_number'],
                    'direct_text': page['direct_text'],
                    'detected_languages': page['detected_languages'],
                    'images_text': [
                        {
                            'image_index': img['image_index'],
                            'text': img['text'],
                            'confidence': img['confidence'],
                            'detected_languages': img['detected_languages']
                        }
                        for img in page['images_text']
                    ]
                }
                serializable_result['pages'].append(serializable_page)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            print(f"JSON saved to: {json_path}")
    
    def print_languages(self):
        """Print supported Indian languages"""
        print("\n" + "="*60)
        print("SUPPORTED INDIAN LANGUAGES")
        print("="*60)
        
        for code, info in self.indian_languages.items():
            print(f"{code}: {info['name']} (Script: {info['script']}, Direction: {info['direction']})")
        
        print("\nLanguage Groups:")
        for group, langs in self.language_groups.items():
            lang_names = [self.indian_languages[lang]['name'] for lang in langs]
            print(f"  {group}: {', '.join(lang_names)}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDF with Indian language support'
    )
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output file path (without extension)')
    parser.add_argument('--format', '-f', choices=['txt', 'json', 'both'], 
                       default='txt', help='Output format')
    parser.add_argument('--languages', '-l', nargs='+', 
                       default=['hin', 'eng', 'guj', 'urd'],
                       help='Languages for OCR (e.g., hin eng guj urd)')
    parser.add_argument('--preprocessing', '-p', choices=['low', 'medium', 'high'],
                       default='medium', help='Image preprocessing level')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip text extraction from images')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug images')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--list-languages', action='store_true',
                       help='List supported Indian languages and exit')
    
    args = parser.parse_args()
    
    if args.list_languages:
        extractor = IndianLanguagePDFExtractor(args.tesseract_path)
        extractor.print_languages()
        return
    
    # Create extractor instance
    extractor = IndianLanguagePDFExtractor(args.tesseract_path)
    
    print(f"\nProcessing PDF: {args.pdf_path}")
    print(f"Languages selected: {', '.join(args.languages)}")
    print(f"Preprocessing level: {args.preprocessing}")
    print("-" * 50)
    
    # Extract text
    result = extractor.extract_from_pdf(
        pdf_path=args.pdf_path,
        languages=args.languages,
        preprocessing_level=args.preprocessing,
        extract_images=not args.no_images,
        save_debug_images=args.debug
    )
    
    # Check for errors
    if 'error' in result:
        print(f"\nError occurred: {result['error']}")
        if 'traceback' in result:
            print(f"Traceback: {result['traceback']}")
        return
    
    # Save output
    extractor.save_output(result, args.format, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total pages processed: {result['statistics']['total_pages']}")
    print(f"Pages with text: {result['statistics']['pages_with_text']}")
    print(f"Images processed: {result['statistics']['images_processed']}")
    print(f"Languages detected: {', '.join(result['statistics']['languages_detected'])}")
    print("="*50)

if __name__ == "__main__":
    main()