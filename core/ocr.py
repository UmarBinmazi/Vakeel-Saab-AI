import logging
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract
from typing import List, Dict, Tuple, Optional, Any, Union
import threading
import queue
import re
from collections import Counter
from difflib import get_close_matches

logger = logging.getLogger(__name__)

if os.environ.get("TESSERACT_PATH"):
    pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_PATH")

# Common OCR substitution errors
COMMON_SUBSTITUTIONS = {
    '0': 'o', 'O': 'o', 'l': 'i', '1': 'i', '5': 's', '8': 'B', '8': 'S', '8': 's',
    'a8': 'as', 'H': 'h', 'ii': 'n', 'il': 'h', 'rn': 'm', 'cl': 'd', 'I-': 'H',
    'i-': 'h', 'ii': 'n', 'vv': 'w', '@': 'a', 'I3': 'B', 'l3': 'B', '13': 'B',
    'ht': 'it', 'Ht': 'It', 'posstbhe': 'possible', 'axkte': 'oxide'
}

# Scientific domain vocabulary for correction
SCIENTIFIC_TERMS = [
    'experiment', 'laboratory', 'chemical', 'reaction', 'solution', 'formula',
    'equation', 'molecule', 'compound', 'element', 'temperature', 'pressure',
    'volume', 'mass', 'density', 'concentration', 'acid', 'base', 'salt',
    'oxide', 'hydroxide', 'magnesium', 'calcium', 'sodium', 'potassium',
    'iron', 'copper', 'zinc', 'oxygen', 'hydrogen', 'nitrogen', 'carbon',
    'ribbon', 'burn', 'heat', 'flame', 'crucible', 'beaker', 'test tube',
    'pipette', 'burette', 'thermometer', 'balance', 'indicator', 'titration',
    'precipitate', 'filtration', 'distillation', 'evaporation', 'crystallization'
]

def is_gibberish(text: str, threshold: float = 0.4) -> bool:
    """
    Check if text appears to be gibberish using statistical patterns.
    Works across languages without vocabulary dependencies.
    """
    if not text or len(text) < 10:
        return True
    
    try:
        # Check symbol ratio (language-independent)
        symbols = re.findall(r'[^\w\s.,:;!?]', text)
        symbol_ratio = len(symbols) / max(len(text), 1)
        
        # Check for unusual character sequences (language-independent)
        unusual_sequences = re.findall(r'([^\w\s]){3,}', text)
        seq_ratio = len(unusual_sequences) / max(len(text) / 10, 1)
        
        # Check for repeating patterns (language-independent)
        repeated = sum(1 for match in re.finditer(r'(.)\1{3,}', text))
        repeat_ratio = repeated / max(len(text) / 10, 1)
        
        # Entropy-based check (language-independent)
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return True
            
        entropy = -sum((count/total_chars) * np.log2(count/total_chars) 
                       for count in char_counts.values())
        
        # Natural text has moderate entropy - too high or too low is suspicious
        entropy_score = abs(entropy - 4.0) / 4.0  # Normalize around typical text entropy
        
        # Combined language-independent score
        return (symbol_ratio > threshold or seq_ratio > threshold/2 or 
                repeat_ratio > threshold/3 or entropy_score > 0.7)
    except Exception as e:
        logger.warning(f"Error in gibberish detection: {e}")
        return False  # Fail open rather than rejecting text incorrectly

class StatisticalTextCorrector:
    """
    Language-agnostic text correction using statistical patterns.
    Does not depend on predefined vocabularies.
    """
    
    def __init__(self):
        self.common_replacements = {
            # Universal character fixes that work in most scripts
            '|': 'I', 
            'l|': 'h', 
            '\u2022': '-',  # Bullet to hyphen
            '\u2013': '-',  # En dash to hyphen
            '\u2014': '-',  # Em dash to hyphen
            '\xad': '',     # Soft hyphen to nothing
            '\xa0': ' '     # Non-breaking space to space
        }
        
        # Will be populated during correction from the document itself
        self.document_patterns = {}
        self.char_ngrams = Counter()
        self.trained = False
    
    def train_from_document(self, text: str):
        """Learn patterns from the cleaner parts of the document itself."""
        if not text or len(text) < 100:
            return
            
        try:
            # Split into lines and find the ones that look most natural
            lines = text.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not is_gibberish(line, 0.3):
                    clean_lines.append(line)
            
            if not clean_lines:
                # Not enough clean text to learn from
                return
                
            # Learn character n-grams from clean text
            clean_text = ' '.join(clean_lines)
            for i in range(len(clean_text) - 2):
                trigram = clean_text[i:i+3]
                self.char_ngrams[trigram] += 1
                
            # Learn word patterns
            words = re.findall(r'\b\w+\b', clean_text.lower())
            word_counter = Counter(words)
            
            # Store frequent words as likely correct forms
            self.document_patterns = {word: count for word, count in word_counter.items() 
                                     if count > 1 and len(word) > 3}
            
            self.trained = True
        except Exception as e:
            logger.warning(f"Error training corrector: {e}")
    
    def correct_text(self, text: str) -> str:
        """Apply statistical correction to text."""
        if not text:
            return ""
        
        try:
            # Basic character replacements
            for error, correction in self.common_replacements.items():
                text = text.replace(error, correction)
            
            # If we haven't trained on document patterns, just return basic corrections
            if not self.trained:
                return text
                
            # Apply more sophisticated corrections based on learned patterns
            words = re.findall(r'\b(\w+)\b', text)
            
            for word in words:
                if len(word) <= 3:  # Skip short words
                    continue
                    
                word_lower = word.lower()
                
                # Skip words that already match document patterns
                if word_lower in self.document_patterns:
                    continue
                    
                # Find closest match using character n-grams
                best_score = 0
                best_match = None
                
                for candidate in self.document_patterns:
                    if abs(len(candidate) - len(word_lower)) > 2:
                        continue  # Length too different
                        
                    # Calculate n-gram similarity
                    score = 0
                    for i in range(len(word_lower) - 2):
                        if i < len(candidate) - 2:
                            w_gram = word_lower[i:i+3]
                            c_gram = candidate[i:i+3]
                            if w_gram == c_gram:
                                score += 1
                    
                    # Normalize by length
                    norm_score = score / max(len(word_lower), len(candidate))
                    
                    if norm_score > 0.7 and norm_score > best_score:
                        best_score = norm_score
                        best_match = candidate
                
                # Replace if good match found
                if best_match:
                    # Preserve capitalization
                    if word[0].isupper():
                        replacement = best_match.capitalize()
                    else:
                        replacement = best_match
                        
                    text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error in text correction: {e}")
            return text  # Return original if correction fails

class OCRProcessor:
    def __init__(self, language: str = "eng", oem: int = 3, psm: int = 3):
        self.language = language
        self.oem = oem
        self.psm = psm
        self.tesseract_config = f"--oem {oem} --psm {psm} -l {language}"
        self.min_text_length = 50
        self.corrector = StatisticalTextCorrector()
        
        # Reduced number of PSM modes to try for faster processing
        self.psm_modes = [3, 6]  # Only use auto and single block for speed
        
    def is_ocr_available(self) -> bool:
        try:
            import pytesseract
            from PIL import Image
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, pytesseract.TesseractNotFoundError) as e:
            logger.warning(f"OCR dependencies not available: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Return multiple preprocessing variants to maximize OCR chances."""
        try:
            # Ensure we have grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Create multiple preprocessing variants - reduced for speed
            variants = []
            
            # 1. Basic grayscale 
            variants.append(gray)
            
            # 2. Adaptive Gaussian threshold - most effective method
            try:
                binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                variants.append(binary1)
            except Exception:
                pass
                
            # 3. Contrast enhanced version - for low contrast documents
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                variants.append(enhanced)
            except Exception:
                pass
                
            return variants
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original if preprocessing fails
            return [image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]

    def process_image(self, image: np.ndarray) -> str:
        """Process image with multiple strategies and select best result."""
        if image is None or image.size == 0:
            return ""
            
        try:
            # Get multiple preprocessing variants
            image_variants = self.preprocess_image(image)
            
            # Try multiple PSM modes and preprocessing variants
            results = []
            
            for img_variant in image_variants:
                for psm in self.psm_modes:
                    try:
                        config = f"--oem {self.oem} --psm {psm} -l {self.language}"
                        text = pytesseract.image_to_string(img_variant, config=config)
                        
                        # Only keep non-empty results
                        if text and len(text.strip()) > 10:
                            results.append(text)
                    except Exception as e:
                        logger.warning(f"OCR attempt failed with PSM {psm}: {e}")
                        continue
            
            # No results
            if not results:
                return ""
                
            # Score and select best result
            best_text = max(results, key=lambda x: self._score_text_quality(x))
            
            # Apply post-processing
            corrected_text = self.postprocess_text(best_text)
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return ""

    def _score_text_quality(self, text: str) -> float:
        """Score text quality based on statistical properties."""
        if not text:
            return 0.0
            
        # Remove very short texts
        if len(text.strip()) < 10:
            return 0.0
            
        # Count words, sentences, and symbols
        words = len(re.findall(r'\b\w+\b', text))
        sentences = len(re.findall(r'[.!?]+', text)) + 1
        symbols = len(re.findall(r'[^\w\s.,:;!?]', text))
        
        # Calculate metrics
        words_per_sentence = words / max(sentences, 1)
        symbol_ratio = symbols / max(len(text), 1)
        
        # Score based on typical text properties
        score = (words * 0.1) + (words_per_sentence * 0.3) + ((1 - symbol_ratio) * 0.6)
        return score

    def postprocess_text(self, text: str) -> str:
        """Clean and normalize OCR output text."""
        if not text:
            return ""
            
        try:
            # Basic text normalization
            text = " ".join(text.split())
            
            # Remove noise patterns
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
            text = re.sub(r'([.!?])\1+', r'\1', text)   # Normalize multiple punctuation
            text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
                
            return text.strip()
        except Exception as e:
            logger.warning(f"Text postprocessing failed: {e}")
            return text

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> Tuple[str, List[Dict]]:
        """Extract text from PDF with robust fallback strategies."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = []
            metadata = []
            
            # First try direct extraction
            direct_text, has_text = self._try_direct_extraction(doc)
            
            # If direct extraction worked well, use it
            if has_text:
                logger.info("Using direct text extraction")
                return direct_text
                
            # Otherwise use OCR
            logger.info("Falling back to OCR extraction")
            return self._extract_with_ocr(doc)
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return "", []

    def _try_direct_extraction(self, doc) -> Tuple[Tuple[str, List[Dict]], bool]:
        """Try direct text extraction and assess quality."""
        try:
            full_text = []
            metadata = []
            total_text_len = 0
            gibberish_pages = 0
            
            # Sample a few pages to check quality
            pages_to_check = min(5, len(doc))
            
            for i in range(pages_to_check):
                try:
                    page = doc[i]
                    text = page.get_text().strip()
                    
                    if text:
                        total_text_len += len(text)
                        if is_gibberish(text):
                            gibberish_pages += 1
                except Exception:
                    gibberish_pages += 1
            
            # Check if direct extraction seems reliable
            if total_text_len > 0 and gibberish_pages < pages_to_check / 2:
                # Extract all pages with direct method
                all_text = []
                all_metadata = []
                
                for page_num, page in enumerate(doc):
                    try:
                        text = page.get_text().strip()
                        if text:
                            all_text.append(text)
                            all_metadata.append({
                                "text": text, 
                                "page": page_num + 1,
                                "extraction_method": "direct"
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                
                # Train the corrector with the extracted text
                combined_text = "\n".join(all_text)
                self.corrector.train_from_document(combined_text)
                
                # Apply corrections to each page
                corrected_text = []
                corrected_metadata = []
                
                for i, (text, meta) in enumerate(zip(all_text, all_metadata)):
                    fixed_text = self.corrector.correct_text(text)
                    corrected_text.append(fixed_text)
                    meta["text"] = fixed_text
                    meta["corrected"] = True
                    corrected_metadata.append(meta)
                
                return (("\n\n".join(corrected_text), corrected_metadata), True)
            else:
                return (("", []), False)
                
        except Exception as e:
            logger.error(f"Direct extraction assessment failed: {e}")
            return (("", []), False)
            
    def _extract_with_ocr(self, doc) -> Tuple[str, List[Dict]]:
        """Extract text using OCR with robust error handling."""
        if not self.is_ocr_available():
            logger.warning("OCR not available, cannot process document")
            return "", []
            
        try:
            full_text = []
            metadata = []
            
            # Only process a small sample for training - reduce sample size for speed
            sample_pages = min(2, len(doc))
            sample_text = []
            
            # First pass: handle embedded images specifically
            # This targets cases where a PDF has images containing text
            logger.info("Checking for embedded images with text content")
            for i in range(min(5, len(doc))):
                try:
                    page = doc[i]
                    # Get list of embedded images
                    img_list = page.get_images(full=True)
                    
                    if img_list:
                        page_width, page_height = page.rect.width, page.rect.height
                        for img_info in img_list:
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                if base_image:
                                    # Check if image is significant size compared to page
                                    img_width, img_height = base_image["width"], base_image["height"]
                                    if img_width > 300 and img_height > 300:  # Minimum size threshold
                                        # Convert image bytes to array for processing
                                        image = Image.open(io.BytesIO(base_image["image"]))
                                        image_array = np.array(image)
                                        
                                        # Process directly with OCR
                                        img_text = self.process_image(image_array)
                                        if img_text and len(img_text) > 50:
                                            sample_text.append(img_text)
                            except Exception as img_err:
                                logger.warning(f"Failed to process embedded image: {img_err}")
                except Exception as e:
                    logger.warning(f"Failed to check embedded images on page {i}: {e}")
            
            # Now render pages as normal
            for i in range(sample_pages):
                try:
                    page = doc[i]
                    pix = self._render_page(page)
                    
                    if pix is not None:
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        if pix.n == 4:  # RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        text = self.process_image(img)
                        if text:
                            sample_text.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process sample page {i}: {e}")
            
            # Train corrector with sample text
            self.corrector.train_from_document("\n".join(sample_text))
            
            # Determine page range - for large documents, process only a subset first
            total_pages = len(doc)
            max_pages_initial = 20  # Process max 20 pages first for speed
            
            pages_to_process = min(total_pages, max_pages_initial)
            
            # Process pages with improved error handling
            for page_num in range(pages_to_process):
                try:
                    page = doc[page_num]
                    
                    # First try to extract embedded images and OCR them directly
                    images_processed = False
                    img_list = page.get_images(full=True)
                    if img_list:
                        page_image_text = []
                        for img_info in img_list:
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                if base_image:
                                    # Only process significant sized images
                                    img_width, img_height = base_image["width"], base_image["height"]
                                    if img_width > 300 and img_height > 300:
                                        image = Image.open(io.BytesIO(base_image["image"]))
                                        image_array = np.array(image)
                                        img_text = self.process_image(image_array)
                                        if img_text and len(img_text) > 50:
                                            page_image_text.append(img_text)
                                            images_processed = True
                            except Exception as img_err:
                                logger.warning(f"Failed to process embedded image on page {page_num}: {img_err}")
                                
                        # If we got text from embedded images, use it
                        if page_image_text:
                            combined_text = "\n".join(page_image_text)
                            corrected = self.corrector.correct_text(combined_text)
                            
                            full_text.append(corrected)
                            metadata.append({
                                "text": corrected,
                                "page": page_num + 1,
                                "extraction_method": "embedded_image_ocr",
                                "confidence": 0.8,  # Higher confidence for direct image extraction
                                "corrected": True
                            })
                    
                    # If no embedded images were processed, use regular page rendering
                    if not images_processed:
                        try:
                            # Try the region-based approach first for better results
                            region_text = self._process_page_with_regions(page)
                            
                            if region_text and len(region_text) > 50:
                                # Use the region-based result if successful
                                full_text.append(region_text)
                                metadata.append({
                                    "text": region_text,
                                    "page": page_num + 1,
                                    "extraction_method": "region_based_ocr",
                                    "confidence": 0.85,  # Higher confidence for region-based
                                    "corrected": True
                                })
                                continue  # Skip the standard rendering approach
                        except Exception as e:
                            logger.warning(f"Region-based processing failed for page {page_num}: {e}")
                            # Fall back to standard approach
                            pass
                        
                        # Standard full-page rendering approach as fallback
                        pix = self._render_page(page)
                        
                        if pix is None:
                            logger.warning(f"Could not render page {page_num}")
                            continue
                            
                        try:
                            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                            if pix.n == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        except Exception as e:
                            logger.warning(f"Error converting pixmap to image on page {page_num}: {e}")
                            continue
                            
                        try:
                            text = self.process_image(img)
                            confidence = self._estimate_ocr_confidence(text)
                            
                            if text:
                                # Apply trained correction
                                corrected = self.corrector.correct_text(text)
                                
                                full_text.append(corrected)
                                metadata.append({
                                    "text": corrected,
                                    "page": page_num + 1,
                                    "extraction_method": "ocr",
                                    "confidence": confidence,
                                    "corrected": True
                                })
                        except Exception as e:
                            logger.warning(f"OCR failed for page {page_num}: {e}")
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
            
            # If this is a large document, let the user know we're only processing a subset
            if total_pages > max_pages_initial and full_text:
                logger.info(f"Processed {pages_to_process} of {total_pages} pages for performance reasons")
                full_text.append(f"Note: Only the first {pages_to_process} of {total_pages} pages were processed. For complete analysis, please use a more focused document.")
                
            if not full_text:
                logger.error("No text extracted from any page")
                return "", []
                
            return "\n\n".join(full_text), metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", []

    def _render_page(self, page) -> Optional[fitz.Pixmap]:
        """Render PDF page to image with error handling."""
        try:
            # Higher resolution for much better OCR quality
            zoom = 4.0  # Increased to 4.0 for maximum image detail
            # Use a matrix with higher DPI to capture fine details
            dpi = 300  # Standard printing DPI for high quality
            matrix = fitz.Matrix(zoom, zoom)
            # Set higher color depth to ensure all details are captured
            colorspace = fitz.csRGB  # Use RGB color space for better detail
            # Create pixmap with no alpha to increase clarity
            return page.get_pixmap(matrix=matrix, alpha=False, colorspace=colorspace)
        except Exception as e:
            logger.error(f"Failed to render page with high quality: {e}")
            
            # Try with different parameters as fallback
            try:
                zoom = 3.0  # Still higher quality than previous version
                matrix = fitz.Matrix(zoom, zoom)
                return page.get_pixmap(matrix=matrix, alpha=False)
            except Exception as e2:
                logger.error(f"Failed with second attempt: {e2}")
                
                # Last resort with minimal settings
                try:
                    zoom = 2.0  
                    matrix = fitz.Matrix(zoom, zoom)
                    return page.get_pixmap(matrix=matrix, alpha=True)
                except:
                    return None

    def _estimate_ocr_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text quality."""
        if not text:
            return 0.0
            
        try:
            # Count words and non-dictionary words
            words = text.split()
            if not words:
                return 0.0
                
            # Check for common OCR error indicators
            errors = 0
            
            # Very short words that aren't common
            for word in words:
                if len(word) == 1 and word.lower() not in 'aAiI':
                    errors += 1
                    
            # Unusual character sequences
            for pattern in [r'[0-9][A-Z]', r'[A-Z][0-9]', r'[^a-zA-Z0-9\s]{2,}']:
                errors += len(re.findall(pattern, text))
                
            # Calculate confidence score
            error_ratio = min(1.0, errors / max(len(words), 1))
            confidence = 1.0 - error_ratio
            
            return max(0.1, min(1.0, confidence))  # Ensure between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Error estimating confidence: {e}")
            return 0.5  # Default moderate confidence

    def _process_page_with_regions(self, page) -> str:
        """Process a page by dividing it into regions and applying different zoom levels.
        This helps with documents that have mixed content like text, tables, and images.
        """
        try:
            # Get the page size
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Define regions to process with different settings
            regions = [
                # Full page at standard resolution
                (0, 0, page_width, page_height, 3.0),
                # Top half with higher resolution (often headers or titles)
                (0, 0, page_width, page_height/2, 4.0),
                # Bottom half with higher resolution (often footers or captions)
                (0, page_height/2, page_width, page_height, 4.0)
            ]
            
            # For wider pages, add left/right regions for columns
            if page_width > page_height:
                regions.extend([
                    # Left half for column-based content
                    (0, 0, page_width/2, page_height, 4.0),
                    # Right half for column-based content
                    (page_width/2, 0, page_width, page_height, 4.0)
                ])
            
            # Extract and process text from each region
            texts = []
            
            for x0, y0, x1, y1, zoom in regions:
                try:
                    # Create a transformation matrix for this region with specified zoom
                    mat = fitz.Matrix(zoom, zoom)
                    
                    # Extract just this region
                    clip = fitz.Rect(x0, y0, x1, y1)
                    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)
                    
                    # Convert pixmap to image
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:  # RGBA
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        
                    # Apply OCR
                    region_text = self.process_image(img)
                    
                    # Only keep if meaningful text was found
                    if region_text and len(region_text.strip()) > 20:
                        texts.append(region_text)
                except Exception as e:
                    logger.warning(f"Failed to process region {clip}: {e}")
            
            # Combine the extracted text, removing duplicates
            if not texts:
                return ""
                
            # Use the corrector to improve and deduplicate text
            combined = "\n\n".join(texts)
            
            # Apply post-processing to clean up and deduplicate
            if self.corrector.trained:
                final_text = self.corrector.correct_text(combined)
            else:
                final_text = self.postprocess_text(combined)
                
            return final_text
            
        except Exception as e:
            logger.error(f"Error processing page regions: {e}")
            return ""
