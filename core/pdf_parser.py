import logging
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional, Any
import io
import re
from core.ocr import OCRProcessor, is_gibberish, StatisticalTextCorrector

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.text_corrector = StatisticalTextCorrector()
        # Font types that often indicate encoding issues
        self.problematic_fonts = {
            'Symbol', 'ZapfDingbats', 'Wingdings', 'Webdings', 
            'MT Extra', 'Marlett', 'CID', 'Identity'
        }
        
    def process_pdf(self, pdf_bytes: bytes) -> Tuple[str, List[Dict], Dict]:
        try:
            doc_info = self._extract_document_info(pdf_bytes)
            
            # Check for custom fonts that might cause problems
            fonts_info = self._analyze_fonts(pdf_bytes)
            has_problematic_fonts = fonts_info.get('has_problematic_fonts', False)
            
            # First try standard text extraction
            text, metadata = self._extract_text(pdf_bytes)
            
            # If no text was extracted, try OCR immediately
            if not text or len(text.strip()) < 50:
                logger.info("Minimal text extracted, trying OCR")
                text, metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes)
            # Check if we need OCR
            elif self._has_encoding_issues(text) or has_problematic_fonts:
                logger.info(f"Text has encoding issues or problematic fonts detected, falling back to OCR")
                text, metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes)
            elif self._needs_ocr(pdf_bytes, text):  # Pass extracted text to help with decision
                logger.info("Document appears to be scanned or contains embedded images, using OCR")
                text, metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes)
            
            # Handle the case where no text was extracted either way
            if not text or len(text.strip()) < 20:
                logger.error("No usable text could be extracted from the document")
                return "", [], doc_info

            enhanced_metadata = self._enhance_metadata(metadata)
            return text, enhanced_metadata, doc_info
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            # Try OCR as a last resort
            try:
                logger.info("Attempting OCR as fallback after error")
                text, metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes)
                if text:
                    enhanced_metadata = self._enhance_metadata(metadata)
                    return text, enhanced_metadata, {"error": str(e), "recovery": "used OCR fallback"}
            except Exception as nested_e:
                logger.error(f"OCR fallback also failed: {nested_e}")
            
            return "", [], {"error": str(e)}

    def _extract_document_info(self, pdf_bytes: bytes) -> Dict:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            info = doc.metadata
            info['page_count'] = len(doc)
            info['encrypted'] = doc.is_encrypted
            if hasattr(doc, 'permissions'):
                info['permissions'] = doc.permissions
            if hasattr(doc, 'pdf_version'):
                info['pdf_version'] = doc.pdf_version
            return info
        except Exception as e:
            logger.error(f"Error extracting document info: {e}")
            return {'error': str(e)}
            
    def _analyze_fonts(self, pdf_bytes: bytes) -> Dict:
        """Analyze fonts used in the PDF to detect potential problems."""
        result = {
            'has_problematic_fonts': False,
            'font_types': set(),
            'embedded_fonts': 0,
            'total_fonts': 0
        }
        
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # Only check first few pages - reduced for performance
            pages_to_check = min(3, len(doc))
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                try:
                    fonts = page.get_fonts()
                    
                    if fonts:
                        result['total_fonts'] += len(fonts)
                        
                        for font in fonts:
                            font_name = font[3]
                            is_embedded = font[1] == 'Type1C' or font[1] == 'CIDFontType0C'
                            font_type = font[1]
                            
                            if is_embedded:
                                result['embedded_fonts'] += 1
                                
                            result['font_types'].add(font_type)
                            
                            # Quick check for problematic fonts
                            for prob_font in self.problematic_fonts:
                                if prob_font.lower() in font_name.lower():
                                    result['has_problematic_fonts'] = True
                                    # Early return for performance once we know there's an issue
                                    return result
                                    
                            # Check for custom encodings
                            if font[1] == 'Type3' or 'Identity' in font_name:
                                result['has_problematic_fonts'] = True
                                # Early return for performance
                                return result
                except Exception as e:
                    logger.warning(f"Error analyzing fonts on page {page_num}: {e}")
                    continue
                    
                # Test extract text from this page
                try:
                    text = page.get_text().strip()
                    if is_gibberish(text):
                        result['has_problematic_fonts'] = True
                        # Early return for performance
                        return result
                except Exception:
                    result['has_problematic_fonts'] = True
                    
            # If no embedded fonts but has fonts, consider problematic
            if result['total_fonts'] > 0 and result['embedded_fonts'] == 0:
                result['has_problematic_fonts'] = True
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing fonts: {e}")
            return {'has_problematic_fonts': True, 'error': str(e)}

    def _needs_ocr(self, pdf_bytes: bytes, text: str = None) -> bool:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # Only check first 2 pages for better performance
            pages_to_check = min(2, len(doc))
            text_length = 0
            has_encoding_issues = False
            has_significant_images = False
            text_to_image_ratio = 1.0  # Default ratio (higher means more text than images)
            structured_content_detected = False
            
            # If text was already extracted, analyze its quality
            if text:
                # Check for suspicious patterns that suggest image-based text
                suspicious_patterns = [
                    r'[^\w\s.,;:?!()-]',  # Non-standard characters
                    r'[\u0000-\u001F]',    # Control characters
                    r'[^\x00-\x7F]+'       # Non-ASCII characters (could be valid in some languages)
                ]
                
                pattern_matches = 0
                for pattern in suspicious_patterns:
                    matches = re.findall(pattern, text)
                    if len(matches) > len(text) / 100:  # More than 1% matches
                        pattern_matches += 1
                
                # If text has suspicious patterns, favor OCR
                if pattern_matches >= 2:
                    return True
                
                # Check for potential structured content like tables, indexes, TOC
                structured_patterns = [
                    r'\.\.\.\.*\d+',  # TOC patterns like "....123"
                    r'\d+\s*\.\s*\d+', # Numbered sections like "1.2"
                    r'^\s*\d+\.\s+\w+', # Chapter numbers
                    r'\[\d+\]',  # Reference numbers
                    r'^\s*[A-Z]\.\s+\w+', # Lettered lists
                    r'^\s*\d+\)\s+\w+'  # Numbered lists with parentheses
                ]
                
                for pattern in structured_patterns:
                    if len(re.findall(pattern, text, re.MULTILINE)) > 3:
                        structured_content_detected = True
                        # For structured content, direct extraction often loses formatting
                        # So even if we got text, we might want to try OCR
                        logger.info("Structured content detected (indexes, TOC, lists), OCR might improve extraction")
                        break
            
            for i in range(pages_to_check):
                try:
                    page = doc[i]
                    page_text = page.get_text().strip()
                    
                    # Quick check for gibberish - if first page is gibberish, likely need OCR
                    if i == 0 and is_gibberish(page_text):
                        return True
                    if is_gibberish(page_text):
                        has_encoding_issues = True
                    text_length += len(page_text)
                    
                    # Check for lack of expected characters in text docs
                    if page_text and len(page_text) > 100:
                        # Normal text docs should have common punctuation
                        if not any(c in page_text for c in '.,:;?!()-'):
                            has_encoding_issues = True
                    
                    # Check images more thoroughly
                    img_list = page.get_images(full=True)
                    if img_list:
                        # Get page dimensions
                        page_width, page_height = page.rect.width, page.rect.height
                        page_area = page_width * page_height
                        
                        # Calculate approximate pixel count for all images on the page
                        total_image_pixels = 0
                        
                        # Analyze images to detect potential text in images
                        try:
                            for img_info in img_list:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                if base_image:
                                    img_width, img_height = base_image["width"], base_image["height"]
                                    total_image_pixels += img_width * img_height
                            
                            # If images cover significant portion of page
                            if total_image_pixels > 0.2 * page_area:  # Lowered threshold to 20%
                                has_significant_images = True
                                
                                # Calculate text-to-image ratio (bytes of text per pixel of image)
                                if total_image_pixels > 0:
                                    text_to_image_ratio = len(page_text) / (total_image_pixels/1000)
                        except Exception as img_err:
                            logger.warning(f"Error analyzing image: {img_err}")
                        
                        # Check layout - pages with many small images might be diagrams/tables
                        if len(img_list) > 5:
                            structured_content_detected = True
                            
                except Exception as e:
                    logger.warning(f"Error checking page {i}: {e}")
                    has_encoding_issues = True
            
            if pages_to_check > 0:
                avg_text = text_length / pages_to_check
            else:
                avg_text = 0
                
            # Enhanced decision logic with aggressive fallback to OCR
            should_use_ocr = (
                avg_text < 100 or                        # Very little text
                has_encoding_issues or                   # Text encoding problems 
                has_significant_images or                # Has large images that might contain text
                text_to_image_ratio < 0.2 or             # Very little text relative to image size (more aggressive)
                structured_content_detected or           # Detected structured content like tables/indexes
                (len(img_list) > 0 and avg_text < 500)   # Has images and limited text
            )
            
            logger.info(f"OCR decision: {should_use_ocr} (text: {avg_text}, images: {has_significant_images}, structured: {structured_content_detected}, ratio: {text_to_image_ratio})")
            return should_use_ocr
            
        except Exception as e:
            logger.error(f"Error determining OCR need: {e}")
            return True

    def _extract_text(self, pdf_bytes: bytes) -> Tuple[str, List[Dict]]:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = []
            metadata = []
            
            # Get sample text for training the corrector - only from first 3 pages
            sample_text = ""
            for i in range(min(3, len(doc))):
                try:
                    text = doc[i].get_text()
                    if text and not is_gibberish(text):
                        sample_text += text + "\n"
                        # If we have enough sample text, stop early
                        if len(sample_text) > 1000:
                            break
                except Exception:
                    pass
                    
            # Train statistical corrector on sample
            if len(sample_text) > 100:
                self.text_corrector.train_from_document(sample_text)
            
            # For very large documents, limit initial processing
            total_pages = len(doc)
            max_pages = min(50, total_pages)  # Process max 50 pages for performance
            
            for page_num in range(max_pages):
                try:
                    # Try different extraction methods
                    dict_method = page.get_text("dict")
                    blocks = dict_method.get("blocks", [])
                    
                    # If no blocks found, try raw extraction
                    if not blocks:
                        raw_text = page.get_text("text").strip()
                        if raw_text:
                            # Apply correction if needed and trained
                            if is_gibberish(raw_text) and self.text_corrector.trained:
                                corrected = self.text_corrector.correct_text(raw_text)
                                raw_text = corrected
                            
                            full_text.append(raw_text)
                            metadata.append({
                                "text": raw_text,
                                "page": page_num + 1,
                                "block_type": "raw_text",
                                "confidence": 0.7 if is_gibberish(raw_text) else 0.9,
                                "corrected": is_gibberish(raw_text)
                            })
                        continue
                    
                    page_text = []
                    page_metadata = []
                    
                    for block in blocks:
                        if "lines" not in block:
                            continue
                            
                        block_text = ""
                        block_type = self._determine_block_type(block)
                        confidence = 1.0
                        
                        # Extract font information when available - simplified for performance
                        font_info = set()
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                # Collect font info
                                if "font" in span:
                                    font_name = span.get("font", "")
                                    if font_name:
                                        font_info.add(font_name)
                                        
                                if span.get("text", "").strip():
                                    block_text += span.get("text", "") + " "
                                    
                        block_text = block_text.strip()
                        
                        # Check if text might have encoding problems
                        needs_correction = is_gibberish(block_text)
                        if needs_correction:
                            confidence = 0.5
                            if self.text_corrector.trained:
                                block_text = self.text_corrector.correct_text(block_text)
                            
                        if block_text:
                            page_text.append(block_text)
                            page_metadata.append({
                                "text": block_text,
                                "page": page_num + 1,
                                "block_type": block_type,
                                "bbox": block.get("bbox", [0, 0, 0, 0]),
                                "confidence": confidence,
                                "fonts": list(font_info) if font_info else [],
                                "corrected": needs_correction
                            })
                    
                    if page_text:
                        full_text.append("\n".join(page_text))
                        metadata.extend(page_metadata)
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
            
            # Note if we didn't process all pages
            if total_pages > max_pages:
                note = f"\n\nNote: Only processed first {max_pages} of {total_pages} pages for performance."
                full_text.append(note)
                    
            return "\n\n".join(full_text), metadata
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "", []

    def _determine_block_type(self, block: Dict) -> str:
        block_type = "paragraph"
        try:
            if block.get("type", 0) == 1:
                return "image"
            max_font_size = 0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = span.get("size", 0)
                    max_font_size = max(max_font_size, font_size)
            if max_font_size > 14:
                block_type = "heading1"
            elif max_font_size > 12:
                block_type = "heading2"
            elif max_font_size > 10:
                block_type = "heading3"
            text = "".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
            if text.strip().startswith(("•", "-", "*", "1.", "a.", "i.", "A.")):
                block_type = "list_item"
                
            # Check for table-like structures
            if self._is_likely_table(block):
                block_type = "table_cell"
        except Exception as e:
            logger.warning(f"Error determining block type: {e}")
        return block_type
        
    def _is_likely_table(self, block: Dict) -> bool:
        """Check if a block is likely part of a table structure."""
        try:
            # Tables often have specific layout characteristics
            bbox = block.get("bbox", [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            # Table cells are often relatively small
            if height < 25:
                return True
                
            # Check for numeric content which is common in tables
            text = "".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
            text = text.strip()
            
            # If text is purely numeric or short with numbers, likely a table cell
            if text.isdigit() or (len(text) < 20 and re.search(r'\d', text)):
                return True
                
            return False
        except Exception:
            return False

    def _enhance_metadata(self, metadata: List[Dict]) -> List[Dict]:
        try:
            enhanced = []
            current_section = None
            
            for block in metadata:
                # Skip empty blocks
                if not block.get("text", "").strip():
                    continue
                    
                block_type = block.get("block_type", "")
                if block_type in ("heading1", "heading2", "heading3", "heading"):
                    current_section = block.get("text", "")
                    
                if current_section:
                    block["section"] = current_section
                    
                # Add word count and token estimation
                text = block.get("text", "")
                block["length"] = len(text)
                block["word_count"] = len(re.findall(r'\b\w+\b', text))
                
                enhanced.append(block)
                
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing metadata: {e}")
            return metadata

    def _has_encoding_issues(self, text: str) -> bool:
        if not text:
            return False
            
        try:
            # Check for unusual characters
            special_chars = set('❑✬✚✲✭✮✯✥✕▲◆✽✾❏❂❁❍▼❃❩✗✙✢✚✔✒✛✚✓✘✔✣✖✗✫✔✛✔✣✔✙✖✕✏✙✓✦▼✏✑✦✛✚✢✔♦✘✓✦✓✥✙✒✕◗❘❙❚❯❱❨❬❭❪❫❴❵❭❛❜❝❞❡❢❣❤✐✐❥❦❧♠♣qrs✉r✈✇①②③④⑤⑥⑦⑧⑨⑩❶❷❸❺❻❼❸❽')
            problem_count = sum(1 for c in text if (c in special_chars or 
                                                 (ord(c) > 127 and ord(c) < 160)))
            problem_ratio = problem_count / max(len(text), 1)
            
            # Check for weird patterns
            weird_patterns = ['✩✩', '❑✬', '✸✺✻', '✦❵❛']
            has_patterns = any(pattern in text for pattern in weird_patterns)
            
            # Check for character repetition patterns that indicate encoding issues
            unusual_repeats = re.findall(r'(.)\1{5,}', text)  # Same character repeated 5+ times
            unusual_sequences = re.findall(r'([^\w\s]){3,}', text)  # 3+ consecutive symbols
            
            # Check for control characters
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
            control_ratio = control_chars / max(len(text), 1)
            
            return (problem_ratio > 0.05 or 
                    has_patterns or 
                    len(unusual_repeats) > 0 or 
                    len(unusual_sequences) > 0 or
                    control_ratio > 0.01)
        except Exception as e:
            logger.warning(f"Error checking for encoding issues: {e}")
            return False  # Assume no issues if check fails
