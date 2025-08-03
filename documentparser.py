# models/document_parser.py
import PyPDF2
import docx
import re
from typing import List, Dict, Any

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document and extract structured information"""
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self._parse_pdf(file_path)
        elif file_extension == 'txt':
            return self._parse_text(file_path)
        elif file_extension == 'docx':
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        return self._structure_content(text)
    
    def _parse_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return self._structure_content(text)
    
    def _parse_docx(self, file_path: str) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return self._structure_content(text)
    
    def _structure_content(self, text: str) -> Dict[str, Any]:
        """Structure the extracted text into questions and metadata"""
        # Extract questions using regex patterns
        question_patterns = [
            r'^\d+\.\s+(.+?)(?=^\d+\.|$)',  # Numbered questions
            r'^Q\d+[.:\s]+(.+?)(?=^Q\d+|$)',  # Q1, Q2 format
            r'^\([a-zA-Z]\)\s+(.+?)(?=^\([a-zA-Z]\)|$)'  # (a), (b) format
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            questions.extend(matches)
        
        # Extract metadata
        metadata = {
            'total_questions': len(questions),
            'estimated_marks': self._estimate_marks(text),
            'subject': self._extract_subject(text),
            'difficulty_level': self._assess_difficulty(questions),
            'question_types': self._classify_question_types(questions)
        }
        
        return {
            'questions': questions,
            'metadata': metadata,
            'raw_text': text
        }
    
    def _estimate_marks(self, text: str) -> int:
        """Estimate total marks from text"""
        mark_patterns = [
            r'(\d+)\s*marks?',
            r'\[(\d+)\]',
            r'Total:?\s*(\d+)'
        ]
        
        marks = []
        for pattern in mark_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            marks.extend([int(m) for m in matches])
        
        return max(marks) if marks else 100  # Default to 100 marks
    
    def _extract_subject(self, text: str) -> str:
        """Extract subject from text"""
        subjects = [
            'mathematics', 'physics', 'chemistry', 'biology',
            'computer science', 'english', 'history', 'geography'
        ]
        
        text_lower = text.lower()
        for subject in subjects:
            if subject in text_lower:
                return subject.title()
        
        return "General"
    
    def _assess_difficulty(self, questions: List[str]) -> str:
        """Assess difficulty level of questions"""
        difficulty_keywords = {
            'easy': ['define', 'what is', 'list', 'name'],
            'medium': ['explain', 'describe', 'compare', 'analyze'],
            'hard': ['evaluate', 'synthesize', 'design', 'prove']
        }
        
        scores = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for question in questions:
            question_lower = question.lower()
            for level, keywords in difficulty_keywords.items():
                for keyword in keywords:
                    if keyword in question_lower:
                        scores[level] += 1
                        break
        
        return max(scores, key=scores.get)
    
    def _classify_question_types(self, questions: List[str]) -> Dict[str, int]:
        """Classify questions by type"""
        types = {
            'mcq': 0,
            'short_answer': 0,
            'long_answer': 0,
            'numerical': 0
        }
        
        for question in questions:
            if re.search(r'\([a-d]\)', question, re.IGNORECASE):
                types['mcq'] += 1
            elif len(question.split()) > 50:
                types['long_answer'] += 1
            elif re.search(r'\d+', question):
                types['numerical'] += 1
            else:
                types['short_answer'] += 1
        
        return types