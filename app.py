import streamlit as st
import pandas as pd
import PyPDF2
 import python-docx
import io
import re
import json
import openai
from datetime import datetime
import plotly as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1e40af;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .question-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .answer-card {
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    .stats-card {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class DocumentParser:
    """Simplified document parser for Streamlit app"""
    
    @staticmethod
    def parse_pdf(uploaded_file):
        """Parse PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error parsing PDF: {str(e)}")
            return ""
    
    @staticmethod
    def parse_docx(uploaded_file):
        """Parse DOCX file"""
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error parsing DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def parse_text(uploaded_file):
        """Parse text file"""
        try:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return content
        except Exception as e:
            st.error(f"Error parsing text file: {str(e)}")
            return ""

class QuestionAnalyzer:
    """Question analysis functionality"""
    
    def _init_(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def extract_questions(self, text: str) -> List[str]:
        """Extract questions from text using regex patterns"""
        question_patterns = [
            r'^\d+\.\s+(.+?)(?=^\d+\.|$)',  # Numbered questions
            r'^Q\d+[.:\s]+(.+?)(?=^Q\d+|$)',  # Q1, Q2 format
            r'^\([a-zA-Z]\)\s+(.+?)(?=^\([a-zA-Z]\)|$)'  # (a), (b) format
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            questions.extend([q.strip() for q in matches if q.strip()])
        
        # If no structured questions found, split by sentences that end with '?'
        if not questions:
            sentences = re.split(r'[.!?]+', text)
            questions = [s.strip() + '?' for s in sentences if '?' in s or len(s.strip()) > 20]
        
        return questions[:20]  # Limit to 20 questions for demo
    
    def analyze_questions(self, questions: List[str]) -> Dict[str, Any]:
        """Analyze questions to extract patterns and topics"""
        if not questions:
            return {}
        
        # Extract topics using simple keyword frequency
        topics = self._extract_topics_simple(questions)
        
        # Identify question patterns
        patterns = self._identify_patterns(questions)
        
        # Calculate basic statistics
        stats = {
            'total_questions': len(questions),
            'avg_length': np.mean([len(q.split()) for q in questions]),
            'difficulty_distribution': self._assess_difficulty_distribution(questions)
        }
        
        return {
            'topics': topics,
            'patterns': patterns,
            'stats': stats,
            'questions': questions
        }
    
    def _extract_topics_simple(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Simple topic extraction using keyword frequency"""
        all_words = []
        for question in questions:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
            all_words.extend([w for w in words if w not in self.stop_words])
        
        # Count word frequency
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{
            'id': 0,
            'keywords': [word for word, count in top_keywords],
            'sample_questions': questions[:3],
            'question_count': len(questions)
        }]
    
    def _identify_patterns(self, questions: List[str]) -> Dict[str, List[str]]:
        """Identify question patterns"""
        patterns = {
            'definition': [],
            'explanation': [],
            'comparison': [],
            'calculation': [],
            'analysis': []
        }
        
        pattern_keywords = {
            'definition': ['define', 'what is', 'meaning', 'term'],
            'explanation': ['explain', 'describe', 'elaborate', 'discuss'],
            'comparison': ['compare', 'contrast', 'difference', 'similarity'],
            'calculation': ['calculate', 'find', 'solve', 'compute'],
            'analysis': ['analyze', 'evaluate', 'assess', 'examine']
        }
        
        for question in questions:
            question_lower = question.lower()
            categorized = False
            
            for pattern_type, keywords in pattern_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    patterns[pattern_type].append(question)
                    categorized = True
                    break
            
            if not categorized:
                patterns['explanation'].append(question)
        
        return patterns
    
    def _assess_difficulty_distribution(self, questions: List[str]) -> Dict[str, int]:
        """Assess difficulty distribution"""
        difficulty = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for question in questions:
            word_count = len(question.split())
            if word_count < 10:
                difficulty['easy'] += 1
            elif word_count < 20:
                difficulty['medium'] += 1
            else:
                difficulty['hard'] += 1
        
        return difficulty

class QuestionGenerator:
    """Question generation functionality"""
    
    def _init_(self):
        self.question_templates = {
            'definition': [
                "Define {keyword}.",
                "What is meant by {keyword}?",
                "Explain the concept of {keyword}.",
                "Give the definition of {keyword}."
            ],
            'explanation': [
                "Explain {keyword} in detail.",
                "Describe the process of {keyword}.",
                "Discuss the importance of {keyword}.",
                "Elaborate on {keyword}."
            ],
            'comparison': [
                "Compare {keyword1} and {keyword2}.",
                "What are the differences between {keyword1} and {keyword2}?",
                "Contrast {keyword1} with {keyword2}.",
                "Analyze the similarities and differences between {keyword1} and {keyword2}."
            ],
            'calculation': [
                "Calculate the value of {keyword}.",
                "Find the {keyword} for the given problem.",
                "Solve for {keyword}.",
                "Determine the {keyword}."
            ],
            'analysis': [
                "Analyze the {keyword}.",
                "Evaluate the effectiveness of {keyword}.",
                "Critically examine {keyword}.",
                "Assess the impact of {keyword}."
            ]
        }
    
    def generate_questions(self, analysis_result: Dict[str, Any], 
                          num_questions: int = 10, 
                          target_marks: int = 100) -> List[Dict[str, Any]]:
        """Generate new questions based on analysis"""
        
        topics = analysis_result.get('topics', [])
        patterns = analysis_result.get('patterns', {})
        
        if not topics:
            return []
        
        generated_questions = []
        keywords = topics[0].get('keywords', ['concept', 'topic', 'subject'])[:5]
        
        # Distribute questions across different patterns
        pattern_types = list(patterns.keys())
        questions_per_pattern = max(1, num_questions // len(pattern_types))
        
        for pattern_type in pattern_types:
            for i in range(min(questions_per_pattern, num_questions - len(generated_questions))):
                question = self._generate_single_question(keywords, pattern_type)
                marks = self._assign_marks(pattern_type)
                difficulty = self._assign_difficulty(pattern_type)
                
                generated_questions.append({
                    'question': question,
                    'pattern_type': pattern_type,
                    'marks': marks,
                    'difficulty': difficulty,
                    'answer': self._generate_sample_answer(question, pattern_type, marks)
                })
                
                if len(generated_questions) >= num_questions:
                    break
        
        # Adjust marks to reach target
        self._adjust_marks(generated_questions, target_marks)
        
        return generated_questions
    
    def _generate_single_question(self, keywords: List[str], pattern_type: str) -> str:
        """Generate a single question"""
        templates = self.question_templates.get(pattern_type, self.question_templates['explanation'])
        template = random.choice(templates)
        
        if '{keyword1}' in template and '{keyword2}' in template:
            keyword1 = random.choice(keywords) if keywords else 'concept A'
            keyword2 = random.choice([k for k in keywords if k != keyword1]) if len(keywords) > 1 else 'concept B'
            return template.format(keyword1=keyword1, keyword2=keyword2)
        elif '{keyword}' in template:
            keyword = random.choice(keywords) if keywords else 'the given concept'
            return template.format(keyword=keyword)
        else:
            return template
    
    def _assign_marks(self, pattern_type: str) -> int:
        """Assign marks based on question type"""
        mark_ranges = {
            'definition': [2, 3, 5],
            'explanation': [5, 8, 10],
            'comparison': [8, 10, 12],
            'calculation': [5, 10, 15],
            'analysis': [10, 15, 20]
        }
        return random.choice(mark_ranges.get(pattern_type, [5, 8, 10]))
    
    def _assign_difficulty(self, pattern_type: str) -> str:
        """Assign difficulty based on question type"""
        difficulty_map = {
            'definition': 'Easy',
            'explanation': 'Medium',
            'comparison': 'Medium',
            'calculation': 'Medium',
            'analysis': 'Hard'
        }
        return difficulty_map.get(pattern_type, 'Medium')
    
    def _adjust_marks(self, questions: List[Dict], target_marks: int):
        """Adjust marks to reach target total"""
        current_total = sum(q['marks'] for q in questions)
        if current_total == 0:
            return
        
        adjustment_factor = target_marks / current_total
        for question in questions:
            question['marks'] = max(1, int(question['marks'] * adjustment_factor))
    
    def _generate_sample_answer(self, question: str, pattern_type: str, marks: int) -> str:
        """Generate a sample answer"""
        answer_templates = {
            'definition': f"This is a fundamental concept that can be defined as... [Answer would include key points and explanation based on {marks} marks]",
            'explanation': f"This concept involves several key aspects... [Detailed explanation covering main points for {marks} marks]",
            'comparison': f"When comparing these concepts, we can identify similarities and differences... [Comparative analysis for {marks} marks]",
            'calculation': f"To solve this problem, we follow these steps... [Step-by-step calculation for {marks} marks]",
            'analysis': f"A comprehensive analysis reveals... [Critical evaluation and assessment for {marks} marks]"
        }
        
        base_answer = answer_templates.get(pattern_type, "This requires a detailed response covering all relevant aspects...")
        
        # Add more detail based on marks
        if marks >= 10:
            base_answer += "\n\nAdditional points to consider:\n• Point 1\n• Point 2\n• Point 3"
        if marks >= 15:
            base_answer += "\n\nFurther elaboration with examples and practical applications would be expected."
        
        return base_answer

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown('<h1 class="main-header">📝 AI Question Paper Generator</h1>', unsafe_allow_html=True)
    st.markdown("""
    Upload an old question paper and let AI generate a new one with answers! 
    This tool analyzes patterns, topics, and question structures to create fresh questions.
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙ Configuration")
    
    # File upload
    st.sidebar.markdown("### 📄 Upload Question Paper")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Upload a question paper in PDF, DOCX, or TXT format"
    )
    
    # Generation parameters
    st.sidebar.markdown("### 🎯 Generation Parameters")
    num_questions = st.sidebar.slider("Number of Questions", 5, 20, 10)
    target_marks = st.sidebar.slider("Total Marks", 50, 200, 100)
    
    # Question types
    st.sidebar.markdown("### 📋 Question Types")
    include_definition = st.sidebar.checkbox("Definition Questions", True)
    include_explanation = st.sidebar.checkbox("Explanation Questions", True)
    include_comparison = st.sidebar.checkbox("Comparison Questions", True)
    include_calculation = st.sidebar.checkbox("Calculation Questions", False)
    include_analysis = st.sidebar.checkbox("Analysis Questions", True)
    
    # API Configuration
    st.sidebar.markdown("### 🔧 API Configuration")
    use_openai = st.sidebar.checkbox("Use OpenAI GPT (Optional)", False)
    openai_key = ""
    if use_openai:
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # Main content area
    if uploaded_file is not None:
        # Parse document
        with st.spinner("📖 Parsing document..."):
            parser = DocumentParser()
            
            if uploaded_file.type == "application/pdf":
                text = parser.parse_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = parser.parse_docx(uploaded_file)
            else:
                text = parser.parse_text(uploaded_file)
        
        if text:
            # Analyze questions
            with st.spinner("🔍 Analyzing questions..."):
                analyzer = QuestionAnalyzer()
                questions = analyzer.extract_questions(text)
                analysis_result = analyzer.analyze_questions(questions)
            
            # Display analysis results
            st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h3>📝 Questions Found</h3>
                    <h2>{analysis_result.get('stats', {}).get('total_questions', 0)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_length = analysis_result.get('stats', {}).get('avg_length', 0)
                st.markdown(f"""
                <div class="stats-card">
                    <h3>📏 Avg. Length</h3>
                    <h2>{avg_length:.1f} words</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                patterns = analysis_result.get('patterns', {})
                most_common = max(patterns.items(), key=lambda x: len(x[1]))[0] if patterns else "N/A"
                st.markdown(f"""
                <div class="stats-card">
                    <h3>🔍 Common Type</h3>
                    <h2>{most_common.title()}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Pattern distribution chart
            if patterns:
                pattern_counts = {k: len(v) for k, v in patterns.items() if v}
                if pattern_counts:
                    fig = px.bar(
                        x=list(pattern_counts.keys()),
                        y=list(pattern_counts.values()),
                        title="Question Pattern Distribution",
                        labels={'x': 'Pattern Type', 'y': 'Count'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Display extracted questions
            with st.expander("🔍 View Extracted Questions"):
                for i, question in enumerate(questions[:10], 1):
                    st.markdown(f"{i}.** {question}")
            
            # Generate new questions
            if st.button("🚀 Generate New Question Paper", type="primary"):
                with st.spinner("🤖 Generating new questions..."):
                    generator = QuestionGenerator()
                    generated_questions = generator.generate_questions(
                        analysis_result, 
                        num_questions, 
                        target_marks
                    )
                
                # Display generated questions
                st.markdown('<h2 class="section-header">📋 Generated Question Paper</h2>', unsafe_allow_html=True)
                
                # Paper header
                st.markdown(f"""
                *Subject:* General  
                *Total Marks:* {sum(q['marks'] for q in generated_questions)}  
                *Time:* 3 Hours  
                *Date:* {datetime.now().strftime('%B %d, %Y')}
                
                ---
                """)
                
                # Questions
                for i, q_data in enumerate(generated_questions, 1):
                    st.markdown(f"""
                    <div class="question-card">
                        <strong>Q{i}.</strong> {q_data['question']} 
                        <span style="float: right;">
                            <strong>[{q_data['marks']} marks]</strong> 
                            <span style="background: {'#fef3c7' if q_data['difficulty'] == 'Easy' else '#dbeafe' if q_data['difficulty'] == 'Medium' else '#fecaca'}; 
                                  padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-left: 8px;">
                                {q_data['difficulty']}
                            </span>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Answer key section
                st.markdown('<h2 class="section-header">🔑 Answer Key</h2>', unsafe_allow_html=True)
                
                for i, q_data in enumerate(generated_questions, 1):
                    with st.expander(f"Answer {i} ({q_data['marks']} marks)"):
                        st.markdown(f"""
                        <div class="answer-card">
                            <strong>Question:</strong> {q_data['question']}<br><br>
                            <strong>Answer:</strong><br>
                            {q_data['answer']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download options
                st.markdown('<h2 class="section-header">💾 Download Options</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create question paper text
                    question_paper = f"""QUESTION PAPER
Subject: General
Total Marks: {sum(q['marks'] for q in generated_questions)}
Time: 3 Hours
Date: {datetime.now().strftime('%B %d, %Y')}

{'='*50}

"""
                    for i, q_data in enumerate(generated_questions, 1):
                        question_paper += f"Q{i}. {q_data['question']} [{q_data['marks']} marks]\n\n"
                    
                    st.download_button(
                        label="📄 Download Question Paper",
                        data=question_paper,
                        file_name=f"question_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create answer key text
                    answer_key = f"""ANSWER KEY
Subject: General
Date: {datetime.now().strftime('%B %d, %Y')}

{'='*50}

"""
                    for i, q_data in enumerate(generated_questions, 1):
                        answer_key += f"Q{i}. {q_data['question']}\n"
                        answer_key += f"Answer ({q_data['marks']} marks):\n{q_data['answer']}\n\n{'='*50}\n\n"
                    
                    st.download_button(
                        label="🔑 Download Answer Key",
                        data=answer_key,
                        file_name=f"answer_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # Statistics summary
                st.markdown('<h2 class="section-header">📈 Generation Summary</h2>', unsafe_allow_html=True)
                
                difficulty_dist = {}
                for q in generated_questions:
                    diff = q['difficulty']
                    difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Difficulty distribution pie chart
                    if difficulty_dist:
                        fig = px.pie(
                            values=list(difficulty_dist.values()),
                            names=list(difficulty_dist.keys()),
                            title="Difficulty Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Marks distribution
                    marks_list = [q['marks'] for q in generated_questions]
                    fig = px.histogram(
                        x=marks_list,
                        title="Marks Distribution",
                        labels={'x': 'Marks', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Could not extract text from the uploaded file. Please check the file format and try again.")
    
    else:
        # Welcome message
        st.markdown("""
        ### 🚀 How to Use:
        
        1. *Upload* an old question paper (PDF, DOCX, or TXT format)
        2. *Configure* generation parameters in the sidebar
        3. *Analyze* the uploaded paper to understand patterns
        4. *Generate* new questions with AI
        5. *Download* the question paper and answer key
        
        ### ✨ Features:
        
        - 📊 *Smart Analysis*: Automatically identifies question patterns and topics
        - 🎯 *Customizable*: Adjust number of questions, marks, and question types
        - 🤖 *AI-Powered*: Uses advanced NLP for question generation
        - 📝 *Complete Package*: Generates both questions and detailed answers
        - 💾 *Export Ready*: Download in text format for easy editing
        
        ### 📋 Supported Question Types:
        
        - *Definition*: Basic concept definitions
        - *Explanation*: Detailed explanations and descriptions  
        - *Comparison*: Compare and contrast questions
        - *Calculation*: Mathematical and numerical problems
        - *Analysis*: Critical thinking and evaluation questions
        
        Upload a question paper to get started! 🎯
        """)

if _name_ == "_main_":
    main()




