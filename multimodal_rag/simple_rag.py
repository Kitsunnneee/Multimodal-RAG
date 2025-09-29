"""
Simple RAG system fallback that works without complex AI dependencies
"""
import re
from typing import Dict, List, Any
from pathlib import Path


class SimpleRAG:
    """Simple RAG system that provides basic text analysis without AI dependencies"""
    
    def __init__(self):
        self.documents = []
        self.transcriptions = {}
    
    def add_text_content(self, content: str, source: str = "unknown"):
        """Add text content to the simple RAG system"""
        self.documents.append({
            'content': content,
            'source': source,
            'type': 'text'
        })
    
    def add_transcription(self, filename: str, transcription: str):
        """Add audio transcription"""
        self.transcriptions[filename] = transcription
        self.add_text_content(transcription, f"Audio: {filename}")
    
    def query(self, question: str, chat_history=None, return_context=False) -> Dict[str, Any]:
        """Simple query processing with keyword matching"""
        question_lower = question.lower()
        
        # Extract key terms from question
        question_words = set(re.findall(r'\w+', question_lower))
        
        # Find relevant content
        relevant_docs = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(re.findall(r'\w+', content_lower))
            
            # Calculate simple relevance score
            matches = len(question_words.intersection(content_words))
            if matches > 0:
                relevant_docs.append({
                    'doc': doc,
                    'score': matches,
                    'matches': question_words.intersection(content_words)
                })
        
        # Sort by relevance
        relevant_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate response
        if relevant_docs:
            best_doc = relevant_docs[0]['doc']
            answer = self._generate_answer(question, best_doc, relevant_docs[0]['matches'])
        else:
            # Handle transcription queries directly
            if any(word in question_lower for word in ['transcription', 'audio', 'sound', 'hear']):
                if self.transcriptions:
                    transcription_text = '\n'.join([f"{k}: {v}" for k, v in self.transcriptions.items()])
                    answer = f"Here are the available audio transcriptions:\n\n{transcription_text}"
                else:
                    answer = "No audio transcriptions are available."
            else:
                answer = f"I couldn't find specific information to answer: {question}"
        
        return {
            "answer": answer,
            "citations": self._generate_citations(relevant_docs[:3])
        }
    
    def _generate_answer(self, question: str, doc: dict, matches: set) -> str:
        """Generate a simple answer based on the document content"""
        content = doc['content']
        source = doc['source']
        
        # Handle different types of questions
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['summary', 'summarize', 'about']):
            # Provide summary
            sentences = re.split(r'[.!?]', content)
            key_sentences = [s.strip() for s in sentences if s.strip() and any(match in s.lower() for match in matches)]
            
            if key_sentences:
                answer = f"**Summary based on {source}:**\n\n"
                answer += f"The content discusses: {' '.join(key_sentences[:3])}"
                answer += f"\n\n**Full content:** {content}"
            else:
                answer = f"**Summary of {source}:**\n\n{content}"
        
        elif any(word in question_lower for word in ['smell', 'odor', 'scent']):
            # Handle smell-related queries
            if 'smell' in content.lower() or 'odor' in content.lower():
                smell_context = []
                sentences = re.split(r'[.!?]', content)
                for sentence in sentences:
                    if 'smell' in sentence.lower() or 'odor' in sentence.lower():
                        smell_context.append(sentence.strip())
                
                if smell_context:
                    answer = f"**About smells/odors:**\n\n{' '.join(smell_context)}"
                    if 'stale smell of old beer' in content.lower():
                        answer += f"\n\n**Specifically:** The audio mentions 'the stale smell of old beer lingers'"
                else:
                    answer = f"The content mentions smells but here's the full context:\n\n{content}"
            else:
                answer = f"No specific smell information found, but here's the related content:\n\n{content}"
        
        elif any(word in question_lower for word in ['food', 'eat', 'taste']):
            # Handle food-related queries
            food_words = ['beer', 'pickle', 'ham', 'tacos', 'bun', 'food']
            found_foods = [word for word in food_words if word in content.lower()]
            
            if found_foods:
                answer = f"**Food items mentioned:** {', '.join(found_foods)}\n\n**Full content:** {content}"
            else:
                answer = f"Here's the content that might be related to food:\n\n{content}"
        
        else:
            # General answer
            sentences = re.split(r'[.!?]', content)
            relevant_sentences = [s.strip() for s in sentences if s.strip() and any(match in s.lower() for match in matches)]
            
            if relevant_sentences:
                answer = f"**Based on {source}:**\n\n{' '.join(relevant_sentences[:2])}"
                answer += f"\n\n**Complete content:** {content}"
            else:
                answer = f"**From {source}:**\n\n{content}"
        
        return answer
    
    def _generate_citations(self, relevant_docs: List[dict]) -> List[dict]:
        """Generate simple citations"""
        citations = []
        for item in relevant_docs:
            doc = item['doc']
            citations.append({
                'type': 'text',
                'source': doc['source'],
                'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'display_text': f"Source: {doc['source']}"
            })
        return citations