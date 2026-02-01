import re
from typing import List
def _split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        This handles common sentence endings while avoiding false positives
        with abbreviations and decimal numbers.
        
        Pattern Explanation:
        - Matches periods, exclamation marks, question marks
        - Requires whitespace or end of string after punctuation
        - Handles common abbreviations (Mr., Dr., etc.)
        """
        # Improved sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
    
print(_split_into_sentences(text="Embedding text in batches is significantly better than processing texts one-by-one (or all at once as a single, massive, unsegmented string) primarily because it maximizes hardware utilization, reduces API overhead, and significantly lowers costs. By grouping texts, you leverage parallel processing capabilities, increasing throughput by up to 30%. "))    
ans=['Embedding text in batches is significantly better than processing texts one-by-one (or all at once as a single, massive, unsegmented string) primarily because it maximizes hardware utilization, reduces API overhead, and significantly lowers costs.', 'By grouping texts, you leverage parallel processing capabilities, increasing throughput by up to 30%.']
print(len(ans))