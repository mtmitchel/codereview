import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable, Tuple, Any
from utils.api_client import APIClient

class ReviewStrategy(ABC):
    """
    Base abstract class for different code review strategies.
    """
    
    def __init__(self, api_client: APIClient, status_callback: Optional[Callable] = None):
        """
        Initialize the review strategy.
        
        Args:
            api_client: Client for API calls to language models
            status_callback: Optional callback function for status updates
        """
        self.api_client = api_client
        self.status_callback = status_callback
        
    def update_status(self, message: str) -> None:
        """
        Update status using the callback if available.
        
        Args:
            message: Status message to report
        """
        if self.status_callback:
            self.status_callback(message)
            
    @abstractmethod
    def review_file(self, file_path: str, file_content: str) -> Dict[str, Any]:
        """
        Review a single file and return the results.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            
        Returns:
            Dictionary containing review results
        """
        pass
        
    @abstractmethod
    def review_chunk(self, chunk: Dict[str, str]) -> Dict[str, Any]:
        """
        Review a chunk of a file and return the results.
        
        Args:
            chunk: Dictionary containing chunk content and metadata
            
        Returns:
            Dictionary containing review results
        """
        pass
        
    @abstractmethod
    def consolidate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate multiple review results into a final review.
        
        Args:
            results: List of review results
            
        Returns:
            Dictionary containing consolidated review results
        """
        pass


class ChunkBasedReview(ReviewStrategy):
    """
    Strategy that reviews files by breaking them into chunks.
    """
    
    def __init__(self, 
                api_client: APIClient, 
                chunk_size: int = 4000, 
                status_callback: Optional[Callable] = None,
                review_prompt: str = None):
        """
        Initialize chunk-based review strategy.
        
        Args:
            api_client: Client for API calls to language models
            chunk_size: Maximum size for each chunk in characters
            status_callback: Optional callback function for status updates
            review_prompt: Custom prompt for the review
        """
        super().__init__(api_client, status_callback)
        self.chunk_size = chunk_size
        self.review_prompt = review_prompt or self._get_default_prompt()
        
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt for code review.
        
        Returns:
            String containing the default prompt
        """
        return """
        You are a senior software engineer conducting a code review. 
        Please review the following code chunk with a focus on providing **concise and actionable feedback**:
        
        1. Code quality and readability
        2. Potential bugs and errors (indicate likelihood if possible)
        3. Performance issues (mention potential impact)
        4. Security vulnerabilities (indicate severity)
        5. Best practices and design patterns adherence
        
        Provide specific feedback with line numbers where applicable. Focus on the most important issues.
        
        CODE CHUNK:
        ```
        {code}
        ```
        
        File Path: {file_path}
        Lines in Chunk: {start_line} to {end_line}
        
        Respond ONLY in markdown format with clear sections for:
        - Issues: List specific issues found. **For each issue, assign a severity rating (Critical, High, Medium, Low, Info).**
        - Suggestions: Briefly suggest improvements for the identified issues.
        - Positive Aspects: Briefly mention any notable positive aspects (optional, be concise).
        """
        
    def review_file(self, file_path: str, file_content: str) -> Dict[str, Any]:
        """
        Review a complete file by breaking it into chunks.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            
        Returns:
            Dictionary containing review results
        """
        # Calculate lines in the file
        lines = file_content.split('\n')
        total_lines = len(lines)
        
        self.update_status(f"Reviewing {file_path} ({total_lines} lines)")
        
        # Break file into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.chunk_size and current_chunk:
                # Store current chunk
                chunks.append({
                    'content': '\n'.join(current_chunk),
                    'start_line': start_line,
                    'end_line': i,
                    'file_path': file_path
                })
                
                # Start a new chunk with overlap for context
                overlap_lines = min(50, len(current_chunk))
                current_chunk = current_chunk[-overlap_lines:]
                current_size = sum(len(l) + 1 for l in current_chunk)
                start_line = max(1, i + 1 - overlap_lines)
                
            current_chunk.append(line)
            current_size += line_size
            
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'start_line': start_line,
                'end_line': total_lines,
                'file_path': file_path
            })
            
        # Review each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            self.update_status(f"Reviewing chunk {i+1}/{len(chunks)} of {file_path}")
            chunk_result = self.review_chunk(chunk)
            chunk_results.append(chunk_result)
            
        # Consolidate results
        consolidated_result = self.consolidate_results(chunk_results)
        consolidated_result['file_path'] = file_path
        consolidated_result['total_lines'] = total_lines
        
        return consolidated_result
        
    def review_chunk(self, chunk: Dict[str, str]) -> Dict[str, Any]:
        """
        Review a chunk of a file.
        
        Args:
            chunk: Dictionary containing chunk information
            
        Returns:
            Dictionary containing chunk review results
        """
        prompt = self.review_prompt.format(
            code=chunk['content'],
            file_path=chunk['file_path'],
            start_line=chunk['start_line'],
            end_line=chunk['end_line']
        )
        
        # Get review from LLM
        response = self.api_client.get_completion(prompt)
        
        return {
            'chunk_review': response,
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'file_path': chunk['file_path']
        }
        
    def consolidate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate chunk review results into a complete file review.
        
        Args:
            results: List of chunk review results, each containing 'chunk_review', 'start_line', 'end_line', 'file_path'.
            
        Returns:
            Dictionary containing consolidated review
        """
        if not results:
            return {'consolidated_review': '', 'issues': [], 'file_path': 'unknown', 'total_lines': 0}
            
        # Extract file path and total lines from the first result if possible
        # (Assume they are consistent across results for the same file)
        file_path = results[0].get('file_path', 'unknown')
        total_lines = results[0].get('total_lines', 0) # Note: total_lines might not be in chunk results, needs to be passed correctly
        
        # Create a prompt for consolidation
        consolidation_prompt = f"""
        You are a senior software engineer consolidating code review feedback for the file: `{file_path}`.
        
        Below are code review feedback snippets generated for different chunks of this file. 
        Your task is to synthesize these snippets into a single, comprehensive, and actionable review for the **entire file**.
        
        **Instructions:**
        1.  **Synthesize and Deduplicate:** Combine related points from different chunks. Remove redundant feedback.
        2.  **Prioritize:** Focus on the most impactful issues identified in the snippets. Use the severity ratings provided in the chunk reviews to guide prioritization.
        3.  **Structure:** Format the final review clearly in Markdown.
        4.  **Conciseness:** Be clear and to the point. Avoid verbose explanations unless necessary for clarity.

        **Input Chunk Reviews:**
        """
        
        # Add each chunk review, clearly demarcated
        for i, result in enumerate(results):
            consolidation_prompt += f"\n\n--- REVIEW FOR CHUNK {i+1} (Lines ~{result.get('start_line', '?')} - {result.get('end_line', '?')}) ---\n"
            # Ensure we only include the actual review text
            chunk_review_text = result.get('chunk_review', '[No review content found for this chunk]')
            consolidation_prompt += chunk_review_text
            
        consolidation_prompt += f"""
        
        --- END OF CHUNK REVIEWS ---

        **Required Output Format (Markdown):**

        ## Executive Summary 
        (Provide a brief, high-level overview (2-4 sentences) of the code quality, key findings, and overall assessment for `{file_path}`.)

        ## Issues & Suggestions
        (List the consolidated and prioritized issues. For each issue, include:
        - Description: Clear explanation of the issue.
        - Severity: (Critical, High, Medium, Low, Info) - inferred from chunk reviews.
        - Location: Approximate line number(s) or code context if possible.
        - Suggestion: Actionable recommendation for fixing the issue.)
        
        *(Example Issue Format):*
        *   **[Severity]** Issue description (Lines X-Y). *Suggestion: How to fix it.*

        ## Positive Aspects
        (Briefly mention any notable positive aspects or well-implemented patterns found across the file.)

        ## Overall Assessment
        (A final concluding sentence or two.)

        --- 
        **IMPORTANT:** Adhere strictly to the requested Markdown structure. Consolidate effectively and prioritize based on the severity ratings mentioned in the input chunk reviews.
        """
        
        # Get consolidated review from LLM
        self.update_status(f"Consolidating feedback for {file_path}...")
        consolidated_review = self.api_client.get_completion(consolidation_prompt)
        
        # We return the raw consolidated review; parsing specific sections can be done later if needed.
        return {
            'consolidated_review': consolidated_review,
            'file_path': file_path, 
            'total_lines': total_lines # Pass along file info
            # 'chunk_reviews': [r.get('chunk_review', '') for r in results] # Optionally include raw chunk reviews
        }


class WholeFileReview(ReviewStrategy):
    """
    Strategy that reviews entire files at once, suitable for smaller files.
    """
    
    def __init__(self, 
                api_client: APIClient, 
                max_file_size: int = 8000, 
                status_callback: Optional[Callable] = None,
                review_prompt: str = None):
        """
        Initialize whole file review strategy.
        
        Args:
            api_client: Client for API calls to language models
            max_file_size: Maximum file size in characters
            status_callback: Optional callback function for status updates
            review_prompt: Custom prompt for the review
        """
        super().__init__(api_client, status_callback)
        self.max_file_size = max_file_size
        self.review_prompt = review_prompt or self._get_default_prompt()
        
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt for code review.
        
        Returns:
            String containing the default prompt
        """
        return """
        You are a senior software engineer conducting a code review. 
        Please review the following code with a focus on:
        
        1. Code quality and readability
        2. Potential bugs and errors
        3. Performance issues
        4. Security vulnerabilities
        5. Best practices and design patterns
        
        Provide specific feedback with line numbers.
        
        CODE:
        {code}
        
        File: {file_path}
        
        Respond in markdown format with sections for:
        - Summary
        - Issues (with severity ratings: Critical, High, Medium, Low)
        - Suggestions for improvement
        - Positive aspects
        """
        
    def review_file(self, file_path: str, file_content: str) -> Dict[str, Any]:
        """
        Review a complete file.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            
        Returns:
            Dictionary containing review results
        """
        # Check if file is too large
        if len(file_content) > self.max_file_size:
            self.update_status(f"File {file_path} is too large for whole-file review. Using chunk-based review.")
            # Fall back to chunk-based review
            chunk_review = ChunkBasedReview(
                self.api_client, 
                chunk_size=self.max_file_size // 2,
                status_callback=self.status_callback,
                review_prompt=self.review_prompt
            )
            return chunk_review.review_file(file_path, file_content)
            
        # Calculate lines in the file
        total_lines = file_content.count('\n') + 1
        
        self.update_status(f"Reviewing {file_path} ({total_lines} lines)")
        
        # Create a chunk with the whole file
        chunk = {
            'content': file_content,
            'start_line': 1,
            'end_line': total_lines,
            'file_path': file_path
        }
        
        # Review the chunk
        result = self.review_chunk(chunk)
        result['file_path'] = file_path
        result['total_lines'] = total_lines
        
        return result
        
    def review_chunk(self, chunk: Dict[str, str]) -> Dict[str, Any]:
        """
        Review a chunk (in this case, a whole file).
        
        Args:
            chunk: Dictionary containing chunk information
            
        Returns:
            Dictionary containing review results
        """
        prompt = self.review_prompt.format(
            code=chunk['content'],
            file_path=chunk['file_path']
        )
        
        # Get review from LLM
        response = self.api_client.get_completion(prompt)
        
        return {
            'consolidated_review': response,
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'file_path': chunk['file_path']
        }
        
    def consolidate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate results (not needed for whole file review).
        
        Args:
            results: List of review results
            
        Returns:
            Dictionary containing first review result
        """
        # For whole file review, there should only be one result
        if not results:
            return {'consolidated_review': '', 'issues': []}
        
        # Just return the first result as is
        return results[0]


class CrossFileReview(ReviewStrategy):
    """
    Strategy that reviews multiple files in relation to each other.
    """
    
    def __init__(self, 
                api_client: APIClient, 
                status_callback: Optional[Callable] = None,
                review_prompt: str = None):
        """
        Initialize cross-file review strategy.
        
        Args:
            api_client: Client for API calls to language models
            status_callback: Optional callback function for status updates
            review_prompt: Custom prompt for the review
        """
        super().__init__(api_client, status_callback)
        self.review_prompt = review_prompt or self._get_default_prompt()
        
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt for cross-file review.
        
        Returns:
            String containing the default prompt
        """
        return """
        You are a senior software engineer conducting a cross-file code review. 
        Please analyze how these files interact with each other with a focus on:
        
        1. Interface consistency
        2. Dependency management
        3. Potential integration issues
        4. Architectural concerns
        5. Duplication across files
        
        FILES:
        {file_summaries}
        
        Respond in markdown format with sections for:
        - Cross-file issues
        - Architecture assessment
        - Recommendations for improving cross-file interactions
        """
        
    def review_file(self, file_path: str, file_content: str) -> Dict[str, Any]:
        """
        This is not used directly in cross-file review.
        We'll implement a minimal version that just returns file content.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            
        Returns:
            Dictionary containing file content
        """
        return {
            'file_path': file_path,
            'content': file_content,
            'lines': file_content.count('\n') + 1
        }
        
    def review_chunk(self, chunk: Dict[str, str]) -> Dict[str, Any]:
        """
        This is not used directly in cross-file review.
        
        Args:
            chunk: Dictionary containing chunk information
            
        Returns:
            Empty dictionary
        """
        return {}
        
    def consolidate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform cross-file review using the consolidated results of individual files.
        
        Args:
            results: List of individual file review results
            
        Returns:
            Dictionary containing cross-file review
        """
        if not results:
            return {'cross_file_review': ''}
            
        # Create summaries of each file
        file_summaries = ""
        for result in results:
            file_path = result['file_path']
            file_summaries += f"\n--- {file_path} ---\n"
            
            # If there is a consolidated review, extract its summary section
            if 'consolidated_review' in result and result['consolidated_review']:
                # Try to extract just the summary section to keep prompt size reasonable
                review = result['consolidated_review']
                summary_section = self._extract_summary_section(review)
                file_summaries += summary_section
            else:
                # If no review, include some basic info
                file_summaries += f"Lines: {result.get('total_lines', 'unknown')}\n"
                
        # Create a prompt for cross-file review
        prompt = self.review_prompt.format(
            file_summaries=file_summaries
        )
        
        # Get cross-file review from LLM
        cross_file_review = self.api_client.get_completion(prompt)
        
        return {
            'cross_file_review': cross_file_review
        }
        
    def _extract_summary_section(self, review: str) -> str:
        """
        Extract just the summary section from a review to keep prompt size manageable.
        
        Args:
            review: Full review text
            
        Returns:
            Summary section or the first few lines if no clear summary
        """
        # Look for summary section
        lines = review.split('\n')
        summary = []
        in_summary = False
        
        for line in lines:
            if line.lower().startswith('# summary') or line.lower().startswith('## summary'):
                in_summary = True
                summary.append(line)
            elif in_summary and line.startswith('#'):
                # End of summary section
                break
            elif in_summary:
                summary.append(line)
                
        if summary:
            return '\n'.join(summary)
        
        # If no clear summary section, just return the first few lines
        return '\n'.join(lines[:10]) + '\n...' 