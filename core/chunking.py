import ast
import os

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

# Check if ast.get_source_segment is available (Python 3.8+)
HAS_GET_SOURCE_SEGMENT = hasattr(ast, 'get_source_segment')

def split_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """
    Split text into chunks with overlap.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        next_start = start + chunk_size - overlap
        if next_start <= start:
            start += 1
        else:
            start = next_start

    return chunks

def chunk_python_code_by_structure(code_content, max_char_size, overlap):
    """
    Chunk Python code based on its AST structure.
    Falls back to character chunking if AST parsing fails or Python version < 3.8.
    
    Args:
        code_content (str): Python code content to chunk
        max_char_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of code chunks
    """
    chunks = []
    if not code_content:
        return chunks

    if not HAS_GET_SOURCE_SEGMENT:
        print("Syntax chunking fallback: Using character-based chunking due to Python version.")
        return split_text(code_content, max_char_size, overlap)

    try:
        tree = ast.parse(code_content)
        lines = code_content.splitlines(keepends=True)
        current_pos = 0

        for node in tree.body:
            start_lineno = node.lineno
            end_lineno = node.end_lineno

            node_start_char_pos = 0
            temp_line_count = 0
            temp_char_count = 0
            while temp_line_count < start_lineno - 1 and temp_line_count < len(lines):
                temp_char_count += len(lines[temp_line_count])
                temp_line_count += 1
            node_start_char_pos = temp_char_count

            if node_start_char_pos > current_pos:
                prefix_code = code_content[current_pos:node_start_char_pos].strip()
                if prefix_code:
                    if len(prefix_code) > max_char_size:
                        chunks.extend(split_text(prefix_code, max_char_size, overlap))
                    else:
                        chunks.append(prefix_code)

            segment = ast.get_source_segment(code_content, node, padded=True)
            if segment:
                segment = segment.strip()
                if len(segment) > max_char_size:
                    node_type = type(node).__name__
                    node_name = getattr(node, 'name', '[Unnamed]')
                    chunks.append(f"# --- Start of large {node_type}: {node_name} ---")
                    chunks.extend(split_text(segment, max_char_size, overlap))
                    chunks.append(f"# --- End of large {node_type}: {node_name} ---")
                else:
                    chunks.append(segment)

            node_end_char_pos = 0
            temp_line_count = 0
            temp_char_count = 0
            while temp_line_count < end_lineno and temp_line_count < len(lines):
                temp_char_count += len(lines[temp_line_count])
                temp_line_count += 1
            current_pos = temp_char_count

        if current_pos < len(code_content):
            suffix_code = code_content[current_pos:].strip()
            if suffix_code:
                if len(suffix_code) > max_char_size:
                    chunks.extend(split_text(suffix_code, max_char_size, overlap))
                else:
                    chunks.append(suffix_code)

    except (SyntaxError, ValueError) as e:
        print(f"Warning: AST parsing failed for syntax-aware chunking: {e}. Falling back to character chunking.")
        return split_text(code_content, max_char_size, overlap)
    except Exception as e:
        print(f"Unexpected error during syntax-aware chunking: {e}. Falling back to character chunking.")
        import traceback
        print(traceback.format_exc())
        return split_text(code_content, max_char_size, overlap)

    return [chunk for chunk in chunks if chunk] 