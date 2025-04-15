import os
import subprocess
import shlex
import re
import logging

logger = logging.getLogger("CodeReviewTool.Flake8")

def run_flake8(filepath):
    """
    Run flake8 static analysis on a Python file.
    
    Args:
        filepath (str): Path to the Python file to analyze
        
    Returns:
        str: flake8 output or error message
    """
    # Validate file exists and is a Python file
    if not filepath or not isinstance(filepath, str):
        logger.warning("Invalid filepath provided to run_flake8")
        return "(Error: Invalid file path)"
        
    # Normalize path and validate it exists
    try:
        filepath = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.isfile(filepath):
            logger.warning(f"File does not exist: {filepath}")
            return "(Error: File not found)"
    except Exception as e:
        logger.error(f"Error validating filepath: {e}")
        return f"(Error: Invalid file path: {e})"
    
    # Only analyze Python files
    if not filepath.lower().endswith('.py'):
        return None
    
    # Security check: Validate filepath contains no shell metacharacters
    if not re.match(r'^[\w\-. /\\:]+$', filepath):
        logger.warning(f"Filepath contains potentially unsafe characters: {filepath}")
        return "(Error: Filepath contains invalid characters)"
        
    try:
        # Use a list of arguments instead of shell=True for security
        # and escape the filepath properly
        result = subprocess.run(
            ['flake8', filepath],
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        if result.stderr:
            logger.warning(f"Flake8 stderr for {os.path.basename(filepath)}: {result.stderr}")
        return result.stdout.strip() if result.stdout else "(No issues found by flake8)"

    except FileNotFoundError:
        logger.error("'flake8' command not found. Is it installed and in PATH?")
        return "(Error: flake8 command not found)"
    except subprocess.TimeoutExpired:
        logger.warning(f"Flake8 timed out for {os.path.basename(filepath)}.")
        return "(Error: flake8 timed out)"
    except Exception as e:
        logger.error(f"Error running flake8 on {os.path.basename(filepath)}: {e}")
        return f"(Error running flake8: {e})"

def filter_flake8_output(flake8_text, ignore_codes=None):
    """
    Filter flake8 output to remove noise and focus on significant issues.
    
    Args:
        flake8_text (str): Raw flake8 output text
        ignore_codes (list): List of flake8 code prefixes to ignore
        
    Returns:
        str: Filtered flake8 output
    """
    if ignore_codes is None:
        ignore_codes = ['E1', 'E2', 'E501', 'W']
        
    if not flake8_text or flake8_text.strip() == "(No issues found by flake8)":
        return ""

    filtered_lines = []
    original_lines = flake8_text.strip().split('\n')

    for line in original_lines:
        try:
            parts = line.split(':', 3)
            if len(parts) == 4:
                code_and_message = parts[3].strip()
                is_ignored = False
                for ignore_code in ignore_codes:
                    if code_and_message.startswith(ignore_code + ' '):
                        is_ignored = True
                        break
                if not is_ignored:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        except Exception as e:
            print(f"Warning: Error parsing flake8 line '{line}': {e}")
            filtered_lines.append(line)

    if not filtered_lines:
        return "  -> No significant flake8 issues found after filtering.\n"
    elif len(filtered_lines) == len(original_lines):
        return flake8_text
    else:
        header = "  -> Filtered flake8 issues found:\n"
        return header + "\n".join(f"    {l}" for l in filtered_lines) + "\n" 