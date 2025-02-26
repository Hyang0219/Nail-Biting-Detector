#!/usr/bin/env python3

import argparse
import sys
import time
from duckduckgo_search import DDGS
import logging

def search_with_retry(query, max_results=10, max_retries=3):
    """
    Search using DuckDuckGo and return results with image URLs.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts
    """
    backoff_time = 1  # Initial backoff time in seconds
    
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: Searching for query: {query} (attempt {attempt + 1}/{max_retries})", 
                  file=sys.stderr)
            
            with DDGS() as ddgs:
                # Use the images method to get image search results
                results = []
                for r in ddgs.images(
                    query,
                    region='us-en',  # Use English results
                    safesearch='off',  # Allow all results
                    size='All',  # All sizes
                    color='color',  # Only color images
                    type_image=None,
                    layout=None,
                    license_image=None,
                    max_results=max_results * 2  # Get extra results in case some fail
                ):
                    if isinstance(r, dict) and 'image' in r:
                        results.append(r)
                        if len(results) >= max_results:
                            break
                
            if not results:
                print("DEBUG: No results found", file=sys.stderr)
                return []
            
            print(f"DEBUG: Found {len(results)} results", file=sys.stderr)
            return results
                
        except Exception as e:
            print(f"ERROR: Attempt {attempt + 1}/{max_retries} failed: {str(e)}", file=sys.stderr)
            if attempt < max_retries - 1:  # If not the last attempt
                print(f"DEBUG: Waiting {backoff_time} seconds before retry...", file=sys.stderr)
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
            else:
                print(f"ERROR: All {max_retries} attempts failed", file=sys.stderr)
                return []  # Return empty list instead of raising exception

def format_results(results):
    """Format and print search results."""
    for i, r in enumerate(results, 1):
        print(f"\n=== Result {i} ===")
        print(f"URL: {r.get('image', 'N/A')}")
        print(f"Title: {r.get('title', 'N/A')}")
        print(f"Source: {r.get('source', 'N/A')}")

def search(query, max_results=10, max_retries=3):
    """
    Main search function that handles search with retry mechanism.
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts
    """
    try:
        results = search_with_retry(query, max_results, max_retries)
        if results:
            format_results(results)
            
    except Exception as e:
        print(f"ERROR: Search failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

def search_web(query, max_results=20):
    """
    Search the web using DuckDuckGo's image search.
    Returns a list of dictionaries containing 'title', 'url', and 'image' URLs.
    """
    try:
        results = []
        with DDGS() as ddgs:
            # Use the images method to get image search results
            for r in ddgs.images(
                query,
                region='us-en',  # Use English results
                safesearch='off',  # Allow all results
                size='All',  # All sizes
                color='color',  # Only color images
                type_image=None,
                layout=None,
                license_image=None,
                max_results=max_results * 2  # Get extra results in case some fail
            ):
                if isinstance(r, dict) and 'image' in r:
                    results.append(r)
                    if len(results) >= max_results:
                        break
                        
            if not results:
                logging.error(f"No results found for query: {query}")
                return []
                
            logging.info(f"Found {len(results)} results for query: {query}")
            return results
            
    except Exception as e:
        logging.error(f"Error searching web: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Search using DuckDuckGo API")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10,
                      help="Maximum number of results (default: 10)")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="Maximum number of retry attempts (default: 3)")
    
    args = parser.parse_args()
    search(args.query, args.max_results, args.max_retries)

if __name__ == "__main__":
    main()
