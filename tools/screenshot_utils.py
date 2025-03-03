#!/usr/bin/env python3

import asyncio
from playwright.async_api import async_playwright
import os
import tempfile
from pathlib import Path

# Add the project root to the path for relative imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


async def take_screenshot(url: str, output_path: str = None, width: int = 1280, height: int = 720) -> str:
    """
    Take a screenshot of a webpage using Playwright.
    
    Args:
        url (str): The URL to take a screenshot of
        output_path (str, optional): Path to save the screenshot. If None, saves to a temporary file.
        width (int, optional): Viewport width. Defaults to 1280.
        height (int, optional): Viewport height. Defaults to 720.
    
    Returns:
        str: Path to the saved screenshot
    """
    if output_path is None:
        # Create a temporary file with .png extension
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        output_path = temp_file.name
        temp_file.close()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': width, 'height': height})
        
        try:
            await page.goto(url, wait_until='networkidle')
            await page.screenshot(path=output_path, full_page=True)
        finally:
            await browser.close()
    
    return output_path

def take_screenshot_sync(url: str, output_path: str = None, width: int = 1280, height: int = 720) -> str:
    """
    Synchronous wrapper for take_screenshot.
    """
    return asyncio.run(take_screenshot(url, output_path, width, height))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Take a screenshot of a webpage')
    parser.add_argument('url', help='URL to take screenshot of')
    parser.add_argument('--output', '-o', help='Output path for screenshot')
    parser.add_argument('--width', '-w', type=int, default=1280, help='Viewport width')
    parser.add_argument('--height', '-H', type=int, default=720, help='Viewport height')
    
    args = parser.parse_args()
    output_path = take_screenshot_sync(args.url, args.output, args.width, args.height)
    print(f"Screenshot saved to: {output_path}") 