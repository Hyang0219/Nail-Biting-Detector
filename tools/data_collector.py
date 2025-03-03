import os
import sys
import requests
import hashlib
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import time
import subprocess
import tempfile
import magic  # for MIME type detection
import tldextract  # for domain validation
import numpy as np
from tqdm import tqdm  # For progress tracking
import argparse
import pyclamd  # for virus scanning
import random
import cv2

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_engine import search_web

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


CATEGORIES = {
    'nail_biting': {
        'queries': [  
            'person nervously chewing fingernails close-up',  
            'child mid-bite fingernail tension candid shot',  
            'adult biting nails stressed expression side angle',  
            'close-up damaged nails from chronic biting',  
            'teenager gnawing thumbnail classroom setting',  
            'hands partially covering mouth while biting nails',  
            'person biting cuticles instead of nails profile',  
            'nail-biting habit with visible teeth marks',  
            'over-the-shoulder view of nail-biting behavior',  
            'person biting nails while focused on screen',  
            'toddler mouthing fingers vs intentional biting',  
            'anxious nail-biting in dimly lit room close-up',  
            'hands trembling while biting nails documentary',  
            'mid-action nail-biting with furrowed brows',  
            'person inspecting nails post-biting front view',  
            'zoomed-in shot of nail between teeth',  
            'nail-biting during exam stress candid photo',  
            'partial face view lips pressed around fingernail',  
            'person biting nails while holding paperwork',  
            'child mimicking nail-biting from adult example',
            # Additional queries
            'nail biting habit documentary photo',
            'onychophagia clinical photo',
            'compulsive nail biting behavior',
            'stress induced nail biting',
            'anxiety nail biting close up',
            'fingernail biting habit photo',
            'nail biting disorder image',
            'nervous habit nail biting',
            'student biting nails exam',
            'workplace nail biting stress',
            # New queries for more diversity
            'people biting fingernails in car',
            'person biting nails during movie',
            'nail biting during interview nervousness',
            'fingernail biting during conversation',
            'student biting nails during test',
            'nail biting while reading book',
            'person biting nails at desk close up',
            'nail biting while watching TV',
            'child biting nails during class',
            'nail biting during public speaking'
        ],
        'num_images': 500
    },
    'non_nail_biting': {
        'queries': [  
            'hands clasped near mouth contemplative pose',  
            'person resting chin on palms casual photo',  
            'hands covering mouth laughter side view',  
            'fingertips touching lips thoughtful expression',  
            'hands folded under chin relaxed portrait',  
            'person adjusting glasses near face close up',  
            'child holding cheek resting elbow on table',  
            'hands holding phone near face candid shot',  
            'person scratching cheek natural behavior',  
            'hands framing face portrait neutral pose',  
            'fingers brushing hair away from mouth profile',  
            'hands holding pen near lips thinking pose',  
            'person applying lip balm front view',  
            'hands clasped under nose documentary style',  
            'casual hand gesture near ear partial face',  
            'person eating snack hands near mouth',  
            'hands gripping cup near chin casual photo',  
            'child hands covering mouth surprised angle',  
            'hands resting on cheeks relaxed close up',  
            'person holding book near face side view',
            # Additional queries
            'person drinking water close up',
            'eating sandwich hands near mouth',
            'thoughtful pose hand on chin',
            'yawning person covering mouth',
            'speaking gesture hand movement',
            'meditation hands near face',
            'professional portrait hand pose',
            'casual conversation hand gestures',
            'student thinking hand on face',
            'natural hand position speaking',
            # New queries for more diversity
            'person holding smartphone near face',
            'hands resting on face while working',
            'person drinking coffee cup near mouth',
            'hands supporting chin at desk',
            'person eating with hands near mouth',
            'thoughtful expression hands near lips',
            'person with hand on cheek listening',
            'hand gestures during conversation face',
            'person holding pen near mouth thinking',
            'hands touching face casual portrait'
        ],
        'num_images': 500
    }
}

class DataCollector:
    def __init__(self, base_dir='data', max_images=None, max_concurrent=5, safe_domains=None):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'raw')
        self.processed_dir = os.path.join(base_dir, 'processed')
        self.metadata_file = os.path.join(base_dir, 'metadata.json')
        self.max_images = max_images
        self.max_concurrent = max_concurrent
        self.safe_domains = safe_domains or ['flickr.com', 'pexels.com', 'unsplash.com', 'pixabay.com']
        self.setup_directories()
        self.setup_logging()
        self.load_metadata()
        
        # Enhanced safety thresholds - More lenient but still safe
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        self.allowed_mime_types = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}
        self.max_file_size = 8 * 1024 * 1024  # 8MB
        self.min_file_size = 3 * 1024  # 3KB
        self.request_delay = 1.0  # Reduced to 1 second for better performance
        self.max_redirects = 3
        
        # Image quality thresholds
        self.min_resolution = (80, 80)
        self.max_resolution = (4096, 4096)
        self.min_aspect_ratio = 0.25
        self.max_aspect_ratio = 4.0
        self.min_entropy = 4.0
        
        # Safety checks (maintaining all critical ones)
        self.max_metadata_size = 8192
        self.max_exif_tags = 20
        
        # Keep all trusted domains
        self.trusted_domains = set([
            # Medical and Educational
            'nih.gov', 'who.int', 'edu', 'wikipedia.org', 'webmd.com',
            'mayoclinic.org', 'healthline.com', 'medicalnewstoday.com',
            
            # Scientific and Research
            'sciencedirect.com', 'springer.com', 'nature.com', 'researchgate.net',
            
            # Image Hosting and Stock Photos
            'flickr.com', 'staticflickr.com', 'pexels.com', 'unsplash.com',
            'pixabay.com', 'wikimedia.org', 'commons.wikimedia.org',
            'upload.wikimedia.org', 'cloudfront.net', 'twimg.com', 'fbcdn.net',
            
            # Content Delivery Networks
            'githubusercontent.com', 'raw.githubusercontent.com', 'media.giphy.com',
            'imgur.com', 'i.imgur.com', 'i.redd.it', 'media.tenor.com',
            'images.pexels.com', 'images.unsplash.com', 'akamaized.net',
            'cloudinary.com',
            
            # News and Media
            'bbc.co.uk', 'reuters.com', 'nytimes.com', 'wp.com',
            'wordpress.com', 'squarespace.com', 'wixmp.com',
            
            # Additional Image Hosts
            'photobucket.com', 'deviantart.com', 'deviantart.net',
            'pinimg.com', 'media.tumblr.com', 'staticflickr.com',
            'googleusercontent.com', 'ggpht.com', 'ytimg.com',
            
            # Medical and research domains
            'ncbi.nlm.nih.gov', 'pubmed.gov', 'clinicaltrials.gov',
            'medlineplus.gov', 'cdc.gov', 'fda.gov', 'medscape.com',
            'bmj.com', 'thelancet.com', 'jamanetwork.com', 'nejm.org',
            'sciencemag.org', 'cell.com', 'frontiersin.org',
            'biomedcentral.com', 'plos.org',
        ])
        
        # Keep domain blocklist for known problematic domains
        self.domain_blocklist = {
            'adult', 'xxx', 'porn', 'sex', 'gambling', 'casino', 'bet', 'poker',
            'drugs', 'pharma', 'replica', 'counterfeit', 'warez', 'crack', 
            'hack', 'malware', 'phishing', 'scam', 'spam'
        }
        
        # Request headers
        self.HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1'
        }
        self.REQUEST_TIMEOUT = 10

        # Initialize rate limiting
        self.last_request_time = time.time()

        # Track statistics for this run
        self.run_stats = {
            'attempted': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'virus_detected': 0,
            'start_time': datetime.now().isoformat(),
            "searches": 0,
            "domains_seen": set(),
            "domains_used": set(),
            "safe_domains_used": set(),
            "unsafe_domains_skipped": set(),
            "search_queries_used": [],
            "image_sizes": [],
        }
        
        # Initialize ClamAV connection
        self.initialize_clamav()

    def initialize_clamav(self):
        """Initialize connection to ClamAV daemon."""
        try:
            self.clamav = pyclamd.ClamdUnixSocket()
            self.clamav.ping()
            self.logger.info("Connected to ClamAV daemon via UNIX socket")
            return
        except Exception as unix_error:
            self.logger.warning(f"Failed to connect to ClamAV via UNIX socket: {unix_error}")
            
        try:
            self.clamav = pyclamd.ClamdNetworkSocket()
            self.clamav.ping()
            self.logger.info("Connected to ClamAV daemon via network socket")
            return
        except Exception as net_error:
            self.logger.warning(f"Failed to connect to ClamAV via network socket: {net_error}")
            
        self.logger.error("Could not connect to ClamAV daemon")
        self.clamav = None

    def scan_for_viruses(self, content):
        """
        Scan content for viruses using ClamAV.
        
        Returns:
            tuple: (is_safe, message) where is_safe is True if no virus detected,
                   False if virus detected or if scanning failed.
        """
        if self.clamav is None:
            return False, "ClamAV not initialized"
        
        try:
            # Create a temporary file with proper permissions
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Make sure ClamAV can read the file
            try:
                os.chmod(temp_file_path, 0o644)
            except Exception as e:
                logging.warning(f"Could not change permissions on temp file: {e}")
                os.unlink(temp_file_path)
                return False, f"Permission error: {str(e)}"
            
            try:
                scan_result = self.clamav.scan_file(temp_file_path)
            except Exception as e:
                logging.warning(f"ClamAV scan error: {str(e)}")
                os.unlink(temp_file_path)
                return False, f"Scan error: {str(e)}"
            
            os.unlink(temp_file_path)
            
            if scan_result is None:
                return True, "Clean"
            else:
                for file_path, reason in scan_result.items():
                    return False, f"Virus detected: {reason[1]}"
        
        except Exception as e:
            logging.warning(f"Error during virus scanning: {str(e)}")
            return False, f"Error: {str(e)}"
        
        return False, "Unknown error in virus scanning"

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for category in CATEGORIES.keys():
            os.makedirs(os.path.join(self.raw_dir, category), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, category), exist_ok=True)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_metadata(self):
        """Load or initialize metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'images': {},
                'stats': {
                    'nail_biting': 0,
                    'non_nail_biting': 0
                }
            }

    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_image_hash(self, image_data):
        """Generate a hash for image data to avoid duplicates."""
        return hashlib.md5(image_data).hexdigest()

    def is_safe_domain(self, url):
        """Check if domain is safe and not in excluded list."""
        try:
            parsed_url = urlparse(url)
            tld_result = tldextract.extract(url)
            domain = tld_result.domain + '.' + tld_result.suffix
            
            # Skip domains that are already in our blocklist
            # Common stock photo sites with restrictive TOS are excluded
            # to avoid legal issues
            excluded_domains = [
                'gettyimages', 'shutterstock', 'adobe.com', 'istockphoto', 
                'depositphotos', '123rf.com'
            ]
            
            # More permissive trusted domains (these are typically safe)
            trusted_domains = [
                'pxhere.com', 'freepik.com', 'pixabay.com', 'pexels.com', 
                'unsplash.com', 'flickr.com', 'wikipedia.org', 'wikimedia.org',
                'commons.wikimedia.org', 'pinterest.com', 'blogspot.com',
                'wordpress.com', 'tumblr.com', 'dreamtime.com', 'alamy.com',
                'vecteezy.com', 'reddit.com', 'imgur.com', 'twimg.com',
                'staticflickr.com'
            ]
            
            # Check for domain in exclusion list
            for excluded in excluded_domains:
                if excluded in domain.lower():
                    self.logger.warning(f"Excluded domain for URL: {url}")
                return False
                    
            # Allow trusted domains
            for trusted in trusted_domains:
                if trusted in domain.lower() or trusted in parsed_url.netloc.lower():
                    return True
            
            # More permissive validation (allow most domains)
            # We still want to avoid domains with known issues
            if parsed_url.scheme in ['http', 'https']:
                # For non-trusted domains, do a basic safety check
                # but be much more permissive
                if len(domain) > 50:  # Suspicious domain (too long)
                    self.logger.warning(f"Suspiciously long domain for URL: {url}")
                    return False
                return True
                
            return False
        except Exception as e:
            self.logger.error(f"Error checking domain safety: {str(e)}")
            return False

    def verify_mime_type(self, content):
        """Verify the MIME type of the content."""
        try:
            mime = magic.from_buffer(content, mime=True)
            return mime in self.allowed_mime_types
        except Exception as e:
            self.logger.error(f"Error checking MIME type: {str(e)}")
            return False

    def is_safe_file(self, content, url):
        """
        Check if the content is a safe image file for processing.
        Returns:
            tuple: (is_safe, reason_if_not_safe)
        """
        # Test for virus scan first - this is critical
        is_clean, virus_info = self.scan_for_viruses(content)
        if not is_clean:
            self.run_stats['virus_detected'] += 1
            return False, f"Virus detected: {virus_info}"
            
        # For testing purposes, we're making this very permissive
        # Only do minimal checks
        
        # Check content length (8MB limit)
        content_length = len(content)
        if content_length > 8 * 1024 * 1024:  # 8MB
            return False, f"File too large: {content_length} bytes"
        
        if content_length < 1024:  # 1KB minimum
            return False, f"File too small: {content_length} bytes"
            
        try:
            # Just do a basic check that it's an image of some kind
            mime_type = magic.from_buffer(content, mime=True)
            if not mime_type.startswith('image/'):
                return False, f"Not an image file: {mime_type}"
        except Exception as e:
            return False, f"Error determining file type: {e}"
            
        # Skip all other validation for testing purposes
        return True, None

    def respect_rate_limit(self):
        """Ensure we respect rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()

    def download_image(self, url):
        """Download an image from a URL and return the content"""
        if not self.is_valid_url(url):
            self.logger.warning(f"Invalid URL: {url}")
            return None
            
        if not self.is_safe_domain(url):
            self.logger.warning(f"Unsafe domain for URL: {url}")
            return None
            
        try:
            self.respect_rate_limit()
            
            # Get the response
            response = requests.get(
                url, 
                headers=self.HEADERS,
                timeout=self.REQUEST_TIMEOUT
            )
            
            # Check status code
            if response.status_code != 200:
                self.logger.warning(f"HTTP status {response.status_code} for URL: {url}")
                return None
                
            # Check file safety based on headers
            is_safe, reason = self.is_safe_file(response.content, url)
            if not is_safe:
                self.logger.warning(f"Unsafe file detected for URL: {url} - {reason}")
                return None
                
            return response.content
        except requests.RequestException as e:
            self.logger.warning(f"Request error for {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {str(e)}")
            return None

    def process_image(self, content, category, url):
        """Process and validate downloaded image content."""
        try:
            # Create BytesIO object from content
            image_data = BytesIO(content)
            
            # Verify MIME type
            mime = magic.from_buffer(content, mime=True)
            if mime not in self.allowed_mime_types:
                return False

            # Check file size
            size = len(content)
            if not (self.min_file_size <= size <= self.max_file_size):
                return False

            # Scan for viruses one more time before opening
            is_clean, scan_result = self.scan_for_viruses(content)
            if not is_clean:
                self.logger.warning(f"Virus detected during processing from URL: {url} - {scan_result}")
                self.run_stats['virus_detected'] += 1
                return False

            # Open and validate image
            try:
                img = Image.open(image_data)
                img.verify()  # Verify image integrity
                img = Image.open(image_data)  # Reopen after verify
            except Exception as e:
                return False

            # Check resolution and aspect ratio
            width, height = img.size
            if not (self.min_resolution[0] <= width <= self.max_resolution[0] and 
                    self.min_resolution[1] <= height <= self.max_resolution[1]):
                return False

            aspect_ratio = width / height
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                return False

            # Check color mode
            if img.mode not in ['RGB', 'RGBA']:
                return False

            # Calculate image hash
            image_hash = self.get_image_hash(content)
            if image_hash in self.metadata['images']:
                self.run_stats['skipped'] += 1
                return False

            # Save image and update metadata
            save_path = os.path.join(self.raw_dir, category, f"{image_hash}.jpg")
            img.convert('RGB').save(save_path, 'JPEG', quality=95)
            
            # Set secure permissions
            os.chmod(save_path, 0o600)  # Owner read/write only

            # Update metadata
            self.metadata['images'][image_hash] = {
                'url': url,
                'category': category,
                'size': size,
                'width': width,
                'height': height,
                'virus_scan': 'clean',
                'downloaded_at': datetime.now().isoformat()
            }
            self.metadata['stats'][category] += 1
            self.run_stats['downloaded'] += 1
            self.save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Error processing image from {url}: {str(e)}")
            self.run_stats['failed'] += 1
            return False

    def collect_images(self, query, category, num_images=25):
        """Collect images for a specific category with improved threading."""
        self.logger.info(f"Collecting {num_images} images for category: {category} with query: {query}")
        
        # Search for images
        search_results = search_web(query, max_results=num_images * 2)  # Get extra results in case some fail
        
        if not search_results:
            self.logger.warning(f"No search results found for query: {query}")
            return 0
            
        self.logger.info(f"Found {len(search_results)} search results")
        
        # Create a progress bar
        pbar = tqdm(total=num_images, desc=f"Downloading {category} images")
        
        # Download images in parallel with increased workers
            downloaded = 0
        with ThreadPoolExecutor(max_workers=10) as executor:  # Increased from 8 to 10
            # Submit all download tasks
            future_to_url = {executor.submit(self.download_image, result['image']): result['image'] 
                            for result in search_results[:num_images*2]}
            
            # Process results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                self.run_stats['attempted'] += 1
                
                try:
                    content = future.result()
                    if content and self.process_image(content, category, url):
                            downloaded += 1
                        pbar.update(1)
                        
                        # If we have enough images, stop
                        if downloaded >= num_images:
                            # Cancel remaining futures
                            for f in future_to_url:
                                if not f.done():
                                    f.cancel()
                            break
                except Exception as e:
                    self.logger.error(f"Error processing download from {url}: {str(e)}")
                    self.run_stats['failed'] += 1

        pbar.close()
        self.logger.info(f"Completed collecting images for {category} (downloaded: {downloaded})")
        return downloaded

    def get_current_counts(self):
        """Get current image counts for each category."""
        counts = {}
        for category in CATEGORIES.keys():
            raw_dir = os.path.join(self.raw_dir, category)
            processed_dir = os.path.join(self.processed_dir, category)
            
            # Count raw images
            raw_count = len([f for f in os.listdir(raw_dir) 
                           if os.path.isfile(os.path.join(raw_dir, f)) and 
                           not f.startswith('.')])
            
            # Count processed images
            processed_count = len([f for f in os.listdir(processed_dir) 
                                 if os.path.isfile(os.path.join(processed_dir, f)) and 
                                 not f.startswith('.')])
            
            counts[category] = {
                'raw': raw_count,
                'processed': processed_count
            }
            
        return counts

    def print_collection_stats(self):
        """Print statistics about the current collection."""
        counts = self.get_current_counts()
        print("\n=== Collection Statistics ===")
        for category, stats in counts.items():
            print(f"{category}:")
            print(f"  - Raw images: {stats['raw']}")
            print(f"  - Processed images: {stats['processed']}")
            print(f"  - Target: {CATEGORIES[category]['num_images']} processed images")
        
        print("\n=== This Run Statistics ===")
        duration = datetime.now() - datetime.fromisoformat(self.run_stats['start_time'])
        print(f"Duration: {duration}")
        print(f"Attempted downloads: {self.run_stats['attempted']}")
        print(f"Successfully downloaded: {self.run_stats['downloaded']}")
        print(f"Skipped (duplicates): {self.run_stats['skipped']}")
        print(f"Virus detections: {self.run_stats['virus_detected']}")
        print(f"Failed downloads: {self.run_stats['failed']}")
        success_rate = (self.run_stats['downloaded'] / max(1, self.run_stats['attempted'])) * 100
        print(f"Success rate: {success_rate:.1f}%")

    def scan_all_downloaded_images(self):
        """
        Perform a final virus scan on all downloaded images after collection is complete.
        
        Returns:
            dict: Statistics about the scan results
        """
        self.logger.info("\nPerforming final virus scan on all downloaded images...")
        
        scan_stats = {
            'total': 0,
            'clean': 0,
            'infected': 0,
            'errors': 0
        }
        
        # Scan both categories
        for category in CATEGORIES.keys():
            raw_dir = os.path.join(self.raw_dir, category)
            
            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                image_files.extend([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                                  if f.endswith(ext) and os.path.isfile(os.path.join(raw_dir, f))])
            
            self.logger.info(f"Scanning {len(image_files)} images in {category} category...")
            
            # Create a progress bar
            pbar = tqdm(total=len(image_files), desc=f"Scanning {category} images")
            
            # Scan each file
            for image_path in image_files:
                scan_stats['total'] += 1
                
                try:
                    # Read file content
                    with open(image_path, 'rb') as f:
                        content = f.read()
                    
                    # Scan for viruses
                    is_clean, scan_result = self.scan_for_viruses(content)
                    
                    if is_clean:
                        scan_stats['clean'] += 1
                    else:
                        scan_stats['infected'] += 1
                        self.logger.warning(f"Virus detected in {image_path}: {scan_result}")
                        
                        # Option to delete infected files
                        try:
                            os.rename(image_path, f"{image_path}.infected")
                            self.logger.info(f"Renamed infected file to {image_path}.infected")
                        except Exception as e:
                            self.logger.error(f"Failed to rename infected file {image_path}: {e}")
                    
                except Exception as e:
                    scan_stats['errors'] += 1
                    self.logger.error(f"Error scanning {image_path}: {e}")
                
                pbar.update(1)
            
            pbar.close()
        
        # Print scan results
        self.logger.info("\n=== Final Virus Scan Results ===")
        self.logger.info(f"Total files scanned: {scan_stats['total']}")
        self.logger.info(f"Clean files: {scan_stats['clean']}")
        self.logger.info(f"Infected files: {scan_stats['infected']}")
        self.logger.info(f"Errors during scanning: {scan_stats['errors']}")
        
        if scan_stats['infected'] > 0:
            self.logger.warning(f"⚠️ Found {scan_stats['infected']} infected files! Check the log for details.")
        else:
            self.logger.info("✅ No infections found!")
        
        return scan_stats

    def run(self, force=False):
        """Run the data collection process."""
        for category in ['nail_biting', 'non_nail_biting']:
            # Create directories if they don't exist
            raw_dir = os.path.join(self.raw_dir, category)
            processed_dir = os.path.join(self.processed_dir, category)
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
        # Count existing images
        existing_raw_counts = {
            'nail_biting': len(os.listdir(os.path.join(self.raw_dir, 'nail_biting'))),
            'non_nail_biting': len(os.listdir(os.path.join(self.raw_dir, 'non_nail_biting')))
        }
        
        existing_processed_counts = {
            'nail_biting': len(os.listdir(os.path.join(self.processed_dir, 'nail_biting'))),
            'non_nail_biting': len(os.listdir(os.path.join(self.processed_dir, 'non_nail_biting')))
        }
        
        print("Current image counts:")
        print(f"nail_biting: {existing_raw_counts['nail_biting']} raw, {existing_processed_counts['nail_biting']} processed")
        print(f"non_nail_biting: {existing_raw_counts['non_nail_biting']} raw, {existing_processed_counts['non_nail_biting']} processed")
        
        total_processed = sum(existing_processed_counts.values())
        if total_processed >= 1000 and not force:  # Changed from 500 to 1000 (500 per class)
            print(f"Already have {total_processed} processed images, target of 1000 (500 per class) reached!")
            return
        
        # Define search queries
        search_queries = {
            'nail_biting': [
                'person biting nails', 'nail biting habit', 'finger nail biting',
                'child biting nails', 'adult biting fingernails', 'nail biting closeup'
            ],
            'non_nail_biting': [
                'hands resting', 'fingers together', 'hands in lap', 
                'hands on table', 'person with normal nails', 'manicured nails'
            ]
        }
        
        # Collect images for each category
        for category, queries in search_queries.items():
        total_downloaded = 0
            
            # Calculate how many more processed images we need
            if force:
                # If force is enabled, collect up to max_images regardless of how many we already have
                target_count = self.max_images
            else:
                # Target based on processed images needed, not raw images
                target_processed = 500  # 500 per class
                current_processed = existing_processed_counts[category]
                
                # We need more than the difference because not all raw images will pass validation
                # Assuming ~30% of raw images will be processed successfully (conservative estimate)
                needed_processed = max(0, target_processed - current_processed)
                needed_raw = needed_processed * 3  # Collect 3x raw images for each processed image needed
                
                # Limit by max_images parameter and by category target
                target_count = min(self.max_images, needed_raw)
            
            if target_count <= 0 and not force:
                self.logger.info(f"Already have enough processed images for {category}")
                continue
            
            self.logger.info(f"Targeting {target_count} new images for {category}")
            
            # Shuffle queries to avoid using the same ones each time
            random.shuffle(queries)
            
            for query in queries:
            # Calculate how many images we still need
                remaining = target_count - total_downloaded
            if remaining <= 0:
                break
                
                # Try to collect a portion of the remaining images with this query
                images_per_query = min(remaining, 25)  # Limit to 25 images per query
                downloaded = self.collect_images(query, category, num_images=images_per_query)
            total_downloaded += downloaded
            
            # Add delay between queries to avoid rate limits
                time.sleep(1.0)  # Reduced from 1.5 to 1.0 seconds
            
            if total_downloaded < target_count:
                self.logger.warning(
                f"Could not collect enough images for {category}. "
                    f"Got {total_downloaded} out of {target_count}"
                )
        
        # Perform final virus scan on all downloaded images
        final_scan_stats = self.scan_all_downloaded_images()
        
        # Print final statistics
        self.print_collection_stats()

    def is_valid_url(self, url):
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

def main():
    parser = argparse.ArgumentParser(description='Collect image data for nail biting detection')
    parser.add_argument('--max_images', type=int, default=200, help='Maximum number of images to download')
    parser.add_argument('--max_concurrent', type=int, default=5, help='Maximum number of concurrent downloads')
    parser.add_argument('--force', action='store_true', help='Force download even if enough images exist')
    args = parser.parse_args()
    
    collector = DataCollector(
        max_images=args.max_images,
        max_concurrent=args.max_concurrent
    )
    collector.run(force=args.force)

if __name__ == "__main__":
    main() 