import json
import os
from datetime import datetime, timedelta
import logging
import traceback

class Analytics:
    def __init__(self, analytics_file=None):
        try:
            if analytics_file is None:
                analytics_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
                os.makedirs(analytics_dir, exist_ok=True)
                analytics_file = os.path.join(analytics_dir, 'analytics.json')
            
            self.analytics_file = analytics_file
            self.current_session = {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'detection_events': [],
                'total_detection_count': 0
            }
            self.load_data()
        except Exception as e:
            logging.error(f"Failed to initialize analytics: {e}")
            logging.error(traceback.format_exc())
            # Set up default values to prevent NoneType errors
            self.analytics_file = "analytics.json" if analytics_file is None else analytics_file
            self.current_session = {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'detection_events': [],
                'total_detection_count': 0
            }
            self.data = {
                'sessions': [],
                'total_detection_count': 0,
                'daily_statistics': {}
            }
    
    def load_data(self):
        """Load existing analytics data."""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r') as f:
                    self.data = json.load(f)
            else:
                self.data = {
                    'sessions': [],
                    'total_detection_count': 0,
                    'daily_statistics': {}
                }
        except Exception as e:
            logging.error(f"Failed to load analytics data: {e}")
            logging.error(traceback.format_exc())
            self.data = {
                'sessions': [],
                'total_detection_count': 0,
                'daily_statistics': {}
            }
    
    def save_data(self):
        """Save analytics data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.analytics_file)), exist_ok=True)
            
            with open(self.analytics_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save analytics data: {e}")
            logging.error(traceback.format_exc())
    
    def record_detection(self):
        """Record a nail-biting detection event."""
        try:
            timestamp = datetime.now().isoformat()
            self.current_session['detection_events'].append(timestamp)
            self.current_session['total_detection_count'] += 1
            
            # Update daily statistics
            date = datetime.now().strftime('%Y-%m-%d')
            if date not in self.data['daily_statistics']:
                self.data['daily_statistics'][date] = {
                    'count': 0,
                    'sessions': 0
                }
            self.data['daily_statistics'][date]['count'] += 1
        except Exception as e:
            logging.error(f"Failed to record detection: {e}")
            logging.error(traceback.format_exc())
    
    def end_session(self):
        """End the current monitoring session."""
        try:
            self.current_session['end_time'] = datetime.now().isoformat()
            
            # Update session count in daily statistics
            date = datetime.now().strftime('%Y-%m-%d')
            if date in self.data['daily_statistics']:
                self.data['daily_statistics'][date]['sessions'] += 1
            
            # Add session to history
            self.data['sessions'].append(self.current_session)
            self.data['total_detection_count'] += self.current_session['total_detection_count']
            
            # Save updated data
            self.save_data()
        except Exception as e:
            logging.error(f"Failed to end session: {e}")
            logging.error(traceback.format_exc())
    
    def get_daily_summary(self, days=7):
        """Get detection summary for the last N days."""
        try:
            summary = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days-1)
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str in self.data['daily_statistics']:
                    stats = self.data['daily_statistics'][date_str]
                    summary.append({
                        'date': date_str,
                        'detections': stats['count'],
                        'sessions': stats['sessions']
                    })
                else:
                    summary.append({
                        'date': date_str,
                        'detections': 0,
                        'sessions': 0
                    })
                current_date += timedelta(days=1)
            
            return summary
        except Exception as e:
            logging.error(f"Failed to get daily summary: {e}")
            logging.error(traceback.format_exc())
            # Return a safe default
            return [{'date': datetime.now().strftime('%Y-%m-%d'), 'detections': 0, 'sessions': 0}]
    
    def get_total_statistics(self):
        """Get overall statistics."""
        try:
            return {
                'total_detections': self.data['total_detection_count'],
                'total_sessions': len(self.data['sessions']),
                'average_per_session': (
                    self.data['total_detection_count'] / len(self.data['sessions'])
                    if self.data['sessions'] else 0
                )
            }
        except Exception as e:
            logging.error(f"Failed to get total statistics: {e}")
            logging.error(traceback.format_exc())
            # Return safe defaults
            return {
                'total_detections': 0,
                'total_sessions': 0,
                'average_per_session': 0
            } 