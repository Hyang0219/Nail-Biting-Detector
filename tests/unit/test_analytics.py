import pytest
import os
import json
from datetime import datetime, timedelta
from utils.analytics import Analytics

@pytest.fixture
def temp_analytics_file(tmp_path):
    """Create a temporary analytics file."""
    analytics_file = tmp_path / "test_analytics.json"
    return str(analytics_file)

class TestAnalytics:
    def test_initialization(self, temp_analytics_file):
        """Test analytics initialization."""
        analytics = Analytics(temp_analytics_file)
        
        # Check initial state
        assert analytics.current_session['total_detection_count'] == 0
        assert analytics.current_session['detection_events'] == []
        assert analytics.current_session['start_time'] is not None
        assert analytics.current_session['end_time'] is None
        
        # Check data structure
        assert analytics.data['sessions'] == []
        assert analytics.data['total_detection_count'] == 0
        assert analytics.data['daily_statistics'] == {}
    
    def test_record_detection(self, temp_analytics_file):
        """Test recording detection events."""
        analytics = Analytics(temp_analytics_file)
        
        # Record some detections
        analytics.record_detection()
        analytics.record_detection()
        
        # Check session state
        assert analytics.current_session['total_detection_count'] == 2
        assert len(analytics.current_session['detection_events']) == 2
        
        # Check daily statistics
        today = datetime.now().strftime('%Y-%m-%d')
        assert today in analytics.data['daily_statistics']
        assert analytics.data['daily_statistics'][today]['count'] == 2
    
    def test_end_session(self, temp_analytics_file):
        """Test ending a session."""
        analytics = Analytics(temp_analytics_file)
        
        # Record some activity
        analytics.record_detection()
        analytics.record_detection()
        
        # End session
        analytics.end_session()
        
        # Check if session was saved
        assert len(analytics.data['sessions']) == 1
        assert analytics.data['total_detection_count'] == 2
        
        # Check if file was created and contains correct data
        assert os.path.exists(temp_analytics_file)
        with open(temp_analytics_file, 'r') as f:
            saved_data = json.load(f)
            assert len(saved_data['sessions']) == 1
            assert saved_data['total_detection_count'] == 2
    
    def test_daily_summary(self, temp_analytics_file):
        """Test getting daily summary."""
        analytics = Analytics(temp_analytics_file)
        
        # Record some detections
        analytics.record_detection()
        analytics.record_detection()
        analytics.end_session()
        
        # Get summary for last 7 days
        summary = analytics.get_daily_summary(days=7)
        
        # Check summary structure
        assert len(summary) == 7
        assert all(isinstance(day['date'], str) for day in summary)
        assert all(isinstance(day['detections'], int) for day in summary)
        assert all(isinstance(day['sessions'], int) for day in summary)
        
        # Check today's data
        today = datetime.now().strftime('%Y-%m-%d')
        today_summary = next(day for day in summary if day['date'] == today)
        assert today_summary['detections'] == 2
        assert today_summary['sessions'] == 1
    
    def test_total_statistics(self, temp_analytics_file):
        """Test getting total statistics."""
        analytics = Analytics(temp_analytics_file)
        
        # Record some detections in multiple sessions
        analytics.record_detection()
        analytics.end_session()
        
        analytics = Analytics(temp_analytics_file)  # Start new session
        analytics.record_detection()
        analytics.record_detection()
        analytics.end_session()
        
        # Get total statistics
        stats = analytics.get_total_statistics()
        
        # Check statistics
        assert stats['total_detections'] == 3
        assert stats['total_sessions'] == 2
        assert stats['average_per_session'] == 1.5 