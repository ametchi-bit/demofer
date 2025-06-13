# language.py - Updated LanguageManager class (Fixed callback issue)

import streamlit as st
from typing import Dict, Any
import json
from pathlib import Path


class LanguageManager:
    """Comprehensive language management system with proper state handling"""
    
    def __init__(self):
        self.supported_languages = {
            'en': {'name': 'English', 'flag': '🇺🇸'},
            'fr': {'name': 'Français', 'flag': '🇫🇷'}
        }
        self.default_language = 'en'
        self.translations = self._load_translations()
        
        # Initialize language from query params or session state
        self._initialize_language()
        self.current_language = self._get_current_language()
    
    def _initialize_language(self):
        """Initialize language from query parameters or set default"""
        try:
            # Check query parameters first (using current Streamlit API)
            query_params = st.query_params
            
            if "lang" in query_params:
                lang_from_query = query_params["lang"]
                if lang_from_query in self.supported_languages:
                    st.session_state.language = lang_from_query
                else:
                    # Invalid language in query, set default
                    st.session_state.language = self.default_language
                    st.query_params.lang = self.default_language
            else:
                # No language in query params, check session state
                if 'language' not in st.session_state:
                    st.session_state.language = self.default_language
                
                # Set query param to match session state
                st.query_params.lang = st.session_state.language
                
        except Exception as e:
            # Fallback to default language
            st.session_state.language = self.default_language
            try:
                st.query_params.lang = self.default_language
            except:
                pass
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load all translations"""
        return {
            'en': self._get_english_translations(),
            'fr': self._get_french_translations()
        }
    
    def _get_current_language(self) -> str:
        """Get current language from session state"""
        return st.session_state.get('language', self.default_language)
    
    def set_language(self, language_code: str):
        """Set the current language and update query params"""
        if language_code in self.supported_languages:
            st.session_state.language = language_code
            self.current_language = language_code
            
            # Update query parameters
            try:
                st.query_params.lang = language_code
            except:
                pass
    
    def get_text(self, key: str, default: str = None) -> str:
        """Get translated text for a key"""
        if default is None:
            default = key
        
        # Always get fresh current language
        current_lang = st.session_state.get('language', self.default_language)
        
        return self.translations.get(
            current_lang, 
            self.translations[self.default_language]
        ).get(key, default)
    
    def language_selector(self):
        """Render enhanced language selector in sidebar with proper state management"""
        st.sidebar.markdown("### 🌐 Language / Langue")
        
        # Create language options for display
        language_display_names = []
        language_codes = []
        
        for code, info in self.supported_languages.items():
            language_display_names.append(f"{info['flag']} {info['name']}")
            language_codes.append(code)
        
        # Get current language index
        current_lang = st.session_state.get('language', self.default_language)
        try:
            current_index = language_codes.index(current_lang)
        except ValueError:
            current_index = 0
            st.session_state.language = self.default_language
        
        # Language selector with callback (FIXED - No st.rerun() in callback)
        def on_language_change():
            """Callback function when language changes"""
            selected_index = st.session_state.language_selector_key
            selected_language = language_codes[selected_index]
            
            # Update language (but don't call st.rerun() here)
            self.set_language(selected_language)
            
            # Set a flag to trigger rerun outside of callback
            st.session_state.language_changed = True
        
        # Create the selectbox
        selected_index = st.sidebar.selectbox(
            "Select Language / Sélectionner la Langue",
            range(len(language_display_names)),
            format_func=lambda x: language_display_names[x],
            index=current_index,
            key="language_selector_key",
            on_change=on_language_change
        )
        
        # Check if language was changed and trigger rerun outside of callback
        if st.session_state.get('language_changed', False):
            st.session_state.language_changed = False  # Reset flag
            st.rerun()  # This is outside the callback, so it works
        
        # Show current language status
        # current_lang_info = self.supported_languages[current_lang]
        # st.sidebar.info(f"Current: {current_lang_info['flag']} {current_lang_info['name']}")
    
    def initialize(self):
        """Initialize language manager"""
        self._initialize_language()
        self.current_language = st.session_state.language
    
    def _get_english_translations(self) -> Dict[str, str]:
        """Complete English translations"""
        return {
            # Application Title and General
            'app_title': 'Computer Vision Engin',
            'app_subtitle': 'Advanced Computer Vision & Analytics System',
            'professional_computer_vision_system': 'Professional Computer Vision & Analytics System',
            'vision_platform_title': 'YOLO11 Vision Platform',
            
            # Navigation and Interface
            'model_config': 'Model Configuration',
            'source_selection': 'Source Selection',
            'camera_settings': 'Camera Settings',
            'processing_options': 'Processing Options',
            'system_management': 'System Management',
            
            # Task Selection
            'task_selection': 'Select Analysis Task',
            'confidence': 'Detection Confidence',
            'choose_source': 'Choose Input Source',
            'source_selection_help': 'Select your input source: Image, Video, or Live Camera',
            
            # Model Types
            'enhanced_license_plate': 'Enhanced License Plate Detection',
            'parking_detection': 'Parking Detection',
            'counting_alert_system': 'Counting & Alert System',
            'current_model': 'Current Model',
            
            # Enhanced License Plate
            'professional_tracking': 'Professional vehicle tracking with custom corner borders',
            'advanced_ocr': 'Advanced OCR with format validation and character correction',
            'real_time_visualization': 'Real-time professional visualization with overlay displays',
            'comprehensive_analytics': 'Comprehensive analytics with session persistence',
            'professional_visualization': 'Professional visualization with custom borders',
            'real_time_ocr': 'Real-time OCR with format validation',
            'advanced_tracking': 'Advanced tracking and interpolation',
            'enhanced_license_plate_not_available': 'Enhanced License Plate Detection not available',
            'loading_enhanced_license_plate': 'Loading Enhanced License Plate System',
            'failed_to_initialize_enhanced_system': 'Failed to initialize enhanced license plate system',
            'enhanced_system_loaded': 'Enhanced License Plate System Loaded',
            'license_plate_init_error': 'License plate initialization error',
            
            # Parking Detection
            'task': 'Task',
            'features': 'Features',
            'parking_occupancy': 'Parking Space Occupancy Monitoring',
            'parking_features': 'Rectangle & Polygon spaces, Real-time count',
            'pixel_threshold': 'Parking Pixel Threshold',
            'pixel_threshold_help': 'Lower values = more likely to be "Free". Adjust based on space size.',
            'upload_parking_positions': 'No parking positions file found - upload CarParkPos',
            'preprocessing_error': 'Preprocessing Error',
            'parking_error': 'Parking Error',
            'no_parking_spots': 'No parking spots loaded',
            'no_parking_positions': 'No parking positions loaded',
            'parking_occupancy_mode': 'Parking occupancy monitoring mode',
            'parking_detection_not_available': 'Parking Detection not available',
            'parking_system_loaded': 'Parking System Loaded',
            'spaces': 'spaces',
            'no_parking_positions_loaded': 'No parking positions loaded - please upload configuration',
            'parking_init_error': 'Parking system initialization error',
            'yolo_detections': 'YOLO Detections',
            'detection_visualization': 'Detection Visualization',
            'show_yolo_detections': 'Show YOLO Vehicle Detections',
            'yolo_visualization_help': 'Display YOLO bounding boxes and confidence scores for all detected objects',
            'yolo_box_color': 'YOLO Box Color',
            'parking_spot_colors': 'Parking Spot Colors',
            'free_spots_color': 'Free Spots: Green',
            'occupied_spots_color': 'Occupied Spots: Red',
            'recent_events': 'Recent Events',
            'detection_settings': 'Detection Settings',
            'yolo_visualization': 'YOLO Visualization',
            'pixel_threshold': 'Pixel Threshold',
            'target_classes': 'Target Classes',
            'detection_confidence': 'Detection Confidence',
            
            # Counting & Alert System
            'motion_detection': 'Motion Detection & Line Crossing',
            'counting_features': 'Polygon/Rectangle zones, Counting lines, Alerts',
            'upload_shapes_file': 'Upload Shapes Configuration File',
            'counting_error': 'Counting System Error',
            'no_shapes': 'No shapes configuration loaded',
            'counting_alert_mode': 'Counting & Alert System mode',
            'zone_shapes': 'Zone Shapes (CountingZonePos copy)',
            'hall_shapes': 'Hall Shapes (CountingHallPos copy 2)',
            'json_config': 'JSON Config (shapes_config.json)',
            'select_shape_config': 'Select Shape Configuration',
            'shape_config_help': 'Choose which predefined set of shapes to use',
            'load_shapes': 'Load Shapes',
            'loaded': 'Loaded',
            'shapes': 'shapes',
            'failed_to_load_shapes': 'Failed to load shapes',
            'counting_system_not_available': 'Counting System not available',
            'counting_system_loaded': 'Counting System Loaded',
            'no_shapes_loaded': 'No shapes loaded - please configure shapes',
            'counting_init_error': 'Counting system initialization error',
            'stair_shapes': 'Stair Shapes (CountingStairPos copy)',
            
            # Camera Controls
            'select_camera': 'Select Camera',
            'camera_selection_help': 'Choose camera for live processing',
            'connect_camera': 'Connect Camera',
            'disconnect_camera': 'Disconnect Camera',
            'camera_connected': 'Camera Connected',
            'camera_disconnected': 'Camera Disconnected',
            'camera_error': 'Camera Error',
            'camera_connection_failed': 'Camera connection failed',
            'camera_not_available': 'Camera not available',
            'connection_error': 'Connection error',
            'disconnection_error': 'Disconnection error',
            'camera_not_connected': 'Camera not connected',
            'connecting': 'Connecting',
            'selected': 'Selected',
            
            # Advanced Camera Settings
            'advanced_camera_settings': 'Advanced Camera Settings',
            'connection_mode': 'Connection Mode',
            'connection_mode_help': 'RTSP connection optimization mode',
            'stream_quality': 'Stream Quality',
            'stream_quality_help': 'Video stream quality selection',
            'buffer_size': 'Buffer Size',
            'buffer_size_help': 'Frame buffer size for smooth playback',
            'auto_reconnect': 'Auto Reconnect',
            'auto_reconnect_help': 'Automatically reconnect on disconnection',
            
            # Processing Controls
            'start_processing': 'Start Processing',
            'stop_processing': 'Stop Processing',
            'reset_tracking': 'Reset Tracking',
            'save_results': 'Save Results',
            'processing_started': 'Processing Started',
            'processing_stopped': 'Processing Stopped',
            'tracking_reset': 'Tracking data reset successfully',
            'results_saved': 'Results saved successfully',
            'no_results_to_save': 'No results to save',
            'save_error': 'Error saving results',
            'reset_error': 'Error resetting tracking',
            
            # Live Processing
            'start_live': 'Start Live',
            'stop_live': 'Stop Live',
            'live_processing_started': 'Live processing started',
            'live_processing_stopped': 'Live processing stopped',
            'live_processing_active': 'Live Processing Active',
            'live_processing_error': 'Live processing error',
            'no_frame_available': 'No frame available',
            'connected_ready_for_processing': 'Connected - Ready for Processing',
            'ready': 'Ready',
            'click_start_processing': 'Click Start to Begin Processing',
            
            # Source Types
            'choose_file': 'Choose File',
            'upload_image': 'Upload Image',
            'upload_image_help': 'Upload an image for AI analysis',
            'upload_image_prompt': 'Upload an image to start analysis',
            'upload_image_to_analyze': 'Upload an image to see analysis results',
            'choose_video': 'Choose Video',
            'choose_video_help': 'Select a video file for processing',
            'video_not_found': 'Video file not found',
            'no_camera_connected': 'No Camera Connected',
            'connect_camera_sidebar': 'Use the sidebar to connect to a camera',
            'select_camera_to_start': 'Select and connect a camera to start',
            
            # Processing Interface
            'detection_interface': 'Detection Interface',
            'image_analysis': 'Image Analysis',
            'video_analysis': 'Video Analysis',
            'live_camera_analysis': 'Live Camera Analysis',
            'original_image': 'Original Image',
            'processed_result': 'Processed Result',
            'processed_frame': 'Processed Frame',
            'default_image': 'Default Image',
            'analyze_image': 'Analyze Image',
            'process_video': 'Process Video',
            'processing_with': 'Processing with',
            'image_processing_failed': 'Image processing failed',
            'image_processing_error': 'Image processing error',
            
            # Video Processing
            'processing_video': 'Processing video',
            'frames': 'frames',
            'video_open_error': 'Could not open video file',
            'video_processing_complete': 'Video processing complete',
            'video_processing_error': 'Video processing error',
            'video_saved': 'Video saved',
            'save_processed_video': 'Save Processed Video',
            'processing': 'Processing',
            'speed': 'Speed',
            'completed': 'Completed',
            
            # Live Camera
            'real_time_feed': 'Real-time Feed',
            'live_feed': 'Live Feed',
            'camera': 'Camera',
            'available_cameras': 'Available Cameras',
            'live_controls': 'Live Controls',
            'live_statistics': 'Live Statistics',
            'live_license_plate_info': 'Live License Plate Info',
            'live_parking_info': 'Live Parking Info',
            'live_counting_info': 'Live Counting Info',
            
            # Tabs
            'enhanced_live_detection': 'Enhanced Live Detection',
            'real_time_analytics': 'Real-time Analytics',
            'license_plate_results': 'License Plate Results',
            'reports_export': 'Reports & Export',
            'live_detection': 'Live Detection',
            'analytics_dashboard': 'Analytics Dashboard',
            'detection_results': 'Detection Results',
            
            # Analytics
            'start_processing_for_analytics': 'Start processing to see real-time analytics',
            'total_detections': 'Total Detections',
            'unique_plates': 'Unique Plates',
            'avg_confidence': 'Avg Confidence',
            'detections_per_minute': 'Detections/Minute',
            'detection_timeline': 'Detection Timeline',
            'confidence_analysis': 'Confidence Analysis',
            'plate_discovery': 'Plate Discovery',
            'performance_trends': 'Performance Trends',
            'detection_timeline_chart': 'Real-time License Plate Detection Timeline',
            'confidence_score': 'Confidence Score',
            'detection_time': 'Detection Time',
            'excellent_threshold': 'Excellent Threshold',
            'good_threshold': 'Good Threshold',
            'excellent': 'Excellent',
            'good': 'Good',
            'fair': 'Fair',
            'need_more_detections_for_timeline': 'Need at least 2 detections for timeline chart',
            'confidence_distribution': 'Confidence Score Distribution',
            'frequency': 'Frequency',
            'quality_distribution': 'Detection Quality Distribution',
            'no_confidence_data_available': 'No confidence data available',
            'unique_plates_discovery': 'Unique License Plates Discovery Over Time',
            'cumulative_unique_plates': 'Cumulative Unique Plates',
            'time': 'Time',
            'discovered_plates': 'Discovered Plates',
            'discovery_metrics': 'Discovery Metrics',
            'discovery_rate': 'Discovery Rate',
            'total_unique': 'Total Unique',
            'no_plate_discovery_data': 'No plate discovery data available',
            'performance_trend_analysis': 'Performance Trend Analysis',
            'individual_detections': 'Individual Detections',
            'rolling_average': 'Rolling Average',
            'need_more_data_for_trends': 'Need at least 10 detections for trend analysis',
            
            # Parking Analytics
            'parking_system_not_initialized': 'Parking system not initialized',
            'total_spaces': 'Total Spaces',
            'free_spaces': 'Free Spaces',
            'occupied_spaces': 'Occupied Spaces',
            'occupancy_rate': 'Occupancy Rate',
            'parking_events_analysis': 'Parking Events Analysis',
            'parking_entries_by_hour': 'Parking Entries by Hour',
            'hour_of_day': 'Hour of Day',
            'number_of_entries': 'Number of Entries',
            'avg_parking_duration': 'Avg Parking Duration',
            'max_parking_duration': 'Max Parking Duration',
            'no_parking_events_recorded': 'No parking events recorded',
            'parking_overview': 'Parking Overview',
            'parking_spaces': 'Parking Spaces',
            'spot_status': 'Spot Status',
            'parking_report': 'Parking Report',
            
            # Counting Analytics
            'counting_system_not_initialized': 'Counting system not initialized',
            'total_zones': 'Total Zones',
            'active_alerts': 'Active Alerts',
            'total_crossings': 'Total Crossings',
            'zone_activity_analysis': 'Zone Activity Analysis',
            'zone_name': 'Zone Name',
            'crossings': 'Crossings',
            'crossings_per_zone': 'Crossings per Zone',
            'live_events_timeline': 'Live Events Timeline',
            'events_timeline': 'Events Timeline',
            'counting_results': 'Counting Results',
            'alert_count': 'Alert Count',
            'crossing_count': 'Crossing Count',
            'zone_activity': 'Zone Activity',
            'alerts': 'Alerts',
            'zone_details': 'Zone Details',
            
            # Results Management
            'results_management': 'Results Management',
            'available_result_files': 'Available Result Files',
            'select_result_file': 'Select a result file to analyze',
            'file_info': 'File Information',
            'filename': 'Filename',
            'size': 'Size',
            'modified': 'Modified',
            'estimated_rows': 'Estimated Rows',
            'analyze_selected_file': 'Analyze Selected File',
            'no_result_files_found': 'No result files found',
            'go_to_detection_tab_video': 'Go to Detection tab → Video to process videos',
            'go_to_detection_tab_camera': 'Go to Detection tab → Camera for live sessions',
            'upload_results_file': 'Upload Results File',
            'file_analysis': 'File Analysis',
            'total_records': 'Total Records',
            'columns': 'Columns',
            'license_plates': 'License Plates',
            'data_preview': 'Data Preview',
            'download_csv': 'Download CSV',
            'file_analysis_error': 'File analysis error',
            'file_uploaded_and_saved': 'File uploaded and saved',
            'refresh_page_to_see_file': 'Refresh page to see file in the list',
            'file_upload_error': 'File upload error',
            
            # Detection Results
            'detection_results': 'Detection Results',
            'vehicle_id': 'Vehicle ID',
            'license_plate': 'License Plate',
            'confidence': 'Confidence',
            'detected': 'Detected',
            'license_plates': 'license plates',
            'parking_status': 'Parking Status',
            'counting_alert_status': 'Counting & Alert Status',
            'objects': 'objects',
            'detection_details': 'Detection Details',
            'no_objects_detected': 'No objects detected',
            
            # Reports and Export
            'session_export': 'Session Export',
            'analytics_export': 'Analytics Export',
            'generate_report': 'Generate Report',
            'export_session_data': 'Export Session Data',
            'export_analytics': 'Export Analytics',
            'generate_summary_report': 'Generate Summary Report',
            'license_plate_specific_exports': 'License Plate Specific Exports',
            'export_detection_results': 'Export Detection Results',
            'export_performance_report': 'Export Performance Report',
            'parking_specific_exports': 'Parking Specific Exports',
            'export_parking_report': 'Export Parking Report',
            'counting_specific_exports': 'Counting Specific Exports',
            'export_counting_log': 'Export Counting Log',
            'download_session_json': 'Download Session JSON',
            'download_analytics_json': 'Download Analytics JSON',
            'download_report': 'Download Report',
            'download_license_plate_csv': 'Download License Plate CSV',
            'download_performance_json': 'Download Performance JSON',
            'download_parking_csv': 'Download Parking CSV',
            'download_counting_csv': 'Download Counting CSV',
            'session_data_exported': 'Session data exported successfully',
            'analytics_data_exported': 'Analytics data exported successfully',
            'report_generated': 'Report generated successfully',
            'license_plate_results_exported': 'License plate results exported',
            'performance_report_exported': 'Performance report exported',
            'parking_report_exported': 'Parking report exported',
            'counting_log_exported': 'Counting log exported',
            'export_error': 'Export error',
            'report_generation_error': 'Report generation error',
            'license_processor_not_available': 'License processor not available',
            'no_results_to_export': 'No results to export',
            'download_preparation_failed': 'Download preparation failed',
            'no_performance_data': 'No performance data available',
            'parking_detector_not_available': 'Parking detector not available',
            'no_parking_events_to_export': 'No parking events to export',
            'counting_system_not_available': 'Counting system not available',
            'no_counting_events_to_export': 'No counting events to export',
            
            # System Management
            'system_status': 'System Status',
            'system_information_status': 'System Information & Status',
            'memory_usage': 'Memory Usage',
            'maintenance': 'Maintenance',
            'clear_analytics_data': 'Clear Analytics Data',
            'reset_all_systems': 'Reset All Systems',
            'debug_information': 'Debug Information',
            'session_variables': 'Session Variables',
            'analytics_entries': 'Analytics Entries',
            'analytics_data_cleared': 'Analytics data cleared successfully',
            'all_systems_reset': 'All systems reset successfully',
            'clear_data_error': 'Error clearing data',
            'reset_error': 'Error resetting systems',
            'uptime': 'Uptime',
            'detections': 'Detections',
            'components': 'Components',
            'quick_actions': 'Quick Actions',
            'refresh_system': 'Refresh System',
            'refresh_system_help': 'Clear cache and refresh components',
            'export_session': 'Export Session',
            'export_session_help': 'Quick export of current session data',
            'system_refreshed': 'System refreshed successfully',
            'refresh_failed': 'System refresh failed',
            'download_quick_export': 'Download Quick Export',
            'quick_export_failed': 'Quick export failed',
            
            # Model Loading and Errors
            'model_loading_error': 'Model loading error',
            'model_not_found': 'Model file not found',
            'loading_model': 'Loading model',
            'model_loaded': 'Model loaded successfully',
            'standard_model_error': 'Standard model initialization error',
            'model_initialization_failed': 'Model initialization failed',
            'requirements_not_met': 'System requirements not met',
            'install_missing_components_to_continue': 'Please install missing components to continue',
            'missing_requirements': 'Missing Requirements',
            'missing_components': 'Missing Components',
            'processing_device': 'Processing Device',
            'no_model_loaded': 'No model loaded',
            'processing_error': 'Processing error',
            
            # Performance and Statistics
            'frames_processed': 'Frames Processed',
            'processing_time': 'Processing Time',
            'avg_processing_fps': 'Avg Processing FPS',
            'parking_events': 'Parking Events',
            'counting_events': 'Counting Events',
            'session_duration': 'Session Duration',
            'minutes': 'minutes',
            'fps': 'FPS',
            'plates_detected': 'plates detected',
            'objects_detected': 'objects detected',
            'plates_read': 'Plates Read',
            'tracked_vehicles': 'Tracked Vehicles',
            'vehicle': 'Vehicle',
            'plates': 'Plates',
            'parking': 'Parking',
            'no_vehicles_currently_tracked': 'No vehicles currently tracked',
            'free_spots': 'Free Spots',
            'occupied_spots': 'Occupied Spots',
            
            # Help and Documentation
            'help_documentation': 'Help & Documentation',
            'welcome_to_vision_platform': 'Welcome to YOLO11 Vision Platform',
            'key_features': 'Key Features',
            'enhanced_license_plate_recognition': 'Enhanced License Plate Recognition',
            'professional_grade_ocr': 'Professional-grade OCR with real-time tracking',
            'real_time_analytics': 'Real-time Analytics',
            'live_performance_monitoring': 'Live performance monitoring and insights',
            'smart_parking_management': 'Smart parking space monitoring',
            'motion_detection_and_alerts': 'Motion detection and zone-based alerts',
            'comprehensive_results_management': 'Comprehensive Results Management',
            'advanced_file_analysis': 'Advanced file analysis and export capabilities',
            'getting_started': 'Getting Started',
            'choose_your_model': 'Choose Your Model',
            'select_detection_type': 'Select the type of detection/analysis you need',
            'select_input_source': 'Select Input Source',
            'choose_images_videos_cameras': 'Choose between images, videos, or live cameras',
            'configure_settings': 'Configure Settings',
            'adjust_confidence_parameters': 'Adjust confidence and other parameters',
            'start_processing': 'Start Processing',
            'begin_analysis': 'Begin your analysis session',
            'monitor_export': 'Monitor & Export',
            'track_performance_export_results': 'Track performance and export results',
            'live_camera_processing': 'Live Camera Processing',
            'connect_camera_instructions': 'Connect to RTSP cameras for real-time analysis',
            'start_processing_instructions': 'Start processing to see live results',
            'monitor_live_metrics': 'Monitor live metrics and performance',
            'export_session_data_instructions': 'Export session data and analytics',
            'enhanced_features': 'Enhanced Features',
            'smart_cropping': 'Smart Cropping',
            'intelligent_roi_detection': 'Intelligent region of interest detection',
            'gpu_acceleration': 'GPU Acceleration',
            'hardware_acceleration': 'Hardware acceleration for optimal performance',
            'professional_visualization': 'Professional Visualization',
            'custom_borders_overlays': 'Custom borders and overlay displays',
            'auto_save': 'Auto-save',
            'automatic_result_preservation': 'Automatic result preservation',
            'multi_language': 'Multi-language',
            'english_french_support': 'English and French interface support',
            'troubleshooting': 'Troubleshooting',
            'connection_issues': 'Connection Issues',
            'check_camera_credentials': 'Check camera IP addresses and credentials',
            'performance_problems': 'Performance Problems',
            'enable_gpu_boost': 'Enable GPU acceleration for better performance',
            'missing_components': 'Missing Components',
            'install_dependencies': 'Install required Python dependencies',
            'memory_issues': 'Memory Issues',
            'use_refresh_system_button': 'Use the "Refresh System" button to clear cache',
            'analytics_reporting': 'Analytics & Reporting',
            'tab_2_analytics': 'Tab 2 - Analytics',
            'real_time_interactive_charts': 'Real-time interactive charts and metrics',
            'tab_3_results': 'Tab 3 - Results',
            'file_management_analysis': 'File management and result analysis',
            'tab_4_reports': 'Tab 4 - Reports',
            'professional_reporting': 'Professional reporting and export options',
            'pro_tips': 'Pro Tips',
            'use_enhanced_license_plate_for_maximum_capabilities': 'Use Enhanced License Plate mode for maximum capabilities',
            'enable_auto_save_for_important_sessions': 'Enable auto-save for important processing sessions',
            'monitor_quality_score_for_optimal_performance': 'Monitor confidence scores for optimal performance',
            'export_analytics_for_external_analysis': 'Export analytics for external analysis tools',
            'system_requirements': 'System Requirements',
            'with_required_dependencies': 'with required dependencies installed',
            'gpu_recommended': 'GPU Recommended',
            'for_optimal_performance': 'for optimal performance (CUDA support)',
            'network_access': 'Network Access',
            'for_rtsp_connections': 'for RTSP camera connections',
            'storage_space': 'Storage Space',
            'for_results_exports': 'for result exports and video processing',
            'for_additional_support_use_debug_panel': 'For additional support, use the debug panel in System Management tab',
            
            # Error Handling
            'application_error': 'Application Error',
            'error_details': 'Error Details',
            'recovery_suggestions': 'Recovery Suggestions',
            'refresh_page_to_restart': 'Refresh the page to restart the application',
            'check_system_requirements': 'Check that all system requirements are met',
            'verify_model_files_exist': 'Verify that model files exist in weight_models directory',
            'ensure_dependencies_installed': 'Ensure all Python dependencies are installed',
            'emergency_reset': 'Emergency Reset',
            'emergency_reset_complete': 'Emergency reset completed successfully',
            'page_will_refresh_automatically': 'Page will refresh automatically',
            'emergency_reset_failed': 'Emergency reset failed',
            'application_startup_error': 'Application startup error',
            'emergency_mode': 'Emergency Mode',
            'application_encountered_startup_error': 'The application encountered a startup error',
            'troubleshooting_steps': 'Troubleshooting Steps',
            'refresh_browser_page': 'Refresh your browser page',
            'check_all_dependencies_installed': 'Check that all dependencies are installed',
            'verify_model_files_in_weight_models_directory': 'Verify model files are in weight_models directory',
            'ensure_required_modules_available': 'Ensure required modules are available',
            'system_check': 'System Check',
            
            # Session and Auto-save
            'session_auto_saved': 'Session auto-saved',
            'timeline_entries': 'Timeline Entries',
            'license_plate_utils': 'License Plate Utils',
            'sort_tracker': 'SORT Tracker',
            'post_processing': 'Post Processing',
            'and': 'and',
            'more': 'more',
            'end_of_report': 'End of Report',
            'vision_platform_report': 'YOLO11 Vision Platform Report',
            'session_information': 'Session Information',
            'processing_statistics': 'Processing Statistics',
            'detected_license_plates': 'Detected License Plates',
            'device': 'Device',
            'cameras_available': 'Cameras Available',
            'session_stats': 'Session Stats',
            'unknown': 'Unknown',
            
            # General UI
            'enabled': 'Enabled',
            'available': 'Available',
            'mode': 'Mode',
            'settings': 'Settings',
            'status': 'Status',
            'loading': 'Loading',
            'processing': 'Processing',
            'analyzing': 'Analyzing',
            'saving': 'Saving',
            'exporting': 'Exporting',
            'downloading': 'Downloading',
            'uploading': 'Uploading',
            'error': 'Error',
            'warning': 'Warning',
            
            # Analytics and Results
            'analytics_and_results': 'Analytics & Results',
            'analytics_overview': 'Analytics Overview',
            'results_summary': 'Results Summary',
            'analytics_and_results_description': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_2': 'Export analytics and results for external analysis',
            'analytics_and_results_description_3': 'Track performance and export results',
            'analytics_and_results_description_4': 'Monitor live metrics and performance',
            'analytics_and_results_description_5': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_6': 'Export analytics and results for external analysis',
            'analytics_and_results_description_7': 'Track performance and export results',
            'analytics_and_results_description_8': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_9': 'Export analytics and results for external analysis',
            'analytics_and_results_description_10': 'Track performance and export results',
            'analytics_and_results_description_11': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_12': 'Export analytics and results for external analysis',
            'analytics_and_results_description_13': 'Track performance and export results',
            'analytics_and_results_description_14': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_15': 'Export analytics and results for external analysis',
            'analytics_and_results_description_16': 'Track performance and export results',
            'analytics_and_results_description_17': 'Analyze your data with detailed analytics and export results for further analysis',
            'analytics_and_results_description_18': 'Export analytics and results for external analysis',
            'analytics_and_results_description_19': 'Track performance and export results',
            'stair_shapes': 'Stair Shapes (CountingStairPos copy)',
            'upload_parking_positions': 'No parking positions file found - upload CarParkPos',
            'real_time': 'Real-time',

            # New keys from camera interface
            'opencv_live_processing_window_header': 'OpenCV Live Processing Window',
            'start_live_processing_cv_button': 'Start Live Processing',
            'stop_processing_cv_button': 'Stop Processing',
            'no_active_processing_msg': 'No active processing.',
            'click_start_live_processing_to_begin_msg': "Click 'Start Live Processing' to begin.",
            'instructions_header': 'Instructions',
            'instruction_item_1_cv': 'Click "Start Live Processing" to open OpenCV window',
            'instruction_item_2_cv': 'The video will appear in a separate window',
            'instruction_item_3_cv': 'Use keyboard controls in the OpenCV window:',
            'instruction_item_3a_cv_q': 'Q: Quit processing',
            'instruction_item_3b_cv_s': 'S: Show statistics in console',
            'instruction_item_3c_cv_r': 'R: Reset tracking data',
            'instruction_item_4_cv': 'Return to this Streamlit interface for analytics and export',
            'active_colon': 'Active:',
            'task_colon': 'Task:',
            'refresh_analytics_button': 'Refresh Analytics',
            'counting_&_alert_system': 'Counting & Alert System',
            'model_option_detection': 'Detection',
            'model_option_segmentation': 'Segmentation',
            'model_option_pose_estimation': 'Pose Estimation',
            'model_option_elp': 'Enhanced License Plate',
            'model_option_parking': 'Parking Detection',
            'model_option_counting': 'Counting & Alert System',
        }

    def _get_french_translations(self) -> Dict[str, str]:
        """Complete French translations (placeholder)"""
        # TODO: Add actual French translations here
        return {
            # Application Title and General
            'app_title': 'Moteur d\'Analyse Visuelle',
            'app_subtitle': 'Système avancé de vision par ordinateur & analyse',
            'professional_computer_vision_system': 'Système professionnel de vision et d\'analyse par ordinateur',
            'vision_platform_title': 'Plateforme Visuelle YOLO11',

            # Navigation and Interface
            'model_config': 'Configuration du Modèle',
            'source_selection': 'Sélection de la Source',
            'camera_settings': 'Paramètres de la Caméra',
            'processing_options': 'Options de Traitement',
            'system_management': 'Gestion du Système',

            # Task Selection
            'task_selection': 'Sélectionner une Tâche d\'Analyse',
            'confidence': 'Niveau de Confiance',
            'choose_source': 'Choisir la Source d\'Entrée',
            'source_selection_help': 'Sélectionnez votre source d\'entrée : Image, Vidéo ou Caméra en direct',

            # Model Types
            'enhanced_license_plate': 'Détection Améliorée de Plaques d\'Immatriculation',
            'parking_detection': 'Détection de Stationnement',
            'counting_alert_system': 'Système de Comptage et d\'Alerte',
            'current_model': 'Modèle Actuel',

            # Enhanced License Plate
            'professional_tracking': 'Suivi professionnel des véhicules avec bordures de coin personnalisées',
            'advanced_ocr': 'OCR avancée avec validation de format et correction de caractères',
            'real_time_visualization': 'Visualisation professionnelle en temps réel avec affichages superposés',
            'comprehensive_analytics': 'Analytique complète avec persistance de session',
            'professional_visualization': 'Visualisation professionnelle avec bordures personnalisées',
            'real_time_ocr': 'OCR en temps réel avec validation de format',
            'advanced_tracking': 'Suivi avancé et interpolation',
            'enhanced_license_plate_not_available': 'Détection améliorée de plaques d\'immatriculation non disponible',
            'loading_enhanced_license_plate': 'Chargement du Système de Détection Améliorée de Plaques...',
            'failed_to_initialize_enhanced_system': 'Échec de l\'initialisation du système de détection améliorée',
            'enhanced_system_loaded': 'Système de Détection Améliorée de Plaques Chargé',
            'license_plate_init_error': 'Erreur d\'initialisation du détecteur de plaques',

            # Parking Detection
            'task': 'Tâche',
            'features': 'Fonctionnalités',
            'parking_occupancy': 'Surveillance de l\'Occupation des Places de Stationnement',
            'parking_features': 'Places rectangulaires et polygonales, Comptage en temps réel',
            'pixel_threshold': 'Seuil de Pixels pour Stationnement',
            'pixel_threshold_help': 'Valeurs basses = plus susceptible d\'être "Libre". Ajuster selon la taille de la place.',
            'upload_parking_positions': 'Aucun fichier de positions de stationnement trouvé - téléchargez CarParkPos',
            'preprocessing_error': 'Erreur de Prétraitement',
            'parking_error': 'Erreur de Stationnement',
            'no_parking_spots': 'Aucune place de stationnement chargée',
            'no_parking_positions': 'Aucune position de stationnement chargée',
            'parking_occupancy_mode': 'Mode de surveillance d\'occupation du stationnement',
            'parking_detection_not_available': 'Détection de stationnement non disponible',
            'parking_system_loaded': 'Système de Stationnement Chargé',
            'spaces': 'places',
            'no_parking_positions_loaded': 'Aucune position de stationnement chargée - veuillez téléverser la configuration',
            'parking_init_error': 'Erreur d\'initialisation du système de stationnement',
            'yolo_detections': 'Détections YOLO',
            'detection_visualization': 'Visualisation de Détection',
            'show_yolo_detections': 'Afficher les Détections de Véhicules YOLO', 
            'yolo_visualization_help': 'Afficher les boîtes englobantes YOLO et les scores de confiance pour tous les objets détectés',
            'yolo_box_color': 'Couleur des Boîtes YOLO',
            'parking_spot_colors': 'Couleurs des Places de Stationnement',
            'free_spots_color': 'Places Libres: Vert',
            'occupied_spots_color': 'Places Occupées: Rouge',
            'recent_events': 'Événements Récents',
            'detection_settings': 'Paramètres de Détection',
            'yolo_visualization': 'Visualisation YOLO',
            'pixel_threshold': 'Seuil de Pixels',
            'target_classes': 'Classes Cibles',
            'detection_confidence': 'Confiance de Détection',


            # Counting & Alert System
            'motion_detection': 'Détection de Mouvement et Franchissement de Ligne',
            'counting_features': 'Zones polygonales/rectangulaires, Lignes de comptage, Alertes',
            'upload_shapes_file': 'Téléverser le Fichier de Configuration des Formes',
            'counting_error': 'Erreur du Système de Comptage',
            'no_shapes': 'Aucune configuration de formes chargée',
            'counting_alert_mode': 'Mode Système de Comptage et d\'Alerte',
            'zone_shapes': 'Formes de Zone (CountingZonePos copy)',
            'hall_shapes': 'Formes de Hall (CountingHallPos copy 2)',
            'stair_shapes': 'Formes d\'Escalier (CountingStairPos copy)',
            'json_config': 'Configuration JSON (shapes_config.json)',
            'select_shape_config': 'Sélectionner la Configuration des Formes',
            'shape_config_help': 'Choisir quel ensemble prédéfini de formes utiliser',
            'load_shapes': 'Charger les Formes',
            'loaded': 'Chargé',
            'shapes': 'formes',
            'failed_to_load_shapes': 'Échec du chargement des formes',
            'counting_system_not_available': 'Système de comptage non disponible',
            'counting_system_loaded': 'Système de Comptage Chargé',
            'no_shapes_loaded': 'Aucune forme chargée - veuillez configurer les formes',
            'counting_init_error': 'Erreur d\'initialisation du système de comptage',

            # Camera Controls
            'select_camera': 'Sélectionner une Caméra',
            'camera_selection_help': 'Choisir la caméra pour le traitement en temps réel',
            'connect_camera': 'Connecter la Caméra',
            'disconnect_camera': 'Déconnecter la Caméra',
            'camera_connected': 'Caméra Connectée',
            'camera_disconnected': 'Caméra Déconnectée',
            'camera_error': 'Erreur de Caméra',
            'camera_connection_failed': 'Échec de la connexion à la caméra',
            'camera_not_available': 'Caméra non disponible',
            'connection_error': 'Erreur de connexion',
            'disconnection_error': 'Erreur de déconnexion',
            'camera_not_connected': 'Caméra non connectée',
            'connecting': 'Connexion en cours',
            'selected': 'Sélectionné',

            # Advanced Camera Settings
            'advanced_camera_settings': 'Paramètres Avancés de la Caméra',
            'connection_mode': 'Mode de Connexion',
            'connection_mode_help': 'Mode d\'optimisation de la connexion RTSP',
            'stream_quality': 'Qualité du Flux',
            'stream_quality_help': 'Sélection de la qualité du flux vidéo',
            'buffer_size': 'Taille du Tampon',
            'buffer_size_help': 'Taille du tampon de trames pour une lecture fluide',
            'auto_reconnect': 'Reconnexion Automatique',
            'auto_reconnect_help': 'Se reconnecter automatiquement en cas de déconnexion',

            # Processing Controls
            'start_processing': 'Démarrer le Traitement',
            'stop_processing': 'Arrêter le Traitement',
            'reset_tracking': 'Réinitialiser le Suivi',
            'save_results': 'Enregistrer les Résultats',
            'processing_started': 'Traitement Démarré',
            'processing_stopped': 'Traitement Arrêté',
            'tracking_reset': 'Données de suivi réinitialisées avec succès',
            'results_saved': 'Résultats enregistrés avec succès',
            'no_results_to_save': 'Aucun résultat à enregistrer',
            'save_error': 'Erreur lors de l\'enregistrement des résultats',
            'reset_error': 'Erreur lors de la réinitialisation du suivi',

            # Live Processing
            'start_live': 'Démarrer en Direct',
            'stop_live': 'Arrêter le Direct',
            'live_processing_started': 'Traitement en direct démarré',
            'live_processing_stopped': 'Traitement en direct arrêté',
            'live_processing_active': 'Traitement en Direct Actif',
            'live_processing_error': 'Erreur de traitement en direct',
            'no_frame_available': 'Aucune trame disponible',
            'connected_ready_for_processing': 'Connecté - Prêt pour le Traitement',
            'ready': 'Prêt',
            'click_start_processing': 'Cliquez sur Démarrer pour Commencer le Traitement',

            # Source Types
            'choose_file': 'Choisir un Fichier',
            'upload_image': 'Télécharger une Image',
            'upload_image_help': 'Télécharger une image pour analyse IA',
            'upload_image_prompt': 'Téléchargez une image pour démarrer l\'analyse',
            'upload_image_to_analyze': 'Téléchargez une image pour voir les résultats de l\'analyse',
            'choose_video': 'Choisir une Vidéo',
            'choose_video_help': 'Sélectionner un fichier vidéo pour traitement',
            'video_not_found': 'Fichier vidéo non trouvé',
            'no_camera_connected': 'Aucune Caméra Connectée',
            'connect_camera_sidebar': 'Utilisez la barre latérale pour vous connecter à une caméra',
            'select_camera_to_start': 'Sélectionnez et connectez une caméra pour démarrer',

            # Processing Interface
            'detection_interface': 'Interface de Détection',
            'image_analysis': 'Analyse d\'Image',
            'video_analysis': 'Analyse Vidéo',
            'live_camera_analysis': 'Analyse Caméra en Direct',
            'original_image': 'Image Originale',
            'processed_result': 'Résultat Traité',
            'processed_frame': 'Image Traitée',
            'default_image': 'Image par Défaut',
            'analyze_image': 'Analyser l\'Image',
            'process_video': 'Traiter la Vidéo',
            'processing_with': 'Traitement avec',
            'image_processing_failed': 'Échec du traitement de l\'image',
            'image_processing_error': 'Erreur de traitement d\'image',

            # Video Processing
            'processing_video': 'Traitement de la vidéo',
            'frames': 'images',
            'video_open_error': 'Impossible d\'ouvrir le fichier vidéo',
            'video_processing_complete': 'Traitement vidéo terminé',
            'video_processing_error': 'Erreur de traitement vidéo',
            'video_saved': 'Vidéo enregistrée',
            'save_processed_video': 'Enregistrer la Vidéo Traitée',
            'processing': 'Traitement en cours',
            'speed': 'Vitesse',
            'completed': 'Terminé',

            # Live Camera
            'real_time_feed': 'Flux en Temps Réel',
            'live_feed': 'Flux en Direct',
            'camera': 'Caméra',
            'available_cameras': 'Caméras Disponibles',
            'live_controls': 'Contrôles en Direct',
            'live_statistics': 'Statistiques en Direct',
            'live_license_plate_info': 'Infos Plaques en Direct',
            'live_parking_info': 'Infos Stationnement en Direct',
            'live_counting_info': 'Infos Comptage en Direct',

            # Tabs
            'enhanced_live_detection': 'Détection Améliorée en Direct',
            'real_time_analytics': 'Analytique en Temps Réel',
            'license_plate_results': 'Résultats Plaques d\'Immatriculation',
            'reports_export': 'Rapports et Exportation',
            'live_detection': 'Détection en Direct',
            'analytics_dashboard': 'Tableau de Bord Analytique',
            'detection_results': 'Résultats de Détection',

            # Analytics
            'start_processing_for_analytics': 'Démarrer le traitement pour voir l\'analytique en temps réel',
            'total_detections': 'Nombre Total de Détections',
            'unique_plates': 'Plaques Uniques',
            'avg_confidence': 'Confiance Moyenne',
            'detections_per_minute': 'Détections/Minute',
            'detection_timeline': 'Chronologie des Détections',
            'confidence_analysis': 'Analyse de Confiance',
            'plate_discovery': 'Découverte de Plaques',
            'performance_trends': 'Tendances de Performance',
            'detection_timeline_chart': 'Chronologie des Détections de Plaques en Temps Réel',
            'confidence_score': 'Score de Confiance',
            'detection_time': 'Heure de Détection',
            'excellent_threshold': 'Seuil Excellent',
            'good_threshold': 'Seuil Bon',
            'excellent': 'Excellent',
            'good': 'Bon',
            'fair': 'Passable',
            'need_more_detections_for_timeline': 'Nécessite au moins 2 détections pour la chronologie',
            'confidence_distribution': 'Distribution des Scores de Confiance',
            'frequency': 'Fréquence',
            'quality_distribution': 'Distribution de la Qualité des Détections',
            'no_confidence_data_available': 'Aucune donnée de confiance disponible',
            'unique_plates_discovery': 'Découverte de Plaques Uniques au Fil du Temps',
            'cumulative_unique_plates': 'Plaques Uniques Cumulées',
            'time': 'Heure',
            'discovered_plates': 'Plaques Découvertes',
            'discovery_metrics': 'Métriques de Découverte',
            'discovery_rate': 'Taux de Découverte',
            'total_unique': 'Total Uniques',
            'no_plate_discovery_data': 'Aucune donnée de découverte de plaques disponible',
            'performance_trend_analysis': 'Analyse des Tendances de Performance',
            'individual_detections': 'Détections Individuelles',
            'rolling_average': 'Moyenne Mobile',
            'need_more_data_for_trends': 'Nécessite au moins 10 détections pour l\'analyse des tendances',

            # Parking Analytics
            'parking_system_not_initialized': 'Système de stationnement non initialisé',
            'total_spaces': 'Nombre Total de Places',
            'free_spaces': 'Places Libres',
            'occupied_spaces': 'Places Occupées',
            'occupancy_rate': 'Taux d\'Occupation',
            'parking_events_analysis': 'Analyse des Événements de Stationnement',
            'parking_entries_by_hour': 'Entrées au Stationnement par Heure',
            'hour_of_day': 'Heure de la Journée',
            'number_of_entries': 'Nombre d\'Entrées',
            'avg_parking_duration': 'Durée Moyenne de Stationnement',
            'max_parking_duration': 'Durée Maximale de Stationnement',
            'no_parking_events_recorded': 'Aucun événement de stationnement enregistré',
            'parking_overview': 'Aperçu du Stationnement',
            'parking_spaces': 'Places de Parking',
            'spot_status': 'État des Places',
            'parking_report': 'Rapport de Stationnement',

            # Counting Analytics
            'counting_system_not_initialized': 'Système de comptage non initialisé',
            'total_zones': 'Nombre de Zones',
            'active_alerts': 'Alertes Actives',
            'total_crossings': 'Nombre Total de Passages',
            'zone_activity_analysis': 'Analyse d\'Activité par Zone',
            'zone_name': 'Nom de la Zone',
            'crossings': 'Passages',
            'crossings_per_zone': 'Passages par Zone',
            'live_events_timeline': 'Chronologie des Événements en Direct',
            'events_timeline': 'Chronologie des Événements',
            'counting_results': 'Résultats du Comptage',
            'alert_count': 'Nombre d\'Alertes',
            'crossing_count': 'Nombre de Passages',
            'zone_activity': 'Activité par Zone',
            'alerts': 'Alertes',
            'zone_details': 'Détails par Zone',
            
            # Results Management
            'results_management': 'Gestion des Résultats',
            'available_result_files': 'Fichiers de Résultats Disponibles',
            'select_result_file': 'Sélectionner un fichier de résultats à analyser',
            'file_info': 'Informations sur le Fichier',
            'filename': 'Nom du Fichier',
            'size': 'Taille',
            'modified': 'Modifié le',
            'estimated_rows': 'Lignes Estimées',
            'analyze_selected_file': 'Analyser le Fichier Sélectionné',
            'no_result_files_found': 'Aucun fichier de résultats trouvé',
            'go_to_detection_tab_video': 'Aller à l\'onglet Détection → Vidéo pour traiter les vidéos',
            'go_to_detection_tab_camera': 'Aller à l\'onglet Détection → Caméra pour les sessions en direct',
            'upload_results_file': 'Téléverser un Fichier de Résultats',
            'file_analysis': 'Analyse de Fichier',
            'total_records': 'Total Enregistrements',
            'columns': 'Colonnes',
            'license_plates': 'Plaques d\'Immatriculation',
            'data_preview': 'Aperçu des Données',
            'download_csv': 'Télécharger CSV',
            'file_analysis_error': 'Erreur d\'analyse de fichier',
            'file_uploaded_and_saved': 'Fichier téléversé et enregistré',
            'refresh_page_to_see_file': 'Actualisez la page pour voir le fichier dans la liste',
            'file_upload_error': 'Erreur de téléversement de fichier',

            # Detection Results
            'detection_results': 'Résultats de Détection',
            'vehicle_id': 'ID Véhicule',
            'license_plate': 'Plaque d\'Immatriculation',
            'detected': 'Détecté',
            'parking_status': 'État du Stationnement',
            'counting_alert_status': 'État Comptage & Alertes',
            'objects': 'objets',
            'detection_details': 'Détails de Détection',
            'no_objects_detected': 'Aucun objet détecté',

            # Reports and Export
            'session_export': 'Exportation de Session',
            'analytics_export': 'Exportation d\'Analytique',
            'generate_report': 'Générer un Rapport',
            'export_session_data': 'Exporter les Données de Session',
            'export_analytics': 'Exporter l\'Analytique',
            'generate_summary_report': 'Générer un Rapport Synthétique',
            'license_plate_specific_exports': 'Exports Spécifiques aux Plaques',
            'export_detection_results': 'Exporter les Résultats de Détection',
            'export_performance_report': 'Exporter le Rapport de Performance',
            'parking_specific_exports': 'Exports Spécifiques au Stationnement',
            'export_parking_report': 'Exporter le Rapport de Stationnement',
            'counting_specific_exports': 'Exports Spécifiques au Comptage',
            'export_counting_log': 'Exporter le Journal de Comptage',
            'download_session_json': 'Télécharger JSON de Session',
            'download_analytics_json': 'Télécharger JSON d\'Analytique',
            'download_report': 'Télécharger le Rapport',
            'download_license_plate_csv': 'Télécharger CSV Plaques',
            'download_performance_json': 'Télécharger JSON Performance',
            'download_parking_csv': 'Télécharger CSV Stationnement',
            'download_counting_csv': 'Télécharger CSV Comptage',
            'session_data_exported': 'Données de session exportées',
            'analytics_data_exported': 'Données analytiques exportées',
            'report_generated': 'Rapport généré',
            'license_plate_results_exported': 'Résultats plaques exportés',
            'performance_report_exported': 'Rapport de performance exporté',
            'parking_report_exported': 'Rapport de stationnement exporté',
            'counting_log_exported': 'Journal de comptage exporté',
            'export_error': 'Erreur d\'exportation',
            'report_generation_error': 'Erreur de génération de rapport',
            'license_processor_not_available': 'Processeur de plaques non disponible',
            'no_results_to_export': 'Aucun résultat à exporter',
            'download_preparation_failed': 'Échec de préparation du téléchargement',
            'no_performance_data': 'Aucune donnée de performance',
            'parking_detector_not_available': 'Détecteur de stationnement non disponible',
            'no_parking_events_to_export': 'Aucun événement de stationnement à exporter',
            'counting_system_not_available': 'Système de comptage non disponible',
            'no_counting_events_to_export': 'Aucun événement de comptage à exporter',

            # System Management
            'system_status': 'État du Système',
            'system_information_status': 'Informations et État du Système',
            'memory_usage': 'Utilisation Mémoire',
            'maintenance': 'Maintenance',
            'clear_analytics_data': 'Effacer les Données Analytiques',
            'reset_all_systems': 'Réinitialiser Tous les Systèmes',
            'debug_information': 'Informations de Débogage',
            'session_variables': 'Variables de Session',
            'analytics_entries': 'Entrées Analytiques',
            'analytics_data_cleared': 'Données analytiques effacées',
            'all_systems_reset': 'Tous systèmes réinitialisés',
            'clear_data_error': 'Erreur d\'effacement des données',
            'uptime': 'Temps de Fonctionnement',
            'detections': 'Détections',
            'components': 'Composants',
            'quick_actions': 'Actions Rapides',
            'refresh_system': 'Actualiser le Système',
            'refresh_system_help': 'Vider le cache et actualiser les composants',
            'export_session': 'Exporter la Session',
            'export_session_help': 'Exportation rapide des données de session actuelles',
            'system_refreshed': 'Système actualisé',
            'refresh_failed': 'Échec de l\'actualisation',
            'download_quick_export': 'Télécharger Export Rapide',
            'quick_export_failed': 'Échec de l\'export rapide',

            # Model Loading and Errors
            'model_loading_error': 'Erreur de Chargement du Modèle',
            'model_not_found': 'Fichier modèle non trouvé',
            'loading_model': 'Chargement du modèle',
            'model_loaded': 'Modèle chargé avec succès',
            'standard_model_error': 'Erreur d\'initialisation du modèle standard',
            'model_initialization_failed': 'Échec de l\'initialisation du modèle',
            'requirements_not_met': 'Prérequis système non satisfaits',
            'install_missing_components_to_continue': 'Veuillez installer les composants manquants pour continuer',
            'missing_requirements': 'Prérequis Manquants',
            'missing_components': 'Composants Manquants',
            'processing_device': 'Périphérique de Traitement',
            'no_model_loaded': 'Aucun modèle chargé',
            'processing_error': 'Erreur de Traitement',

            # Performance and Statistics
            'frames_processed': 'Images Traitées',
            'processing_time': 'Temps de Traitement',
            'avg_processing_fps': 'FPS Moyen de Traitement',
            'parking_events': 'Événements de Stationnement',
            'counting_events': 'Événements de Comptage',
            'session_duration': 'Durée de la Session',
            'minutes': 'minutes',
            'fps': 'FPS',
            'plates_detected': 'plaques détectées',
            'objects_detected': 'objets détectés',
            'plates_read': 'Plaques Lues',
            'tracked_vehicles': 'Véhicules Suivis',
            'vehicle': 'Véhicule',
            'plates': 'Plaques',
            'parking': 'Stationnement',
            'no_vehicles_currently_tracked': 'Aucun véhicule suivi actuellement',
            'free_spots': 'Places Libres',
            'occupied_spots': 'Places Occupées',

            # New keys for counting analytics
            'counting_system_not_initialized_nor_logs_found': 'Système de comptage non initialisé et aucun journal historique trouvé.',
            'historical_counting_logs_analysis': 'Analyse des Journaux de Comptage Historiques',
            'no_counting_log_files_found_pattern': 'Aucun fichier journal de comptage (ex: *counting*_crossings.csv) trouvé dans le répertoire actuel.',
            'select_counting_log_file': 'Sélectionner un Fichier Journal de Comptage à Analyser',
            'load_and_analyze_counting_log': 'Charger et Analyser le Journal Sélectionné',
            'analysis_of_selected_log': 'Analyse du Journal Sélectionné',
            'total_crossings_in_log': 'Total des Passages (dans le journal)',
            'from_log': 'du journal',
            'events_timeline_by_object': 'Chronologie des Événements par Type d\'Objet',
            'hourly': 'par heure',
            'could_not_parse_timestamps_for_timeline_chart': 'Impossible d\'analyser les horodatages pour le graphique chronologique',
            'download_analyzed_log_csv': 'Télécharger le Journal Analysé (CSV)',
            'live_counting_status': '🔴 État du Comptage en Direct',
            'live_crossings_per_zone': 'Passages par Zone en Direct',
            'live_counting_system_not_active': 'ℹ️ Le système de comptage en direct n\'est pas actif ou initialisé.',
            'auto_analysis_of_most_recent_log': 'Auto-Analyse du Journal le Plus Récent',
            'manual_log_selection_and_analysis': 'Sélection et Analyse Manuelles du Journal',
            'time_hour': 'Heure',

            # Help and Documentation
            'help_documentation': 'Aide & Documentation',
            'welcome_to_vision_platform': 'Bienvenue sur la Plateforme Visuelle YOLO11',
            'key_features': 'Key Features',
            'enhanced_license_plate_recognition': 'Enhanced License Plate Recognition',
            'professional_grade_ocr': 'Professional-grade OCR with real-time tracking',
            'real_time_analytics': 'Real-time Analytics',
            'live_performance_monitoring': 'Live performance monitoring and insights',
            'smart_parking_management': 'Smart parking space monitoring',
            'motion_detection_and_alerts': 'Motion detection and zone-based alerts',
            'comprehensive_results_management': 'Comprehensive Results Management',
            'advanced_file_analysis': 'Advanced file analysis and export capabilities',
            'getting_started': 'Getting Started',
            'choose_your_model': 'Choose Your Model',
            'select_detection_type': 'Select the type of detection/analysis you need',
            'select_input_source': 'Select Input Source',
            'choose_images_videos_cameras': 'Choose between images, videos, or live cameras',
            'configure_settings': 'Configure Settings',
            'adjust_confidence_parameters': 'Adjust confidence and other parameters',
            'start_processing': 'Start Processing',
            'begin_analysis': 'Begin your analysis session',
            'monitor_export': 'Monitor & Export',
            'track_performance_export_results': 'Track performance and export results',
            'live_camera_processing': 'Live Camera Processing',
            'connect_camera_instructions': 'Connect to RTSP cameras for real-time analysis',
            'start_processing_instructions': 'Start processing to see live results',
            'monitor_live_metrics': 'Monitor live metrics and performance',
            'export_session_data_instructions': 'Export session data and analytics',
            'enhanced_features': 'Enhanced Features',
            'smart_cropping': 'Smart Cropping',
            'intelligent_roi_detection': 'Intelligent region of interest detection',
            'gpu_acceleration': 'GPU Acceleration',
            'hardware_acceleration': 'Hardware acceleration for optimal performance',
            'professional_visualization': 'Professional Visualization',
            'custom_borders_overlays': 'Custom borders and overlay displays',
            'auto_save': 'Auto-save',
            'automatic_result_preservation': 'Automatic result preservation',
            'multi_language': 'Multi-language',
            'english_french_support': 'English and French interface support',
            'troubleshooting': 'Troubleshooting',
            'connection_issues': 'Connection Issues',
            'check_camera_credentials': 'Check camera IP addresses and credentials',
            'performance_problems': 'Performance Problems',
            'enable_gpu_boost': 'Enable GPU acceleration for better performance',
            'missing_components': 'Missing Components',
            'install_dependencies': 'Install required Python dependencies',
            'memory_issues': 'Memory Issues',
            'use_refresh_system_button': 'Use the "Refresh System" button to clear cache',
            'analytics_reporting': 'Analytics & Reporting',
            'tab_2_analytics': 'Tab 2 - Analytics',
            'real_time_interactive_charts': 'Real-time interactive charts and metrics',
            'tab_3_results': 'Tab 3 - Results',
            'file_management_analysis': 'File management and result analysis',
            'tab_4_reports': 'Tab 4 - Reports',
            'professional_reporting': 'Professional reporting and export options',
            'pro_tips': 'Pro Tips',
            'use_enhanced_license_plate_for_maximum_capabilities': 'Use Enhanced License Plate mode for maximum capabilities',
            'enable_auto_save_for_important_sessions': 'Enable auto-save for important processing sessions',
            'monitor_quality_score_for_optimal_performance': 'Monitor confidence scores for optimal performance',
            'export_analytics_for_external_analysis': 'Export analytics for external analysis tools',
            'system_requirements': 'System Requirements',
            'with_required_dependencies': 'with required dependencies installed',
            'gpu_recommended': 'GPU Recommended',
            'for_optimal_performance': 'for optimal performance (CUDA support)',
            'network_access': 'Network Access',
            'for_rtsp_connections': 'for RTSP camera connections',
            'storage_space': 'Storage Space',
            'for_results_exports': 'for result exports and video processing',
            'for_additional_support_use_debug_panel': 'For additional support, use the debug panel in System Management tab',
            
            # Error Handling
            'application_error': 'Erreur d\'Application',
            'error_details': 'Détails de l\'Erreur',
            'recovery_suggestions': 'Suggestions de Récupération',
            'refresh_page_to_restart': 'Actualisez la page pour redémarrer l\'application',
            'check_system_requirements': 'Vérifiez que tous les prérequis système sont satisfaits',
            'verify_model_files_exist': 'Vérifiez que les fichiers modèles existent dans le dossier weight_models',
            'ensure_dependencies_installed': 'Assurez-vous que toutes les dépendances Python sont installées',
            'emergency_reset': 'Réinitialisation d\'Urgence',
            'emergency_reset_complete': 'Réinitialisation d\'urgence terminée avec succès',
            'page_will_refresh_automatically': 'La page va s\'actualiser automatiquement',
            'emergency_reset_failed': 'Échec de la réinitialisation d\'urgence',
            'application_startup_error': 'Erreur au démarrage de l\'application',
            'emergency_mode': 'Mode d\'Urgence',
            'application_encountered_startup_error': 'L\'application a rencontré une erreur au démarrage',
            'troubleshooting_steps': 'Étapes de Dépannage',
            'refresh_browser_page': 'Actualisez la page de votre navigateur',
            'check_all_dependencies_installed': 'Vérifiez que toutes les dépendances sont installées',
            'verify_model_files_in_weight_models_directory': 'Vérifiez les fichiers modèles dans le répertoire weight_models',
            'ensure_required_modules_available': 'Assurez-vous que les modules requis sont disponibles',
            'system_check': 'Vérification Système',
            
            # Session and Auto-save
            'session_auto_saved': 'Session auto-enregistrée',
            'timeline_entries': 'Entrées Chronologiques',
            'license_plate_utils': 'Utilitaires Plaques d\'Immatriculation',
            'sort_tracker': 'Suiveur SORT',
            'post_processing': 'Post-Traitement',
            'and': 'et',
            'more': 'plus',
            'end_of_report': 'Fin du Rapport',
            'vision_platform_report': 'Rapport Plateforme Visuelle YOLO11',
            'session_information': 'Informations de Session',
            'processing_statistics': 'Statistiques de Traitement',
            'detected_license_plates': 'Plaques Détectées',
            'device': 'Périphérique',
            'cameras_available': 'Caméras Disponibles',
            'session_stats': 'Statistiques de Session',
            'unknown': 'Inconnu',

            # General UI
            'enabled': 'Activé',
            'available': 'Disponible',
            'mode': 'Mode',
            'settings': 'Paramètres',
            'status': 'Statut',
            'loading': 'Chargement',
            'analyzing': 'Analyse en cours',
            'saving': 'Enregistrement',
            'exporting': 'Exportation',
            'downloading': 'Téléchargement',
            'uploading': 'Téléversement',
            'error': 'Erreur',
            'warning': 'Avertissement',
            
            # File uploader specific (if you have keys for these)
            # 'drag_and_drop_file_here': 'Glissez-déposez un fichier ici',
            # 'limit_500mb_per_file': 'Limite 500Mo par fichier...',
            # 'browse_files': 'Parcourir les fichiers',

            # Fallback for any missed keys, copy from English or mark
            'file_error': 'Erreur de Fichier',
            'tracking_reset': 'Réinitialisation du Suivi',
            'upload_parking_positions': 'Aucun fichier de positions de stationnement trouvé - téléchargez CarParkPos',
            'real_time': 'Temps Réel',

            # Model selection dropdown options
            'model_option_detection': 'Détection',
            'model_option_segmentation': 'Segmentation',
            'model_option_pose_estimation': 'Estimation de Pose',
            'model_option_elp': 'Plaque d\'Immatriculation Améliorée',
            'model_option_parking': 'Détection de Stationnement',
            'model_option_counting': 'Comptage et Système d\'Alerte',
        }

# --- Global instance and helper functions ---
lang_manager = LanguageManager()

def get_text(key: str, default: str = None) -> str:
    """Global helper to get translated text with fresh language state."""
    return lang_manager.get_text(key, default)

def _(key: str, default: str = None) -> str:
    """Alias for get_text for convenience (common i18n practice)."""
    return lang_manager.get_text(key, default)

# Initialize the manager to ensure session_state is set up
lang_manager.initialize()