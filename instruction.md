instruction.md

Step 3: Complete Tab 1 - Live Processing Integration
optimize and complete the Tab 1: Enhanced Live Detection system. Here's the specific code you need to integrate:

🔧 1. Enhanced Live Processing Engine
Replace the live processing section in Tab 1 with this optimized version:


# Enhanced Live Processing Engine - Replace existing Tab 1 live processing code

def create_optimized_live_processing_interface():
    """Create optimized live processing interface with enhanced performance"""
    
    # Enhanced camera connection status tracking
    if 'camera_connections' not in st.session_state:
        st.session_state.camera_connections = {}
    if 'processing_queues' not in st.session_state:
        st.session_state.processing_queues = {}
    if 'live_performance_stats' not in st.session_state:
        st.session_state.live_performance_stats = {}

def initialize_enhanced_camera_manager(camera_number):
    """Initialize enhanced camera manager with optimized settings"""
    
    if f'enhanced_camera_{camera_number}' not in st.session_state:
        try:
            # Create enhanced camera manager
            camera_manager = RTSPCameraManager(int(camera_number))
            
            # Enhanced connection settings
            camera_manager.max_retry_attempts = 5
            camera_manager.connection_timeout = 15
            camera_manager.frame_buffer_size = 10
            camera_manager.auto_reconnect_enabled = True
            camera_manager.performance_mode = "optimized"
            
            st.session_state[f'enhanced_camera_{camera_number}'] = camera_manager
            st.session_state[f'camera_status_{camera_number}'] = 'initialized'
            
            return camera_manager
            
        except Exception as e:
            st.error(f"Failed to initialize camera {camera_number}: {str(e)}")
            st.session_state[f'camera_status_{camera_number}'] = 'error'
            return None
    
    return st.session_state[f'enhanced_camera_{camera_number}']

def create_enhanced_live_controls(camera_number):
    """Create enhanced live processing controls with professional features"""
    
    st.markdown(f"""
    <div class="enhanced-license-info">
        <h4>🎛️ Enhanced Live Control - Camera {camera_number}</h4>
        <p>Professional live processing with real-time optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced connection controls
    with st.expander("🔌 Advanced Connection Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            connection_mode = st.selectbox(
                "Connection Mode",
                ["Auto (Recommended)", "TCP Only", "UDP Fallback", "Custom"],
                key=f"conn_mode_{camera_number}",
                help="RTSP connection optimization mode"
            )
            
            stream_quality = st.selectbox(
                "Stream Quality",
                ["Auto", "High (Main Stream)", "Medium (Sub Stream)", "Low (Mobile)"],
                index=1,
                key=f"stream_quality_{camera_number}",
                help="Video stream quality selection"
            )
        
        with col2:
            buffer_size = st.slider(
                "Buffer Size",
                1, 20, 10,
                key=f"buffer_size_{camera_number}",
                help="Frame buffer size for smooth playback"
            )
            
            reconnect_interval = st.slider(
                "Reconnect Interval (s)",
                5, 60, 15,
                key=f"reconnect_{camera_number}",
                help="Automatic reconnection interval"
            )
    
    # Enhanced processing controls
    with st.expander("⚡ Processing Optimization", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            processing_priority = st.selectbox(
                "Processing Priority",
                ["Balanced", "Speed Priority", "Quality Priority", "Memory Efficient"],
                key=f"proc_priority_{camera_number}",
                help="Processing optimization focus"
            )
            
            frame_skip = st.slider(
                "Frame Skip",
                0, 5, 0,
                key=f"frame_skip_{camera_number}",
                help="Skip frames for performance (0 = process all)"
            )
        
        with col2:
            enable_gpu_boost = st.checkbox(
                "🔥 GPU Boost",
                value=True if "cuda" in str(device).lower() else False,
                key=f"gpu_boost_{camera_number}",
                help="Enable GPU acceleration boost"
            )
            
            enable_smart_crop = st.checkbox(
                "🎯 Smart Cropping",
                value=True,
                key=f"smart_crop_{camera_number}",
                help="Intelligent region of interest detection"
            )
    
    # Enhanced license plate settings (if available)
    if is_enhanced_license_plate:
        with st.expander("🚗 License Plate Optimization", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                plate_detection_sensitivity = st.slider(
                    "Detection Sensitivity",
                    0.1, 1.0, 0.4, 0.05,
                    key=f"plate_sens_{camera_number}",
                    help="License plate detection sensitivity"
                )
                
                ocr_quality_mode = st.selectbox(
                    "OCR Quality Mode",
                    ["Fast", "Balanced", "High Quality", "Ultra Precise"],
                    index=1,
                    key=f"ocr_quality_{camera_number}",
                    help="OCR processing quality vs speed"
                )
            
            with col2:
                enable_format_validation = st.checkbox(
                    "📋 Format Validation",
                    value=True,
                    key=f"format_val_{camera_number}",
                    help="Validate license plate formats"
                )
                
                enable_confidence_boost = st.checkbox(
                    "📈 Confidence Boost",
                    value=True,
                    key=f"conf_boost_{camera_number}",
                    help="Boost confidence for high-quality detections"
                )
    
    # Return settings dictionary
    return {
        'connection_mode': connection_mode,
        'stream_quality': stream_quality,
        'buffer_size': buffer_size,
        'reconnect_interval': reconnect_interval,
        'processing_priority': processing_priority,
        'frame_skip': frame_skip,
        'enable_gpu_boost': enable_gpu_boost,
        'enable_smart_crop': enable_smart_crop,
        'plate_detection_sensitivity': plate_detection_sensitivity if is_enhanced_license_plate else 0.4,
        'ocr_quality_mode': ocr_quality_mode if is_enhanced_license_plate else 'Balanced',
        'enable_format_validation': enable_format_validation if is_enhanced_license_plate else True,
        'enable_confidence_boost': enable_confidence_boost if is_enhanced_license_plate else True
    }

def create_enhanced_status_display(camera_number):
    """Create enhanced status display with real-time metrics"""
    
    # Initialize status tracking
    if f'live_stats_{camera_number}' not in st.session_state:
        st.session_state[f'live_stats_{camera_number}'] = {
            'frames_processed': 0,
            'detections_made': 0,
            'plates_read': 0,
            'avg_fps': 0,
            'last_detection_time': 0,
            'session_start': time.time(),
            'connection_status': 'disconnected',
            'processing_status': 'idle',
            'error_count': 0,
            'quality_score': 0
        }
    
    stats = st.session_state[f'live_stats_{camera_number}']
    
    # Status display with professional styling
    st.markdown(f"""
    <div class="stats-container">
        <h4>📊 Live Status - Camera {camera_number}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-time metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Connection status with color coding
        conn_status = stats['connection_status']
        status_color = "🟢" if conn_status == "connected" else "🟡" if conn_status == "connecting" else "🔴"
        st.metric("Connection", f"{status_color} {conn_status.title()}")
    
    with col2:
        # Processing status
        proc_status = stats['processing_status']
        proc_color = "🟢" if proc_status == "active" else "🟡" if proc_status == "starting" else "⚪"
        st.metric("Processing", f"{proc_color} {proc_status.title()}")
    
    with col3:
        # Frame rate
        fps = stats['avg_fps']
        fps_color = "🟢" if fps > 15 else "🟡" if fps > 8 else "🔴" if fps > 0 else "⚪"
        st.metric("FPS", f"{fps_color} {fps:.1f}")
    
    with col4:
        # Detection rate
        detections = stats['detections_made']
        session_time = time.time() - stats['session_start']
        detection_rate = detections / max(1, session_time / 60)  # per minute
        st.metric("Detections/min", f"{detection_rate:.1f}")
    
    with col5:
        # Quality score
        quality = stats['quality_score']
        quality_color = "🟢" if quality > 0.8 else "🟡" if quality > 0.6 else "🔴" if quality > 0 else "⚪"
        st.metric("Quality", f"{quality_color} {quality:.2f}")
    
    # Detailed metrics
    with st.expander("📈 Detailed Performance Metrics", expanded=False):
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write(f"**📊 Session Statistics:**")
            st.write(f"• Frames Processed: {stats['frames_processed']:,}")
            st.write(f"• Total Detections: {stats['detections_made']:,}")
            st.write(f"• License Plates Read: {stats['plates_read']:,}")
            st.write(f"• Session Duration: {session_time/60:.1f} minutes")
        
        with detail_col2:
            st.write(f"**🔧 System Performance:**")
            st.write(f"• Error Count: {stats['error_count']}")
            st.write(f"• Last Detection: {time.time() - stats['last_detection_time']:.1f}s ago" if stats['last_detection_time'] > 0 else "• Last Detection: Never")
            st.write(f"• Processing Efficiency: {(stats['plates_read']/max(1,stats['detections_made'])*100):.1f}%")
            st.write(f"• System Load: {'High' if fps > 20 else 'Medium' if fps > 10 else 'Low'}")
    
    return stats

def process_enhanced_live_frame(frame, camera_number, settings, processor):
    """Process live frame with enhanced optimization"""
    
    if frame is None:
        return frame, {}
    
    try:
        # Apply frame skip optimization
        frame_skip = settings.get('frame_skip', 0)
        current_frame = st.session_state.get(f'current_frame_{camera_number}', 0)
        st.session_state[f'current_frame_{camera_number}'] = current_frame + 1
        
        if frame_skip > 0 and current_frame % (frame_skip + 1) != 0:
            return frame, {'skipped': True}
        
        # Smart cropping if enabled
        if settings.get('enable_smart_crop', False):
            frame = apply_smart_crop(frame, camera_number)
        
        # Processing priority optimization
        priority = settings.get('processing_priority', 'Balanced')
        if priority == "Speed Priority":
            # Reduce frame size for faster processing
            height, width = frame.shape[:2]
            if width > 1280:
                new_width = 1280
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))
        
        # Enhanced license plate processing
        if is_enhanced_license_plate and processor:
            # Apply enhanced settings
            original_confidence = processor.confidence
            processor.confidence = settings.get('plate_detection_sensitivity', 0.4)
            
            # Process with enhanced visualization
            processed_frame, results = process_frame_for_display(frame, processor)
            
            # Restore original confidence
            processor.confidence = original_confidence
            
            # Apply confidence boost if enabled
            if settings.get('enable_confidence_boost', False) and results.get('results'):
                for vehicle_id, vehicle_data in results['results'].items():
                    if 'license_plate' in vehicle_data:
                        original_score = vehicle_data['license_plate'].get('text_score', 0)
                        if original_score > 0.7:  # Boost high-quality detections
                            vehicle_data['license_plate']['text_score'] = min(1.0, original_score * 1.1)
            
            return processed_frame, results
        
        else:
            # Standard processing for non-license plate modes
            if model:
                # Apply model prediction
                results = model.predict(frame, conf=confidence_value, verbose=False)
                processed_frame = results[0].plot() if results else frame
                
                # Convert BGR to RGB if needed
                if len(processed_frame.shape) == 3:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                detection_count = len(results[0].boxes) if results and results[0].boxes else 0
                
                return processed_frame, {
                    'detections_count': detection_count,
                    'model_type': model_type,
                    'processing_mode': 'standard'
                }
            
            return frame, {}
    
    except Exception as e:
        st.error(f"Processing error for camera {camera_number}: {str(e)}")
        return frame, {'error': str(e)}

def apply_smart_crop(frame, camera_number):
    """Apply intelligent cropping to focus on areas of interest"""
    
    try:
        height, width = frame.shape[:2]
        
        # Smart crop based on common vehicle detection areas
        # Focus on middle 80% of frame (typical road/parking area)
        crop_top = int(height * 0.1)
        crop_bottom = int(height * 0.9)
        crop_left = int(width * 0.1)
        crop_right = int(width * 0.9)
        
        # Apply crop
        cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
        
        # Resize back to original dimensions to maintain consistency
        resized_frame = cv2.resize(cropped_frame, (width, height))
        
        return resized_frame
    
    except Exception as e:
        # Return original frame if crop fails
        return frame

def update_live_statistics(camera_number, results, processing_time):
    """Update live statistics with new processing results"""
    
    if f'live_stats_{camera_number}' not in st.session_state:
        return
    
    stats = st.session_state[f'live_stats_{camera_number}']
    
    # Update frame count
    stats['frames_processed'] += 1
    
    # Update FPS calculation (rolling average)
    if processing_time > 0:
        current_fps = 1.0 / processing_time
        stats['avg_fps'] = 0.8 * stats.get('avg_fps', current_fps) + 0.2 * current_fps
    
    # Update detection statistics
    if results and not results.get('error') and not results.get('skipped'):
        if 'results' in results and results['results']:
            # Enhanced license plate mode
            stats['detections_made'] += len(results['results'])
            stats['plates_read'] += results.get('plates_read', 0)
            stats['last_detection_time'] = time.time()
            
            # Update quality score
            if results.get('plates_read', 0) > 0:
                # Calculate average confidence for this frame
                confidences = []
                for vehicle_data in results['results'].values():
                    if 'license_plate' in vehicle_data:
                        conf = vehicle_data['license_plate'].get('text_score', 0)
                        if conf > 0:
                            confidences.append(conf)
                
                if confidences:
                    frame_quality = np.mean(confidences)
                    stats['quality_score'] = 0.9 * stats.get('quality_score', frame_quality) + 0.1 * frame_quality
        
        elif 'detections_count' in results:
            # Standard detection mode
            stats['detections_made'] += results['detections_count']
            if results['detections_count'] > 0:
                stats['last_detection_time'] = time.time()
    
    # Update error count
    if results.get('error'):
        stats['error_count'] += 1

def create_enhanced_export_options(camera_number):
    """Create enhanced export options for live session"""
    
    st.markdown("### 💾 Enhanced Live Export")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button(f"📊 Export Session Data", key=f"export_data_{camera_number}"):
            if is_enhanced_license_plate and st.session_state.enhanced_license_plate_processor:
                # Export enhanced license plate results
                processor = st.session_state.enhanced_license_plate_processor
                
                if processor.results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file = processor.save_results(f"live_session_camera_{camera_number}_{timestamp}.csv")
                    
                    if csv_file:
                        st.success(f"✅ Session data exported: {csv_file}")
                        
                        # Provide download option
                        try:
                            with open(csv_file, 'r') as f:
                                csv_content = f.read()
                            
                            st.download_button(
                                label="📥 Download Session CSV",
                                data=csv_content,
                                file_name=csv_file,
                                mime="text/csv",
                                key=f"download_csv_{camera_number}"
                            )
                        except Exception as e:
                            st.error(f"Download preparation failed: {e}")
                    else:
                        st.error("Failed to save session data")
                else:
                    st.warning("No session data to export")
            else:
                st.warning("Enhanced license plate processing not active")
    
    with export_col2:
        if st.button(f"📈 Export Analytics", key=f"export_analytics_{camera_number}"):
            # Export comprehensive analytics
            analytics_data = {
                'camera_id': camera_number,
                'session_info': {
                    'start_time': st.session_state.get('session_start', time.time()),
                    'export_time': time.time(),
                    'duration_minutes': (time.time() - st.session_state.get('session_start', time.time())) / 60
                },
                'live_statistics': st.session_state.get(f'live_stats_{camera_number}', {}),
                'detection_analytics': {
                    'total_detections': st.session_state.detection_analytics['total_detections'],
                    'unique_plates': list(st.session_state.detection_analytics['unique_plates']),
                    'confidence_scores': st.session_state.detection_analytics['confidence_scores'],
                    'detection_timeline': st.session_state.detection_analytics['detection_timeline']
                },
                'system_configuration': {
                    'model_type': model_type,
                    'device': str(device),
                    'enhanced_features': is_enhanced_license_plate
                }
            }
            
            analytics_json = json.dumps(analytics_data, indent=2, default=str)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Download Analytics JSON",
                data=analytics_json,
                file_name=f"live_analytics_camera_{camera_number}_{timestamp}.json",
                mime="application/json",
                key=f"download_analytics_{camera_number}"
            )
            
            st.success("📈 Analytics export prepared")
    
    with export_col3:
        if st.button(f"📋 Generate Report", key=f"generate_report_{camera_number}"):
            # Generate live session report
            stats = st.session_state.get(f'live_stats_{camera_number}', {})
            session_duration = time.time() - stats.get('session_start', time.time())
            
            live_report = f"""Live CCTV Session Report
Camera: {camera_number}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SESSION SUMMARY:
- Duration: {session_duration/60:.1f} minutes
- Frames Processed: {stats.get('frames_processed', 0):,}
- Average FPS: {stats.get('avg_fps', 0):.1f}
- Total Detections: {stats.get('detections_made', 0):,}
- License Plates Read: {stats.get('plates_read', 0):,}
- Processing Quality: {stats.get('quality_score', 0):.3f}

PERFORMANCE METRICS:
- Detection Rate: {stats.get('detections_made', 0) / max(1, session_duration/60):.1f} per minute
- Success Rate: {(stats.get('plates_read', 0)/max(1, stats.get('detections_made', 1))*100):.1f}%
- Error Count: {stats.get('error_count', 0)}
- System Efficiency: {'High' if stats.get('avg_fps', 0) > 15 else 'Medium' if stats.get('avg_fps', 0) > 8 else 'Low'}

UNIQUE PLATES DETECTED:
{chr(10).join([f"• {plate}" for plate in list(st.session_state.detection_analytics['unique_plates'])[:10]])}
{'• ... and more' if len(st.session_state.detection_analytics['unique_plates']) > 10 else ''}

SYSTEM STATUS: {'✅ Operational' if stats.get('connection_status') == 'connected' else '⚠️ Connection Issues'}
"""
            
            st.download_button(
                label="📥 Download Live Report",
                data=live_report,
                file_name=f"live_report_camera_{camera_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"download_report_{camera_number}"
            )
            
            st.success("📋 Live session report generated")




 2. Replace Tab 1 CCTV Section
Find the CCTV section in Tab 1 (around line 800-1200) and replace it with this optimized version:


elif source_radio == CCTV:
    st.subheader("📡 Ultimate Live CCTV Analysis")
    
    # Enhanced camera selection interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera selection with enhanced info
        camera_options = list(CAMERA_FUNCTIONS.keys())
        selected_camera = st.selectbox(
            "🎥 Select Camera for Live Processing",
            camera_options,
            help="Choose camera for enhanced live processing"
        )
        
        camera_number = selected_camera.split()[-1]
        camera_ip = camera_map.get(int(camera_number), "Unknown")
        
        # Display camera info
        st.info(f"📍 Selected: {selected_camera} (IP: {camera_ip})")
        
        # Enhanced capability display
        if is_enhanced_license_plate:
            st.markdown("""
            <div class="enhanced-license-info">
                <h4>🚗 Ultimate License Plate Recognition Active</h4>
                <p>✅ Real-time professional vehicle tracking with custom corner borders</p>
                <p>✅ Advanced OCR with format validation and character correction</p>
                <p>✅ Professional visualization with overlay displays and confidence scoring</p>
                <p>✅ Comprehensive analytics with session persistence and data export</p>
                <p>✅ Complete post-processing pipeline integration</p>
            </div>
            """, unsafe_allow_html=True)
        elif supports_shapes:
            st.info(f"🎯 {model_type} with shape-based analytics enabled")
        else:
            st.info(f"📊 {model_type} - Standard detection mode")
    
    with col2:
        # Enhanced control panel
        st.markdown("### 🎛️ Enhanced Control Panel")
        
        # Initialize enhanced camera manager
        camera_manager = initialize_enhanced_camera_manager(camera_number)
        
        # Get enhanced settings
        enhanced_settings = create_enhanced_live_controls(camera_number)
        
        # Enhanced connection controls
        if st.button("🔌 Enhanced Connect", key=f"enhanced_connect_{camera_number}", type="primary", use_container_width=True):
            if camera_manager:
                with st.spinner(f"🚀 Connecting to {selected_camera} with enhanced settings..."):
                    try:
                        # Apply enhanced settings
                        camera_manager.max_retry_attempts = 5
                        camera_manager.connection_timeout = enhanced_settings['reconnect_interval']
                        camera_manager.frame_buffer_size = enhanced_settings['buffer_size']
                        
                        # Attempt enhanced connection
                        if camera_manager.connect():
                            camera_manager.start_capture_thread()
                            
                            # Update status
                            st.session_state[f'live_stats_{camera_number}']['connection_status'] = 'connected'
                            st.session_state[f'enhanced_camera_connected_{camera_number}'] = True
                            
                            st.success(f"✅ Enhanced connection established to {selected_camera}")
                            st.info(f"🎯 Settings Applied: {enhanced_settings['processing_priority']} mode, {enhanced_settings['stream_quality']} quality")
                            
                        else:
                            st.error(f"❌ Failed to connect to {selected_camera}")
                            st.session_state[f'live_stats_{camera_number}']['connection_status'] = 'failed'
                            
                    except Exception as e:
                        st.error(f"❌ Enhanced connection error: {str(e)}")
                        st.session_state[f'live_stats_{camera_number}']['connection_status'] = 'error'
            else:
                st.error("❌ Camera manager initialization failed")
    
    # Enhanced status display
    st.markdown("---")
    live_stats = create_enhanced_status_display(camera_number)
    
    # Main enhanced processing interface
    if st.session_state.get(f'enhanced_camera_connected_{camera_number}', False):
        
        # Enhanced processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Enhanced Processing", 
                        disabled=st.session_state.get(f'enhanced_processing_active_{camera_number}', False),
                        key=f"start_enhanced_{camera_number}", 
                        use_container_width=True):
                
                st.session_state[f'enhanced_processing_active_{camera_number}'] = True
                st.session_state[f'live_stats_{camera_number}']['processing_status'] = 'active'
                st.session_state.processing_active = True
                
                st.success("🚀 Enhanced processing started!")
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Processing", 
                        disabled=not st.session_state.get(f'enhanced_processing_active_{camera_number}', False),
                        key=f"stop_enhanced_{camera_number}", 
                        use_container_width=True):
                
                st.session_state[f'enhanced_processing_active_{camera_number}'] = False
                st.session_state[f'live_stats_{camera_number}']['processing_status'] = 'stopped'
                st.session_state.processing_active = False
                
                # Auto-save if enabled
                if enhanced_settings.get('enable_auto_save', True) and is_enhanced_license_plate:
                    if st.session_state.enhanced_license_plate_processor and st.session_state.enhanced_license_plate_processor.results:
                        timestamp = int(time.time())
                        csv_file = st.session_state.enhanced_license_plate_processor.save_results(
                            f"enhanced_live_session_camera_{camera_number}_{timestamp}.csv"
                        )
                        if csv_file:
                            st.success(f"💾 Session auto-saved: {csv_file}")
                
                st.info("⏹️ Enhanced processing stopped")
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Session", key=f"reset_enhanced_{camera_number}", use_container_width=True):
                # Reset all session data
                st.session_state[f'enhanced_processing_active_{camera_number}'] = False
                st.session_state[f'live_stats_{camera_number}'] = {
                    'frames_processed': 0,
                    'detections_made': 0,
                    'plates_read': 0,
                    'avg_fps': 0,
                    'last_detection_time': 0,
                    'session_start': time.time(),
                    'connection_status': 'connected',
                    'processing_status': 'idle',
                    'error_count': 0,
                    'quality_score': 0
                }
                
                # Reset analytics
                st.session_state.detection_analytics = {
                    'total_detections': 0,
                    'unique_plates': set(),
                    'detection_timeline': [],
                    'confidence_scores': []
                }
                
                if is_enhanced_license_plate and st.session_state.enhanced_license_plate_processor:
                    st.session_state.enhanced_license_plate_processor.license_plates.clear()
                    st.session_state.enhanced_license_plate_processor.vehicle_history.clear()
                    st.session_state.enhanced_license_plate_processor.last_detection_time.clear()
                
                st.success("🔄 Enhanced session reset complete!")
                st.rerun()
        
        # Enhanced live video display
        st.markdown("---")
        st.subheader(f"📹 Enhanced Live Feed - {selected_camera}")
        
        # Video display container with enhanced features
       video_container = st.container()
       status_container = st.container()
       
       # Enhanced live processing loop
       if st.session_state.get(f'enhanced_processing_active_{camera_number}', False):
           
           # Get frame from enhanced camera manager
           if camera_manager and camera_manager.connected:
               start_time = time.time()
               frame = camera_manager.get_frame(timeout=0.5)
               
               if frame is not None:
                   # Process frame with enhanced settings
                   processed_frame, results = process_enhanced_live_frame(
                       frame, camera_number, enhanced_settings, 
                       st.session_state.enhanced_license_plate_processor if is_enhanced_license_plate else None
                   )
                   
                   processing_time = time.time() - start_time
                   
                   # Update statistics
                   update_live_statistics(camera_number, results, processing_time)
                   
                   # Display enhanced frame
                   if processed_frame is not None:
                       # Add enhanced overlay information
                       display_frame = processed_frame.copy()
                       
                       # Add performance overlay
                       current_stats = st.session_state[f'live_stats_{camera_number}']
                       overlay_text = [
                           f"FPS: {current_stats['avg_fps']:.1f}",
                           f"Detections: {current_stats['detections_made']}",
                           f"Quality: {current_stats['quality_score']:.2f}"
                       ]
                       
                       if is_enhanced_license_plate:
                           overlay_text.append(f"Plates: {current_stats['plates_read']}")
                       
                       # Draw overlay
                       y_offset = 30
                       for text in overlay_text:
                           cv2.putText(display_frame, text, (10, y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                           y_offset += 25
                       
                       # Add timestamp
                       timestamp_text = datetime.now().strftime("%H:%M:%S")
                       cv2.putText(display_frame, timestamp_text, (display_frame.shape[1] - 100, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
                       # Convert BGR to RGB for display
                       if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                           display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                       else:
                           display_frame_rgb = display_frame
                       
                       # Display enhanced video
                       with video_container:
                           st.image(
                               display_frame_rgb,
                               caption=f"🚗 Enhanced Live Processing - {selected_camera}",
                               use_container_width=True
                           )
                       
                       # Enhanced status display
                       with status_container:
                           current_time = time.strftime("%H:%M:%S")
                           status_text = f"🟢 **Enhanced Processing Active** | Frame: {current_stats['frames_processed']} | Time: {current_time} | FPS: {current_stats['avg_fps']:.1f}"
                           
                           if results and not results.get('error') and not results.get('skipped'):
                               if is_enhanced_license_plate and results.get('plates_read', 0) > 0:
                                   status_text += f" | 🎯 **{results['plates_read']} plates detected!**"
                               elif results.get('detections_count', 0) > 0:
                                   status_text += f" | 🎯 **{results['detections_count']} objects detected!**"
                           
                           st.markdown(status_text)
                           
                           # Show real-time detection results for license plates
                           if is_enhanced_license_plate and results.get('results'):
                               with st.expander("🔍 Real-time License Plate Results", expanded=False):
                                   for vehicle_id, vehicle_data in results['results'].items():
                                       if 'license_plate' in vehicle_data:
                                           plate_info = vehicle_data['license_plate']
                                           
                                           st.markdown(f"""
                                           **🚙 Vehicle {vehicle_id}:**  
                                           📋 **License Plate:** `{plate_info.get('text', 'Unknown')}`  
                                           📊 **Confidence:** {plate_info.get('text_score', 0):.3f}  
                                           🎯 **Detection Quality:** {'Excellent' if plate_info.get('text_score', 0) > 0.8 else 'Good' if plate_info.get('text_score', 0) > 0.6 else 'Fair'}
                                           """)
                       
                       # Update session analytics
                       if results and not results.get('error') and not results.get('skipped'):
                           current_time_stamp = time.time()
                           
                           if is_enhanced_license_plate and results.get('results'):
                               for vehicle_data in results['results'].values():
                                   if 'license_plate' in vehicle_data:
                                       plate_text = vehicle_data['license_plate'].get('text', '')
                                       confidence = vehicle_data['license_plate'].get('text_score', 0)
                                       
                                       if plate_text and confidence > 0:
                                           # Add to analytics
                                           st.session_state.detection_analytics['unique_plates'].add(plate_text)
                                           st.session_state.detection_analytics['confidence_scores'].append(confidence)
                                           st.session_state.detection_analytics['detection_timeline'].append({
                                               'timestamp': current_time_stamp,
                                               'plate': plate_text,
                                               'confidence': confidence,
                                               'vehicle_id': vehicle_id,
                                               'camera': camera_number
                                           })
                                           
                                           # Keep analytics manageable
                                           if len(st.session_state.detection_analytics['detection_timeline']) > 1000:
                                               st.session_state.detection_analytics['detection_timeline'] = \
                                                   st.session_state.detection_analytics['detection_timeline'][-500:]
                                           
                                           if len(st.session_state.detection_analytics['confidence_scores']) > 1000:
                                               st.session_state.detection_analytics['confidence_scores'] = \
                                                   st.session_state.detection_analytics['confidence_scores'][-500:]
               
               else:
                   # No frame available
                   with status_container:
                       st.markdown("🟡 **Waiting for frame...**")
               
               # Auto-refresh for continuous processing
               time.sleep(0.033)  # ~30 FPS max
               st.rerun()
           
           else:
               # Camera disconnected during processing
               with status_container:
                   st.markdown("🔴 **Camera Disconnected** - Attempting reconnection...")
                   
                   # Attempt reconnection
                   if camera_manager:
                       if camera_manager.connect():
                           camera_manager.start_capture_thread()
                           st.session_state[f'live_stats_{camera_number}']['connection_status'] = 'connected'
                           st.success("✅ Reconnection successful!")
                       else:
                           st.session_state[f'live_stats_{camera_number}']['connection_status'] = 'failed'
                           st.error("❌ Reconnection failed")
               
               time.sleep(2)
               st.rerun()
       
       else:
           # Processing not active - show static frame
           if camera_manager and camera_manager.connected:
               frame = camera_manager.get_frame(timeout=1.0)
               if frame is not None:
                   # Resize for display
                   display_frame = cv2.resize(frame, (1280, 720))
                   frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                   
                   # Add "Ready" overlay
                   cv2.putText(frame_rgb, "READY - Click Start to Begin Processing", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
                   with video_container:
                       st.image(
                           frame_rgb,
                           caption=f"📹 {selected_camera} - Ready for Enhanced Processing",
                           use_container_width=True
                       )
               else:
                   # Show placeholder
                   placeholder_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                   cv2.putText(placeholder_img, f"{selected_camera} - Connected", (400, 360), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                   
                   with video_container:
                       st.image(placeholder_img, channels="BGR", use_container_width=True)
               
               with status_container:
                   st.markdown("🟡 **Connected - Ready for Enhanced Processing**")
           
           else:
               # Show connection placeholder
               placeholder_img = np.zeros((720, 1280, 3), dtype=np.uint8)
               cv2.putText(placeholder_img, "Enhanced Connect Required", (450, 320), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 3)
               cv2.putText(placeholder_img, f"Selected: {selected_camera}", (500, 400), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
               
               with video_container:
                   st.image(placeholder_img, channels="BGR", use_container_width=True)
               
               with status_container:
                   st.markdown("🔴 **Not Connected** - Click Enhanced Connect")
       
       # Enhanced export and analytics section
       if st.session_state.get(f'enhanced_processing_active_{camera_number}', False) or \
          st.session_state[f'live_stats_{camera_number}']['frames_processed'] > 0:
           
           st.markdown("---")
           
           # Enhanced session analytics
           col1, col2 = st.columns([2, 1])
           
           with col1:
               st.subheader("📊 Enhanced Session Analytics")
               
               # Real-time performance chart
               if len(st.session_state.detection_analytics['detection_timeline']) > 5:
                   timeline_df = pd.DataFrame(st.session_state.detection_analytics['detection_timeline'][-50:])
                   timeline_df['time'] = pd.to_datetime(timeline_df['timestamp'], unit='s')
                   
                   # Create real-time chart
                   fig_realtime = px.scatter(
                       timeline_df,
                       x='time',
                       y='confidence',
                       color='plate',
                       title='Real-Time Detection Performance',
                       labels={'confidence': 'OCR Confidence', 'time': 'Time'},
                       hover_data=['vehicle_id', 'camera']
                   )
                   fig_realtime.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Excellent")
                   fig_realtime.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Good")
                   
                   st.plotly_chart(fig_realtime, use_container_width=True)
               
               # Performance summary
               stats = st.session_state[f'live_stats_{camera_number}']
               session_duration = time.time() - stats['session_start']
               
               perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
               
               with perf_col1:
                   st.metric("Session Duration", f"{session_duration/60:.1f} min")
                   st.metric("Processing FPS", f"{stats['avg_fps']:.1f}")
               
               with perf_col2:
                   st.metric("Total Frames", f"{stats['frames_processed']:,}")
                   st.metric("Total Detections", f"{stats['detections_made']:,}")
               
               with perf_col3:
                   if is_enhanced_license_plate:
                       st.metric("Plates Read", f"{stats['plates_read']:,}")
                       st.metric("Unique Plates", f"{len(st.session_state.detection_analytics['unique_plates'])}")
                   else:
                       st.metric("Objects Detected", f"{stats['detections_made']:,}")
                       st.metric("Detection Rate", f"{stats['detections_made']/max(1,session_duration/60):.1f}/min")
               
               with perf_col4:
                   st.metric("Quality Score", f"{stats['quality_score']:.3f}")
                   st.metric("Error Count", f"{stats['error_count']}")
           
           with col2:
               # Enhanced export options
               create_enhanced_export_options(camera_number)
       
       # Enhanced shape analytics integration (if supported)
       if supports_shapes and not is_enhanced_license_plate:
           st.markdown("---")
           st.subheader("🎯 Enhanced Shape Analytics")
           
           # Get latest frame for shape drawing
           if camera_manager and camera_manager.connected:
               frame = camera_manager.get_frame(timeout=1.0)
               if frame is not None:
                   # Integrate shape analytics
                   shape_frame, shape_results = shape_interface_wrapper(
                       frame, camera_number, use_canvas=False
                   )
                   
                   if shape_results:
                       st.success(f"Shape analytics active: {len(shape_results.get('shapes', {}).get('lines_in', []))} IN lines, {len(shape_results.get('shapes', {}).get('lines_out', []))} OUT lines")
   
   else:
       # Not connected - show enhanced connection interface
       st.markdown("""
       <div class="alert-box">
           <h4>📡 Enhanced CCTV Connection Required</h4>
           <p>Click "Enhanced Connect" to establish connection with advanced features:</p>
           <ul>
               <li>🚀 Optimized RTSP connection with auto-reconnection</li>
               <li>⚡ Smart processing with quality/speed optimization</li>
               <li>🎯 Intelligent frame cropping and region of interest</li>
               <li>📊 Real-time performance monitoring and analytics</li>
               <li>💾 Advanced export options with comprehensive reporting</li>
           </ul>
       </div>
       """, unsafe_allow_html=True)
       
       # Show camera information
       st.info(f"Selected Camera: {selected_camera} (IP: {camera_ip})")
       
       # Enhanced connection preview
       st.markdown("### 🔧 Enhanced Connection Preview")
       
       preview_col1, preview_col2 = st.columns(2)
       
       with preview_col1:
           st.markdown("**🚀 Enhanced Features:**")
           st.write("• Optimized RTSP connection management")
           st.write("• Smart buffering and frame rate optimization") 
           st.write("• Advanced error handling and auto-recovery")
           st.write("• Real-time performance monitoring")
           st.write("• Professional visualization overlays")
       
       with preview_col2:
           st.markdown("**📊 Analytics Capabilities:**")
           st.write("• Live detection timeline tracking")
           st.write("• Quality score monitoring and trends")
           st.write("• Session persistence and export options")
           st.write("• Comprehensive reporting tools")
           st.write("• Advanced data visualization")





🔧 3. Add Performance Optimization Functions
Add these helper functions at the end of display.py (before the __all__ export):


def optimize_live_processing_performance():
    """Optimize live processing performance based on system capabilities"""
    
    # Memory management
    if 'live_frame_cache' not in st.session_state:
        st.session_state.live_frame_cache = {}
    
    # Clean up old cache entries
    current_time = time.time()
    for key in list(st.session_state.live_frame_cache.keys()):
        if current_time - st.session_state.live_frame_cache[key].get('timestamp', 0) > 30:
            del st.session_state.live_frame_cache[key]

def handle_live_processing_errors(camera_number, error):
    """Enhanced error handling for live processing"""
    
    error_stats_key = f'error_stats_{camera_number}'
    if error_stats_key not in st.session_state:
        st.session_state[error_stats_key] = {
            'error_count': 0,
            'last_error_time': 0,
            'error_types': defaultdict(int)
        }
    
    error_stats = st.session_state[error_stats_key]
    error_stats['error_count'] += 1
    error_stats['last_error_time'] = time.time()
    error_stats['error_types'][type(error).__name__] += 1
    
    # Update live stats
    if f'live_stats_{camera_number}' in st.session_state:
        st.session_state[f'live_stats_{camera_number}']['error_count'] = error_stats['error_count']
    
    # Auto-recovery logic
    if error_stats['error_count'] > 10:
        st.warning(f"⚠️ Multiple errors detected for camera {camera_number}. Consider reconnecting.")

def cleanup_live_session_data():
    """Clean up live session data to prevent memory issues"""
    
    # Clean up old session states
    keys_to_clean = []
    for key in st.session_state.keys():
        if any(pattern in key for pattern in ['live_stats_', 'enhanced_camera_', 'current_frame_']):
            keys_to_clean.append(key)
    
    # Keep only recent keys (last 5 cameras)
    if len(keys_to_clean) > 50:
        for key in keys_to_clean[:-50]:
            del st.session_state[key]
    
    # Optimize analytics data
    if len(st.session_state.detection_analytics['detection_timeline']) > 2000:
        st.session_state.detection_analytics['detection_timeline'] = \
            st.session_state.detection_analytics['detection_timeline'][-1000:]
    
    if len(st.session_state.detection_analytics['confidence_scores']) > 2000:
        st.session_state.detection_analytics['confidence_scores'] = \
            st.session_state.detection_analytics['confidence_scores'][-1000:]

# Add cleanup to session initialization
def initialize_enhanced_live_session():
    """Initialize enhanced live session with optimization"""
    
    # Set session start time
    if 'session_start' not in st.session_state:
        st.session_state.session_start = time.time()
    
    # Initialize performance tracking
    if 'live_performance_history' not in st.session_state:
        st.session_state.live_performance_history = []
    
    # Periodic cleanup
    if hasattr(st.session_state, 'last_cleanup_time'):
        if time.time() - st.session_state.last_cleanup_time > 300:  # 5 minutes
            cleanup_live_session_data()
            st.session_state.last_cleanup_time = time.time()
    else:
        st.session_state.last_cleanup_time = time.time()
    
    # Performance optimization
    optimize_live_processing_performance()

# Call initialization
initialize_enhanced_live_session()




🎯 4. Integration Instructions
Step-by-Step Integration:

Find Tab 1 CCTV Section in display.py 
Replace the entire elif source_radio == CCTV: section with the enhanced version above
Add the helper functions at the end of the file
Test the integration

Key Files to Modify:
# In display.py:
# 1. Replace Tab 1 CCTV section 
# 2. Add helper functions at the end
# 3. Ensure all imports are present at the top