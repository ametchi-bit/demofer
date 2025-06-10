# styles.py - UI Styling and Themes for Vision Platform

import streamlit as st


class StyleManager:
    """Comprehensive style manager for the Vision Platform"""
    
    def __init__(self):
        # Updated palette based on user request
        self.golden_yellow = "#FFC107"  # Usage: Logo background, gradient elements
        self.bright_yellow = "#FFEB3B"  # Usage: Right-side yellow wave / FM 101.3
        self.orange = "#F57C00"         # Usage: Headset icon, gradient logo edge
        self.dark_gray_navy = "#1C1C1C" # Usage: Top navigation bar (can be primary)
        self.white = "#FFFFFF"          # Usage: Background, text contrast
        self.red_accent = "#FF0000"     # Usage: Flash info highlight and date

        # Keeping original semantic names but mapping to new palette
        self.primary_color = self.dark_gray_navy  # Assuming Dark Gray/Navy is the new primary
        self.secondary_color = self.orange        # Assuming Orange is the new secondary
        
        self.success_color = "#28a745"  # Keeping original green for success
        self.warning_color = self.golden_yellow # Golden Yellow for warnings / logo
        self.error_color = self.red_accent     # Red (Accent) for errors / flash info
        self.info_color = "#17a2b8"      # Keeping original blue for info
        
    def apply_global_styles(self):
        """Apply global CSS styles to the application"""
        st.markdown(self.get_global_css(), unsafe_allow_html=True)
    
    def get_global_css(self):
        """Get comprehensive global CSS styles"""
        return f"""
        <style>
        /* Global Variables */
        :root {{
            /* New Palette Variables */
            --golden-yellow: {self.golden_yellow};
            --bright-yellow: {self.bright_yellow};
            --orange: {self.orange};
            --dark-gray-navy: {self.dark_gray_navy};
            --white: {self.white};
            --red-accent: {self.red_accent};

            /* Semantic Variables using the new palette */
            --primary-color: {self.primary_color};
            --secondary-color: {self.secondary_color};
            --success-color: {self.success_color};
            --warning-color: {self.warning_color}; /* Mapped to Golden Yellow */
            --error-color: {self.error_color};   /* Mapped to Red Accent */
            --info-color: {self.info_color};

            /* Example Gradients (can be customized further) */
            --gradient-primary: linear-gradient(135deg, {self.primary_color} 0%, {self.secondary_color} 100%); /* Dark Gray to Orange */
            --gradient-logo: linear-gradient(135deg, {self.golden_yellow} 0%, {self.orange} 100%); /* Golden Yellow to Orange */
            --gradient-yellow-wave: linear-gradient(to right, {self.bright_yellow}, {self.golden_yellow}); /* For yellow wave effect */


            /* Existing structural variables - adjust if needed */
            --shadow-light: 0 2px 10px rgba(0,0,0,0.1);
            --shadow-medium: 0 4px 20px rgba(0,0,0,0.15);
            --shadow-heavy: 0 8px 30px rgba(0,0,0,0.2);
            --border-radius: 12px;
            --border-radius-small: 8px;
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Main Container Styles */
        .main {{
            padding: 0rem 1rem;
            background-color: var(--white); /* Ensuring main background is white */
        }}
        
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }}
        
        /* Header Styles - Assuming this is the 'Top navigation bar' */
        .main-header {{
            background: var(--dark-gray-navy); /* Using Dark Gray/Navy directly */
            padding: 1rem 2rem; /* Adjusted padding for typical nav bar */
            border-radius: 0; /* Nav bars often aren't rounded unless specified */
            margin-bottom: 2rem;
            box-shadow: var(--shadow-medium);
            color: var(--white); /* Text color on dark nav bar */
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .main-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.05"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.05"/><circle cx="50" cy="10" r="1" fill="white" opacity="0.05"/><circle cx="10" cy="90" r="1" fill="white" opacity="0.05"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>'); /* Subtle grain for dark bg */
            opacity: 0.3;
        }}
        
        .main-header h1 {{
            margin: 0;
            font-size: 1.8rem; /* Adjusted for nav bar */
            font-weight: 600;
            color: var(--white);
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            text-align: left;
        }}
        
        .main-header p {{
            margin: 0.25rem 0 0 0;
            font-size: 1rem;
            color: var(--white);
            opacity: 0.8;
            position: relative;
            z-index: 1;
            text-align: left;
        }}

        /* Placeholder for Right-side yellow wave / FM 101.3 */
        .right-yellow-wave {{ /* You'll need to create an element with this class */
            background: var(--gradient-yellow-wave); 
            width: 100px; /* Example */
            height: 100%; /* Example */
            position: fixed; /* Example */
            right: 0; /* Example */
            top: 0; /* Example */
            /* Add specific dimensions and positioning for the wave effect */
        }}

        /* Placeholder for Headset icon color */
        .headset-icon-class {{ /* Replace with actual class or SVG target */
            fill: var(--orange); /* For SVG icons */
            color: var(--orange); /* For font icons */
        }}

        /* Placeholder for Logo background */
        .logo-container-class {{ /* Replace with actual class for the logo's container */
            background-color: var(--golden-yellow);
            /* Or if using gradient: background: var(--gradient-logo); */
            padding: 10px; /* Example padding */
            display: inline-block; /* Example display */
        }}
        
        /* Placeholder for Flash info highlight and date */
        .flash-info-class, .date-highlight-class {{ /* Replace with actual classes */
            color: var(--red-accent) !important; /* Important to override other styles if needed */
            font-weight: bold; /* Example */
            /* background-color: var(--white); /* Optional: if on dark bg */
        }}

        /* Ensure text has good contrast, especially if background changes */
        body {{
             color: {self.dark_gray_navy}; /* Default text color for contrast with white bg */
             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font stack */
        }}
        h1, h2, h3, h4, h5, h6 {{
             color: {self.dark_gray_navy}; /* Default heading color */
        }}
        
        /* Tab System */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: #f0f2f6; /* Lighter gray for tab list background */
            padding: 8px;
            border-radius: var(--border-radius);
            border: 1px solid #e0e0e0;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 60px;
            padding: 12px 24px;
            background-color: var(--white);
            border-radius: var(--border-radius-small);
            color: var(--dark-gray-navy); /* Text color for inactive tabs */
            font-weight: 500;
            border: 1px solid #d0d0d0;
            transition: var(--transition-smooth);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: var(--primary-color) !important; /* Using primary (Dark Gray/Navy) for active tab */
            color: var(--white) !important;
            border-color: transparent !important;
            transform: translateY(-2px);
            box-shadow: var(--shadow-light);
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #e9ecef; /* Slightly darker hover for inactive tabs */
            transform: translateY(-1px);
            box-shadow: var(--shadow-light);
        }}
        
        /* Card Components */
        .metric-card {{
            background: var(--white);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-light);
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            transition: var(--transition-smooth);
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }}
        
        .metric-card h4 {{
            margin: 0 0 0.5rem 0;
            color: var(--primary-color); /* Card titles use primary color */
            font-weight: 600;
        }}
        
        .metric-card p {{
            margin: 0;
            color: #555; /* Darker gray for card text */
            line-height: 1.5;
        }}
        
        /* Success Box (example of using semantic color) */
        .success-box {{
            background: linear-gradient(135deg, #e6ffed 0%, #d4fce3 100%); /* Lighter green gradient */
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            box-shadow: var(--shadow-light);
            border-left: 4px solid var(--success-color);
        }}
        
        .success-box h4 {{
            margin: 0 0 0.5rem 0;
            color: #1d7a3f; /* Darker green for text */
            font-weight: 600;
        }}
        
        .success-box p {{
            margin: 0;
            color: #1d7a3f;
            line-height: 1.5;
        }}
        
        /* Enhanced License Plate Info - Example, adjust as needed */
        .enhanced-license-info {{
            background: var(--gradient-logo); /* Using logo gradient: Golden Yellow to Orange */
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            color: var(--dark-gray-navy); /* Text color for contrast on yellow/orange */
            box-shadow: var(--shadow-medium);
            position: relative;
            overflow: hidden;
        }}
        
        .enhanced-license-info::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%); /* Subtle effect */
            transform: rotate(45deg);
        }}
        
        .enhanced-license-info h4 {{
            margin: 0 0 0.5rem 0;
            font-weight: 600;
            position: relative;
            z-index: 1;
        }}
        
        .enhanced-license-info p {{
            margin: 0.25rem 0;
            opacity: 0.9;
            line-height: 1.4;
            position: relative;
            z-index: 1;
        }}
        
        /* Model Info Box - Example, adjust as needed */
        .model-info {{
            background: linear-gradient(135deg, var(--info-color) 0%, #56c1d6 100%); /* Using info color */
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            color: var(--white);
            box-shadow: var(--shadow-medium);
        }}
        
        .model-info h4 {{
            margin: 0 0 0.5rem 0;
            font-weight: 600;
        }}
        
        .model-info p {{
            margin: 0.25rem 0;
            opacity: 0.9;
            line-height: 1.4;
        }}
        
        /* Stats Container */
        .stats-container {{
            background: var(--gradient-primary); /* Dark Gray to Orange */
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            color: var(--white);
            box-shadow: var(--shadow-medium);
        }}
        
        .stats-container h3, .stats-container h4 {{
            margin: 0 0 0.5rem 0;
            font-weight: 600;
        }}
        
        .stats-container p {{
            margin: 0.25rem 0;
            opacity: 0.9;
            line-height: 1.4;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stats-grid .metric-card {{
            margin-bottom: 0;
        }}
        
        /* Alert Box */
        .alert-box {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            box-shadow: var(--shadow-light);
            border-left: 4px solid var(--warning-color);
        }}
        
        .alert-box h4 {{
            margin: 0 0 0.5rem 0;
            color: #856404;
            font-weight: 600;
        }}
        
        .alert-box p, .alert-box ul {{
            margin: 0.5rem 0;
            color: #856404;
            line-height: 1.5;
        }}
        
        .alert-box li {{
            margin: 0.25rem 0;
        }}
        
        /* License Plate Detection Result */
        .plate-detection {{
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            box-shadow: var(--shadow-light);
            border: 2px solid var(--primary-color);
            transition: var(--transition-smooth);
        }}
        
        .plate-detection:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }}
        
        .plate-detection h4 {{
            margin: 0 0 0.5rem 0;
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        .plate-detection p {{
            margin: 0.5rem 0;
            line-height: 1.5;
        }}
        
        /* Status Indicators */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 500;
            margin: 0.25rem 0;
            transition: var(--transition-smooth);
        }}
        
        .status-connected {{
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            color: #155724;
        }}
        
        .status-disconnected {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #856404;
        }}
        
        .status-error {{
            background: linear-gradient(135deg, #ffb3ba 0%, #ff9aa2 100%);
            color: #721c24;
        }}
        
        /* License Plate Display */
        .license-plate-display {{
            background: linear-gradient(135deg, #000 0%, #333 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: var(--border-radius-small);
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 1.5rem;
            text-align: center;
            margin: 0.5rem 0;
            border: 3px solid #ffd700;
            box-shadow: var(--shadow-medium);
            letter-spacing: 2px;
        }}
        
        /* Parking Spot Indicators */
        .parking-spot {{
            display: inline-block;
            width: 40px;
            height: 25px;
            margin: 2px;
            border-radius: 4px;
            text-align: center;
            line-height: 25px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }}
        
        .parking-spot-free {{
            background: var(--success-color);
        }}
        
        .parking-spot-occupied {{
            background: var(--error-color);
        }}
        
        /* Button Enhancements */
        .stButton > button {{
            border-radius: var(--border-radius-small);
            border: none;
            font-weight: 500;
            transition: var(--transition-smooth);
            box-shadow: var(--shadow-light);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }}
        
        .stButton > button[kind="primary"] {{
            background: var(--gradient-primary);
        }}
        
        .stButton > button[kind="secondary"] {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #856404;
        }}
        
        /* Selectbox and Input Enhancements */
        .stSelectbox > div > div {{
            border-radius: var(--border-radius-small);
            border: 1px solid #dee2e6;
            transition: var(--transition-smooth);
        }}
        
        .stSelectbox > div > div:focus-within {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }}
        
        .stTextInput > div > div > input {{
            border-radius: var(--border-radius-small);
            border: 1px solid #dee2e6;
            transition: var(--transition-smooth);
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        }}
        
        /* Slider Enhancements */
        .stSlider > div > div > div > div {{
            background: var(--gradient-primary);
        }}
        
        /* Metric Enhancements */
        .metric {{
            background: white;
            padding: 1rem;
            border-radius: var(--border-radius-small);
            box-shadow: var(--shadow-light);
            text-align: center;
            transition: var(--transition-smooth);
        }}
        
        .metric:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-medium);
        }}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background: var(--gradient-primary);
        }}
        
        /* Sidebar Enhancements */
        .css-1d391kg {{
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }}
        
        .sidebar .sidebar-content {{
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }}
        
        /* Dataframe Styling */
        .dataframe {{
            border-radius: var(--border-radius-small);
            overflow: hidden;
            box-shadow: var(--shadow-light);
        }}
        
        /* Expander Styling */
        .streamlit-expanderHeader {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: var(--border-radius-small);
            border: 1px solid #dee2e6;
        }}
        
        /* System Footer */
        .system-footer {{
            background: var(--gradient-primary);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin-top: 2rem;
            color: white;
            text-align: center;
            box-shadow: var(--shadow-medium);
        }}
        
        .system-footer h4 {{
            margin: 0;
            font-weight: 600;
        }}
        
        /* File Upload Area */
        .uploadedFile {{
            border-radius: var(--border-radius-small);
            border: 2px dashed #dee2e6;
            padding: 2rem;
            text-align: center;
            transition: var(--transition-smooth);
        }}
        
        .uploadedFile:hover {{
            border-color: var(--primary-color);
            background-color: rgba(102, 126, 234, 0.05);
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .main-header h1 {{
                font-size: 2rem;
            }}
            
            .main-header p {{
                font-size: 1rem;
            }}
            
            .metric-card {{
                padding: 1rem;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Animation Classes */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        .fade-in-up {{
            animation: fadeInUp 0.6s ease-out;
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        /* Dark mode support (optional) */
        @media (prefers-color-scheme: dark) {{
            .metric-card {{
                background: #2d3748;
                border-color: #4a5568;
                color: #e2e8f0;
            }}
            
            .metric-card h4 {{
                color: #90cdf4;
            }}
        }}
        
        /* Loading Spinner */
        .loading-spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--gradient-primary);
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--secondary-color);
        }}
        </style>
        """
    
    def create_main_header(self, title: str, subtitle: str, logo_data_uri=None) -> str:
        """Create main application header"""
        logo_html = ""
        if logo_data_uri:
            logo_html = f'<div class="header-logo"><img src="{logo_data_uri}" alt="Logo" style="height: 50px; width: auto; max-width: 150px;"></div>'

        text_html = f"""
        <div class="header-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """
        
        return f"""
        <div class="main-header fade-in-up">
            {text_html}
            {logo_html}
        </div>
        """
    
    def create_metric_card(self, title: str, value: str, delta: str = None) -> str:
        """Create metric card component"""
        delta_html = f"<p style='color: {self.success_color}; font-size: 0.9rem; margin-top: 0.5rem;'>{delta}</p>" if delta else ""
        
        return f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: {self.primary_color}; margin: 0.5rem 0;">{value}</p>
            {delta_html}
        </div>
        """
    
    def create_status_indicator(self, status: str, text: str) -> str:
        """Create status indicator component"""
        status_class = f"status-{status}"
        return f"""
        <div class="status-indicator {status_class}">
            {text}
        </div>
        """
    
    def create_license_plate_display(self, plate_text: str) -> str:
        """Create license plate display component"""
        return f"""
        <div class="license-plate-display">
            {plate_text}
        </div>
        """
    
    def create_parking_spot(self, status: str, spot_id: str) -> str:
        """Create parking spot indicator"""
        status_class = f"parking-spot-{status}"
        return f"""
        <div class="parking-spot {status_class}" title="Spot {spot_id}: {status.title()}">
            {spot_id}
        </div>
        """
    
    def create_success_box(self, title: str, content: str) -> str:
        """Create success message box"""
        return f"""
        <div class="success-box fade-in-up">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
        """
    
    def create_alert_box(self, title: str, content: str, alert_type: str = "warning") -> str:
        """Create alert message box"""
        return f"""
        <div class="alert-box fade-in-up">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
        """
    
    def create_loading_spinner(self) -> str:
        """Create loading spinner component"""
        return """
        <div class="loading-spinner"></div>
        """
    
    def create_enhanced_info_box(self, title: str, content: list, box_type: str = "enhanced") -> str:
        """Create enhanced information box"""
        content_html = "".join([f"<p>{item}</p>" for item in content])
        
        if box_type == "enhanced":
            class_name = "enhanced-license-info"
        elif box_type == "model":
            class_name = "model-info"
        elif box_type == "stats":
            class_name = "stats-container"
        else:
            class_name = "metric-card"
        
        return f"""
        <div class="{class_name} fade-in-up">
            <h4>{title}</h4>
            {content_html}
        </div>
        """
    
    def create_results_container(self, title: str, results: dict) -> str:
        """Create results display container"""
        results_html = ""
        for key, value in results.items():
            results_html += f"<p><strong>{key}:</strong> {value}</p>"
        
        return f"""
        <div class="success-box fade-in-up">
            <h4>{title}</h4>
            {results_html}
        </div>
        """


# Create global style manager instance
style_manager = StyleManager()


# Additional utility functions for custom styling
def apply_custom_css(custom_css: str):
    """Apply custom CSS styles"""
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


def create_colored_metric(label: str, value: str, color: str = "#667eea"):
    """Create a colored metric display"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
    ">
        <h4 style="margin: 0; color: {color};">{label}</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold; color: {color};">{value}</p>
    </div>
    """, unsafe_allow_html=True)


def create_gradient_header(title: str, gradient: str = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"):
    """Create a gradient header"""
    st.markdown(f"""
    <div style="
        background: {gradient};
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    ">
        <h2 style="margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def create_info_card(title: str, content: str, icon: str = "ℹ️", color: str = "#17a2b8"):
    """Create an information card"""
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        border-left: 4px solid {color};
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: {color};">{icon} {title}</h4>
        <p style="margin: 0; color: #6c757d; line-height: 1.5;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


# Export all components
__all__ = [
    'StyleManager',
    'style_manager',
    'apply_custom_css',
    'create_colored_metric',
    'create_gradient_header',
    'create_info_card'
]