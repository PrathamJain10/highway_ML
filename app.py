import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
import tempfile
from PIL import Image
import io

class LaneDetectionSystem:
    def _init_(self):
        # Parameters for lane detection
        self.gaussian_blur_kernel = 5
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.roi_vertices_ratio = np.array([
            [0.05, 0.95],  # Bottom left
            [0.45, 0.6],   # Top left
            [0.55, 0.6],   # Top right
            [0.95, 0.95]   # Bottom right
        ])
        self.hough_rho = 2
        self.hough_theta = np.pi/180
        self.hough_threshold = 20
        self.hough_min_line_length = 30
        self.hough_max_line_gap = 100
        
    def process_image(self, image):
        """Process an image to detect lane lines"""
        # Keep a copy of the original image for drawing
        original_image = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # Create a region of interest mask
        roi_vertices = np.array([
            [int(width * self.roi_vertices_ratio[0][0]), int(height * self.roi_vertices_ratio[0][1])],
            [int(width * self.roi_vertices_ratio[1][0]), int(height * self.roi_vertices_ratio[1][1])],
            [int(width * self.roi_vertices_ratio[2][0]), int(height * self.roi_vertices_ratio[2][1])],
            [int(width * self.roi_vertices_ratio[3][0]), int(height * self.roi_vertices_ratio[3][1])]
        ], dtype=np.int32)
        
        # Creating a mask with the region of interest
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_vertices], 255)
        
        # Apply the mask
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        # Separate lines into left and right lanes
        left_lines, right_lines = self.separate_lines(lines, width, height)
        
        # Average and extrapolate the lane lines
        left_line = self.average_slope_intercept(left_lines, height)
        right_line = self.average_slope_intercept(right_lines, height)
        
        # Create an image to draw the lines on
        line_image = np.zeros_like(original_image)
        
        # Draw left and right lane lines
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(line_image, (x1, y1), (x2, y2), [0, 0, 255], 10)
        
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(line_image, (x1, y1), (x2, y2), [0, 0, 255], 10)
        
        # Draw the region of interest polygon
        roi_image = np.zeros_like(original_image)
        cv2.fillPoly(roi_image, [roi_vertices], [0, 255, 0])
        
        # Combine the original image with the line image
        alpha = 0.8
        beta = 1.0
        gamma = 0.0
        result = cv2.addWeighted(original_image, alpha, line_image, beta, gamma)
        
        # Draw the lane area
        lane_area_image = np.zeros_like(original_image)
        if left_line is not None and right_line is not None:
            lane_vertices = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(lane_area_image, [lane_vertices], [0, 255, 0])
            
            # Make the lane area semi-transparent
            lane_result = cv2.addWeighted(result, 0.9, lane_area_image, 0.2, 0)
        else:
            lane_result = result
        
        return {
            'original': cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            'edges': edges,
            'roi': cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB),
            'masked_edges': masked_edges,
            'lines': cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB),
            'result': cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            'lane_area': cv2.cvtColor(lane_area_image, cv2.COLOR_BGR2RGB),
            'lane_result': cv2.cvtColor(lane_result, cv2.COLOR_BGR2RGB)
        }
        
    def separate_lines(self, lines, width, height):
        """Separate lines into left and right lanes based on slope"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter out horizontal lines and lines with too small slopes
            if abs(slope) < 0.3:
                continue
                
            # Categorize lines based on slope and position
            if slope < 0 and x1 < width * 0.7:  # Left lane line
                left_lines.append(line)
            elif slope > 0 and x1 > width * 0.3:  # Right lane line
                right_lines.append(line)
                
        return left_lines, right_lines
    
    def average_slope_intercept(self, lines, height):
        """Calculate average slope and intercept for lines and return extrapolated lane line"""
        if len(lines) == 0:
            return None
            
        x_sum = 0
        y_sum = 0
        m_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:  # Avoid division by zero
                continue
                
            slope = (y2 - y1) / (x2 - x1)
            
            # Calculate intercept
            intercept = y1 - slope * x1
            
            # Add to sums
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += slope
            
        # Calculate averages
        x_avg = x_sum / (len(lines) * 2)
        y_avg = y_sum / (len(lines) * 2)
        m_avg = m_sum / len(lines)
        
        # Calculate intercept using average point
        b = y_avg - m_avg * x_avg
        
        # Calculate the x-coordinate at the bottom of the image
        x_bottom = int((height - b) / m_avg)
        
        # Calculate the x-coordinate at the top of the region of interest
        y_top = int(height * 0.6)  # Matches the ROI top y-coordinate
        x_top = int((y_top - b) / m_avg)
        
        return [x_bottom, height, x_top, y_top]

def main():
    st.set_page_config(page_title="Highway Lane Detection System", layout="wide")
    
    st.title("Highway Lane Detection System")
    st.write("This system detects lane markings in highway images without requiring training data.")
    
    # Initialize the lane detector
    lane_detector = LaneDetectionSystem()
    
    # Create sidebar for parameters
    st.sidebar.header("Detection Parameters")
    
    # Add parameters to sidebar
    lane_detector.gaussian_blur_kernel = st.sidebar.slider(
        "Gaussian Blur Kernel Size", 
        min_value=1, 
        max_value=31, 
        value=lane_detector.gaussian_blur_kernel,
        step=2
    )
    
    lane_detector.canny_low_threshold = st.sidebar.slider(
        "Canny Low Threshold", 
        min_value=10, 
        max_value=200, 
        value=lane_detector.canny_low_threshold
    )
    
    lane_detector.canny_high_threshold = st.sidebar.slider(
        "Canny High Threshold", 
        min_value=20, 
        max_value=300, 
        value=lane_detector.canny_high_threshold
    )
    
    lane_detector.hough_threshold = st.sidebar.slider(
        "Hough Threshold", 
        min_value=5, 
        max_value=100, 
        value=lane_detector.hough_threshold
    )
    
    lane_detector.hough_min_line_length = st.sidebar.slider(
        "Minimum Line Length", 
        min_value=5, 
        max_value=200, 
        value=lane_detector.hough_min_line_length
    )
    
    lane_detector.hough_max_line_gap = st.sidebar.slider(
        "Maximum Line Gap", 
        min_value=5, 
        max_value=300, 
        value=lane_detector.hough_max_line_gap
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a highway image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if image.shape[2] == 3:  # Check if it's a color image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process the image
        try:
            with st.spinner('Processing image...'):
                results = lane_detector.process_image(image)
            
            # Display results
            st.success("Lane detection completed successfully!")
            
            # Display original and final result side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(results['original'], use_column_width=True)
            
            with col2:
                st.subheader("Final Result")
                st.image(results['lane_result'], use_column_width=True)
            
            # Display the intermediate steps with expandable sections
            with st.expander("View Detection Steps"):
                # Create three columns for the middle steps
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Edge Detection")
                    st.image(results['edges'], use_column_width=True)
                
                with col2:
                    st.subheader("Region of Interest")
                    st.image(results['masked_edges'], use_column_width=True)
                
                with col3:
                    st.subheader("Lane Lines")
                    st.image(results['lines'], use_column_width=True)
            
            # Option to download the result
            result_img = Image.fromarray(results['lane_result'])
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Result",
                data=byte_im,
                file_name="lane_detection_result.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        # Display placeholder/instructions when no image is uploaded
        st.info("Please upload a highway image to begin detection.")
        st.markdown("""
        ### How to use:
        1. Upload a highway image using the file uploader above
        2. Adjust the detection parameters in the sidebar if needed
        3. View the detection results
        4. Download the processed image
        """)

if _name_ == "_main_":
    main()