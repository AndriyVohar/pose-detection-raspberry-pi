import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from src.pose_estimation.blazepose_estimator import BlazePoseEstimator
from src.pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from src.pose_estimation.tensorflow_multi_estimator import TensorFlowMultiPoseEstimator
from src.pose_estimation.tensorflow_single_estimator import TensorFlowSinglePoseEstimator
from src.pose_estimation.yolo_estimator import YoloPoseEstimator


class PoseEstimationBenchmark:
    """
    A benchmark utility for comparing different pose estimation implementations.
    This class allows running different pose estimators on the same video and
    comparing their performance in terms of speed, efficiency, and accuracy.
    """
    
    def __init__(self, video_path, output_dir="benchmark_results", show_visualization=False):
        """
        Initialize the benchmark with a video file for testing.
        
        Args:
            video_path: Path to the video file to use for benchmarking
            output_dir: Directory to save benchmark results
            show_visualization: Whether to display real-time visualization during benchmark
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.estimators = {}
        self.results = {}
        self.show_visualization = show_visualization
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_estimator(self, name, estimator_instance):
        """
        Add a pose estimator to the benchmark.
        
        Args:
            name: Name identifier for the estimator
            estimator_instance: An instance of a pose estimator class
        """
        self.estimators[name] = estimator_instance
    
    def run_benchmark(self, num_frames=None):
        """
        Run the benchmark on all added estimators.
        
        Args:
            num_frames: Optional limit on number of frames to process (None = all frames)
            
        Returns:
            Dictionary of benchmark results for each estimator
        """
        # Check if any estimators have been added
        if not self.estimators:
            print("No estimators added to benchmark. Use add_estimator() first.")
            return {}
        
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return {}
            
        # Process each estimator
        for name, estimator in self.estimators.items():
            print(f"\nBenchmarking {name}...")
            metrics = self._benchmark_estimator(name, estimator, num_frames)
            self.results[name] = metrics
            
            # Generate report for this estimator
            self._generate_csv_report(name, metrics)
            
        # Generate comparison charts
        if len(self.estimators) > 1:
            self._generate_comparison_charts()
            
        return self.results
    
    def _benchmark_estimator(self, name, estimator, num_frames=None):
        """
        Run benchmark on a single estimator.
        
        Args:
            name: Name of the estimator
            estimator: Estimator instance
            num_frames: Optional limit on number of frames to process
            
        Returns:
            Dictionary of performance metrics
        """
        # Initialize performance metrics
        metrics = {
            'fps': [],
            'processing_time': [],
            'memory_usage': []
        }
        
        # Check if psutil is available for memory tracking
        has_psutil = False
        psutil = None
        try:
            import psutil
            has_psutil = True
        except ImportError:
            psutil = None
            print("Note: psutil not available, memory usage will not be tracked")
        
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            return metrics
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames if specified
        if num_frames is not None and num_frames > 0:
            frames_to_process = min(num_frames, total_frames)
        else:
            frames_to_process = total_frames
            
        # Setup video writer for saving output using a more compatible approach
        out = None
        
        # Setup window for real-time visualization if requested
        if self.show_visualization:
            window_name = f"Benchmark: {name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            print(f"Displaying real-time visualization in '{window_name}' window")
            print("Press 'q' to quit visualization and continue benchmark")
        
        try:
            # Try different codecs until one works
            codecs = [
                ('M', 'J', 'P', 'G'),  # Motion JPEG
                ('X', 'V', 'I', 'D'),  # XVID
                ('m', 'p', '4', 'v'),  # MP4V
                ('a', 'v', 'c', '1')   # H.264
            ]
            
            for codec in codecs:
                try:
                    # Create fourcc code directly
                    fourcc = cv2.VideoWriter.fourcc(*codec)
                    
                    # Determine file extension based on codec
                    if codec in [('X', 'V', 'I', 'D'), ('M', 'J', 'P', 'G')]:
                        ext = 'avi'
                    else:
                        ext = 'mp4'
                        
                    output_path = os.path.join(self.output_dir, f"{name}_output.{ext}")
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
                    
                    # Test if writer is initialized
                    if out is not None and out.isOpened():
                        print(f"Using {''.join(codec)} codec for output video")
                        break
                    else:
                        if out is not None:
                            out.release()
                        out = None
                except Exception as e:
                    print(f"Failed to use {''.join(codec)} codec: {e}")
                    if out is not None:
                        out.release()
                    out = None
            
            if out is None:
                print("Warning: Could not create video output file with any codec. Results will not be saved as video.")
        except Exception as e:
            print(f"Error creating video writer: {e}")
            out = None
            
        # Process frames
        frame_count = 0
        quit_benchmark = False
        
        # Use tqdm for progress bar (disable if showing visualization to avoid clutter)
        with tqdm(total=frames_to_process, desc=f"Processing", disable=self.show_visualization) as progress_bar:
            while cap.isOpened() and frame_count < frames_to_process and not quit_benchmark:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Measure processing time
                start_time = time.time()
                
                try:
                    # Process frame with the estimator
                    # First try process_frame method
                    if hasattr(estimator, 'process_frame'):
                        result_frame = estimator.process_frame(frame)
                    # Then try _process_frame_impl method (for BasePoseEstimator subclasses)
                    elif hasattr(estimator, '_process_frame_impl'):
                        result_frame = estimator._process_frame_impl(frame)
                    else:
                        print(f"Warning: {name} has no compatible processing method")
                        result_frame = frame
                except Exception as e:
                    print(f"Error processing frame with {name}: {e}")
                    result_frame = frame
                
                # Calculate performance metrics
                end_time = time.time()
                process_time = end_time - start_time
                fps = 1.0 / max(process_time, 0.001)  # Avoid division by zero
                
                # Store metrics
                metrics['fps'].append(fps)
                metrics['processing_time'].append(process_time)
                
                # Track memory usage if psutil is available
                if has_psutil and psutil is not None:
                    try:
                        process = psutil.Process(os.getpid())
                        mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
                        metrics['memory_usage'].append(mem_usage)
                    except Exception as e:
                        print(f"Error tracking memory: {e}")
                        metrics['memory_usage'].append(0)
                else:
                    metrics['memory_usage'].append(0)
                
                # Add performance overlay to the frame
                cv2.putText(result_frame, f"{name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Frame: {frame_count+1}/{frames_to_process}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show visualization if requested
                if self.show_visualization:
                    cv2.imshow(window_name, result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Visualization stopped by user")
                        quit_benchmark = True
                
                # Save processed frame to video if writer is available
                if out is not None and out.isOpened():
                    try:
                        out.write(result_frame)
                    except Exception as e:
                        print(f"Error writing to output video: {e}")
                        out.release()
                        out = None
                
                # Update counters and progress
                frame_count += 1
                progress_bar.update(1)
        
        # Clean up resources
        cap.release()
        if out is not None and out.isOpened():
            out.release()
            
        # Close visualization window if open
        if self.show_visualization:
            cv2.destroyAllWindows()

        # Store summary statistics in the metrics dictionary
        metrics['summary'] = self._calculate_summary_statistics(metrics)
        
        # Print summary
        self._print_summary(name, metrics['summary'])
        
        return metrics

    def _calculate_summary_statistics(self, metrics):
        """
        Calculate summary statistics from the detailed metrics.
        """
        if not metrics['fps']:
            return {
                'avg_fps': 0,
                'min_fps': 0,
                'max_fps': 0,
                'std_fps': 0,
                'avg_processing_time': 0,
                'total_frames': 0
            }

        summary = {
            'avg_fps': np.mean(metrics['fps']),
            'min_fps': np.min(metrics['fps']),
            'max_fps': np.max(metrics['fps']),
            'std_fps': np.std(metrics['fps']),
            'avg_processing_time': np.mean(metrics['processing_time']) * 1000,  # Convert to ms
            'total_frames': len(metrics['fps'])
        }

        # Add memory statistics if values are meaningful
        has_memory_data = False
        for mem_value in metrics['memory_usage']:
            if mem_value > 0:
                has_memory_data = True
                break

        if has_memory_data:
            summary['avg_memory'] = np.mean(metrics['memory_usage'])
            summary['peak_memory'] = np.max(metrics['memory_usage'])

        return summary

    def _print_summary(self, name, summary):
        """
        Print a summary of benchmark results for an estimator.
        """
        print(f"\nResults for {name}:")
        print(f"  Average FPS: {summary['avg_fps']:.2f}")
        print(f"  Min/Max FPS: {summary['min_fps']:.2f} / {summary['max_fps']:.2f}")
        print(f"  Average processing time: {summary['avg_processing_time']:.2f} ms")
        
        if 'avg_memory' in summary:
            print(f"  Average memory usage: {summary['avg_memory']:.2f} MB")
            print(f"  Peak memory usage: {summary['peak_memory']:.2f} MB")
            
        print(f"  Total frames processed: {summary['total_frames']}")
    
    def _generate_csv_report(self, name, metrics):
        """
        Generate CSV report for a single estimator's performance.
        """
        # Summary report
        summary_path = os.path.join(self.output_dir, f"{name}_summary.csv")
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            
            summary = metrics['summary']
            writer.writerow(['Average FPS', f"{summary['avg_fps']:.2f}"])
            writer.writerow(['Min FPS', f"{summary['min_fps']:.2f}"])
            writer.writerow(['Max FPS', f"{summary['max_fps']:.2f}"])
            writer.writerow(['Std Dev FPS', f"{summary['std_fps']:.2f}"])
            writer.writerow(['Avg Processing Time (ms)', f"{summary['avg_processing_time']:.2f}"])
            writer.writerow(['Total Frames Processed', summary['total_frames']])
            
            if 'avg_memory' in summary:
                writer.writerow(['Avg Memory Usage (MB)', f"{summary['avg_memory']:.2f}"])
                writer.writerow(['Peak Memory Usage (MB)', f"{summary['peak_memory']:.2f}"])
        
        # Detailed report (per-frame metrics)
        detailed_path = os.path.join(self.output_dir, f"{name}_detailed.csv")
        with open(detailed_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Frame', 'FPS', 'Processing Time (ms)']
            
            # Check if we have memory data
            has_memory_data = any(val > 0 for val in metrics['memory_usage'])
            if has_memory_data:
                headers.append('Memory Usage (MB)')
                
            writer.writerow(headers)
            
            for i in range(len(metrics['fps'])):
                row = [
                    i+1,
                    f"{metrics['fps'][i]:.2f}",
                    f"{metrics['processing_time'][i] * 1000:.2f}"
                ]
                
                if has_memory_data and i < len(metrics['memory_usage']):
                    row.append(f"{metrics['memory_usage'][i]:.2f}")
                    
                writer.writerow(row)
                
        print(f"  Reports saved to {summary_path} and {detailed_path}")
    
    def _generate_comparison_charts(self):
        """
        Generate comparison charts across all estimators.
        """
        if len(self.results) <= 1:
            return
            
        # Prepare data
        estimator_names = list(self.results.keys())
        
        # Convert numpy values to native Python types for compatibility
        avg_fps_values = [float(self.results[name]['summary']['avg_fps']) for name in estimator_names]
        proc_time_values = [float(self.results[name]['summary']['avg_processing_time']) for name in estimator_names]
        
        # Memory data (if available)
        memory_values = []  # Always start with an empty list
        memory_available = True
        
        # Check for memory data and collect values
        for name in estimator_names:
            summary = self.results[name]['summary']
            if 'avg_memory' in summary:
                # Convert to native Python float to avoid type issues
                try:
                    mem_val = float(summary['avg_memory'])
                    memory_values.append(mem_val)
                except (TypeError, ValueError):
                    memory_available = False
                    break
            else:
                memory_available = False
                break
        
        # Plot FPS comparison
        plt.figure(figsize=(10, 6))
        plt.bar(estimator_names, avg_fps_values, color='skyblue')
        plt.ylabel('Frames per Second')
        plt.title('Average FPS Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fps_chart_path = os.path.join(self.output_dir, 'fps_comparison.png')
        plt.savefig(fps_chart_path)
        plt.close()
        
        # Plot processing time comparison
        plt.figure(figsize=(10, 6))
        plt.bar(estimator_names, proc_time_values, color='salmon')
        plt.ylabel('Processing Time (ms)')
        plt.title('Average Processing Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        time_chart_path = os.path.join(self.output_dir, 'processing_time_comparison.png')
        plt.savefig(time_chart_path)
        plt.close()
        
        # Plot memory usage comparison if available
        if memory_available and len(memory_values) == len(estimator_names):
            plt.figure(figsize=(10, 6))
            plt.bar(estimator_names, memory_values, color='lightgreen')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Average Memory Usage Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            memory_chart_path = os.path.join(self.output_dir, 'memory_usage_comparison.png')
            plt.savefig(memory_chart_path)
            plt.close()
            
        print(f"\nComparison charts saved to {self.output_dir}")
        
    def run_realtime_demo(self, estimator_name=None, duration=60):
        """
        Run a real-time demonstration of the estimator(s) on the video.
        
        Args:
            estimator_name: Specific estimator name to demonstrate (None = all estimators)
            duration: Maximum duration in seconds (default = 60)
        """
        if not self.estimators:
            print("No estimators added to benchmark. Use add_estimator() first.")
            return
            
        if estimator_name is not None and estimator_name not in self.estimators:
            print(f"Estimator '{estimator_name}' not found.")
            return
            
        estimators_to_demo = []
        if estimator_name is not None:
            estimators_to_demo = [(estimator_name, self.estimators[estimator_name])]
        else:
            estimators_to_demo = list(self.estimators.items())
            
        for name, estimator in estimators_to_demo:
            print(f"\nRunning real-time demo for {name}...")
            print("Press 'q' to stop current demo and move to next estimator")
            print("Press 'ESC' to exit all demos")
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {self.video_path}")
                continue
                
            # Create window
            window_name = f"Demo: {name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            # Track performance
            fps_values = []
            start_time = time.time()
            frame_count = 0
            
            # Main demo loop
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    # Loop back to beginning if video ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                # Process frame and measure time
                proc_start = time.time()
                
                try:
                    # Process frame with the estimator
                    if hasattr(estimator, 'process_frame'):
                        result_frame = estimator.process_frame(frame)
                    elif hasattr(estimator, '_process_frame_impl'):
                        result_frame = estimator._process_frame_impl(frame)
                    else:
                        result_frame = frame
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    result_frame = frame
                
                # Calculate FPS
                proc_time = time.time() - proc_start
                current_fps = 1.0 / max(proc_time, 0.001)
                fps_values.append(current_fps)
                
                # Calculate rolling average FPS (last 30 frames)
                avg_fps = sum(fps_values[-30:]) / min(len(fps_values), 30)
                
                # Add information overlay
                cv2.putText(result_frame, f"{name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Avg FPS: {avg_fps:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Time: {int(time.time() - start_time)}s", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display result
                cv2.imshow(window_name, result_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"Stopping demo for {name}")
                    break
                elif key == 27:  # ESC key
                    print("Exiting all demos")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                    
                frame_count += 1
            
            # Cleanup for this estimator
            cap.release()
            cv2.destroyWindow(window_name)
            
            # Print summary for this estimator
            if fps_values:
                print(f"Demo summary for {name}:")
                print(f"  Average FPS: {sum(fps_values) / len(fps_values):.2f}")
                print(f"  Min/Max FPS: {min(fps_values):.2f} / {max(fps_values):.2f}")
                print(f"  Total frames processed: {frame_count}")
        
        print("\nReal-time demo completed")


# Example usage
if __name__ == "__main__":
    # Create benchmark instance with visualization enabled
    benchmark = PoseEstimationBenchmark(
        video_path="../../videos/laught.mp4",
        output_dir="../../benchmark_results",
        show_visualization=True  # Enable real-time visualization
    )
    
    # Add estimators
    benchmark.add_estimator("MediaPipe", MediaPipePoseEstimator())
    benchmark.add_estimator("BlazePose", BlazePoseEstimator())
    benchmark.add_estimator("TensorFlow Single", TensorFlowSinglePoseEstimator())
    benchmark.add_estimator("TensorFlow Multi", TensorFlowMultiPoseEstimator())
    benchmark.add_estimator("YOLO", YoloPoseEstimator())

    # Option 1: Run benchmark with visualization
    # results = benchmark.run_benchmark(num_frames=100)
    
    # Option 2: Run real-time demo (better for visualization)
    benchmark.run_realtime_demo(duration=30)  # Run a 30-second demo for each estimator
    
    # Option 3: Run real-time demo for a specific estimator
    # benchmark.run_realtime_demo(estimator_name="YOLO", duration=30)

