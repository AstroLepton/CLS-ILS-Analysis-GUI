#!/usr/bin/env python3
"""
CLS & ILS Analysis GUI
A GUI application for analyzing BiVO4 samples using both CLS and ILS methods.
Features include file loading, data visualization, concentration prediction, and CSV export.

names and expected concntrations are default set to 6.67,2,1...change name and concentration accrodingly for other samples 
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from pathlib import Path
import glob
from numpy.linalg import inv
import threading
import os
from datetime import datetime
########################################################################################################
class BiVO4CLSAnalysis:
    """Classical Least Squares analysis for BiVO4 samples."""
    
    def __init__(self):
        self.wavelengths = np.arange(400, 710, 10)  # 400-700 nm in 10 nm steps
        self.known_concentrations = {
            'A': 6.67,  # 6.67% concentration
            'B': 2.0,   # 2% concentration
            'C': 1.0    # 1% concentration
        }
        self.spectra_data = {}
        self.background_spectrum = None
        self.calibration_matrix = None
        self.results = {}
        self.data_folder = None
        
    def load_spectra_data(self, data_folder):
        """Load all CSV files and extract reflectance spectra data."""
        self.data_folder = Path(data_folder)
        self.spectra_data = {}
        
        # Find all CSV files in the folder
        csv_files = list(self.data_folder.glob("*.csv"))
        
        for file_path in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(file_path, sep=';', skiprows=3)
                
                # Extract sample name from filename
                sample_name = file_path.stem
                
                # Extract reflectance values (R400 nm to R700 nm)
                reflectance_cols = [col for col in df.columns if col.startswith('R') and 'nm' in col]
                reflectance_values = df[reflectance_cols].iloc[0].values.astype(float)
                
                # Store the data
                self.spectra_data[sample_name] = {
                    'reflectance': reflectance_values,
                    'wavelengths': self.wavelengths,
                    'file_path': file_path
                }
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return len(self.spectra_data)
        
    def create_calibration_matrix(self):
        """Create the calibration matrix E from known concentrations with background subtraction."""
        # Get background spectrum
        if 'BG www' in self.spectra_data:
            self.background_spectrum = self.spectra_data['BG www']['reflectance']
        else:
            self.background_spectrum = np.zeros(31)  # Fallback to zeros
        
        # Get the average spectra for each concentration level (background subtracted)
        concentration_spectra = {}
        
        for conc_level in ['A', 'B', 'C']:
            # Find all samples for this concentration level
            samples = [name for name in self.spectra_data.keys() 
                      if f'{conc_level} H' in name and not name.startswith('BG')]
            
            if samples:
                # Calculate average spectrum for this concentration (background subtracted)
                spectra_list = []
                for sample in samples:
                    raw_spectrum = self.spectra_data[sample]['reflectance']
                    # Subtract background
                    corrected_spectrum = raw_spectrum - self.background_spectrum
                    spectra_list.append(corrected_spectrum)
                
                avg_spectrum = np.mean(spectra_list, axis=0)
                concentration_spectra[conc_level] = avg_spectrum
        
        # Create calibration matrix E (n_components x n_wavelengths)
        E_matrix = np.array([
            concentration_spectra['A'],
            concentration_spectra['B'], 
            concentration_spectra['C']
        ])
        
        # Add background correction (column of 1s)
        background = np.ones(E_matrix.shape[1])
        E_matrix = np.vstack([background, E_matrix])
        
        self.calibration_matrix = E_matrix
        return E_matrix
        
    def classical_least_squares(self, spectrum):
        """Apply Classical Least Squares (CLS) method."""
        E = self.calibration_matrix
        
        # Calculate (E^T * E)^(-1) * E^T
        E_T = E.T
        E_T_E = E @ E_T
        E_T_E_inv = inv(E_T_E)
        pseudo_inverse = E_T_E_inv @ E
        
        # Calculate concentrations
        concentrations = pseudo_inverse @ spectrum
        
        return concentrations
        
    def analyze_all_samples(self):
        """Analyze all samples using CLS method with background subtraction."""
        self.results = {}
        
        for sample_name, data in self.spectra_data.items():
            if sample_name.startswith('BG'):
                continue  # Skip background samples
                
            # Subtract background from the spectrum
            raw_spectrum = data['reflectance']
            corrected_spectrum = raw_spectrum - self.background_spectrum
            
            # Apply CLS analysis
            concentrations = self.classical_least_squares(corrected_spectrum)
            
            # Store results
            self.results[sample_name] = {
                'background': concentrations[0],
                'A_concentration': concentrations[1],
                'B_concentration': concentrations[2], 
                'C_concentration': concentrations[3],
                'total_concentration': np.sum(concentrations[1:4]),
                'raw_spectrum': raw_spectrum,
                'corrected_spectrum': corrected_spectrum
            }
            
    def calculate_absorbance(self, reflectance):
        """Calculate absorbance from reflectance."""
        reflectance = np.clip(reflectance, 1e-10, 1.0)
        absorbance_direct = -np.log10(reflectance)
        kubelka_munk = (1 - reflectance)**2 / (2 * reflectance)
        return absorbance_direct, kubelka_munk
        
    def get_results_dataframe(self):
        """Convert results to pandas DataFrame for CSV export."""
        data = []
        for sample_name, result in self.results.items():
            data.append({
                'Sample': sample_name,
                'A_Concentration': result['A_concentration'],
                'B_Concentration': result['B_concentration'],
                'C_Concentration': result['C_concentration'],
                'Total_Concentration': result['total_concentration'],
                'Background': result['background']
            })
        return pd.DataFrame(data)
##########################################################################################################
class BiVO4ILSAnalysis:
    """Inverse Least Squares analysis for BiVO4 samples."""
    
    def __init__(self):
        self.wavelengths = np.arange(400, 710, 10)  # 400-700 nm in 10 nm steps
        self.known_concentrations = {
            'A': 6.67,  # 6.67% concentration
            'B': 2.0,   # 2% concentration
            'C': 1.0    # 1% concentration
        }
        self.spectra_data = {}
        self.background_spectrum = None
        self.calibration_spectra = {}
        self.calibration_concentrations = {}
        self.ils_model = None
        self.results = {}
        self.data_folder = None
        
    def load_spectra_data(self, data_folder):
        """Load all CSV files and extract reflectance spectra data."""
        self.data_folder = Path(data_folder)
        self.spectra_data = {}
        
        # Find all CSV files in the folder
        csv_files = list(self.data_folder.glob("*.csv"))
        
        for file_path in csv_files:
            try:
                # Read CSV file
                df = pd.read_csv(file_path, sep=';', skiprows=3)
                
                # Extract sample name from filename
                sample_name = file_path.stem
                
                # Extract reflectance values (R400 nm to R700 nm)
                reflectance_cols = [col for col in df.columns if col.startswith('R') and 'nm' in col]
                reflectance_values = df[reflectance_cols].iloc[0].values.astype(float)
                
                # Store the data
                self.spectra_data[sample_name] = {
                    'reflectance': reflectance_values,
                    'wavelengths': self.wavelengths,
                    'file_path': file_path
                }
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return len(self.spectra_data)
        
    def build_ils_model(self):
        """Build the ILS calibration model using first sample from each group."""
        # Get background spectrum
        if 'BG www' in self.spectra_data:
            self.background_spectrum = self.spectra_data['BG www']['reflectance']
        else:
            self.background_spectrum = np.zeros(31)  # Fallback to zeros
        
        # Get first sample from each concentration level for calibration
        self.calibration_spectra = {}
        self.calibration_concentrations = {}
        
        for conc_level in ['A', 'B', 'C']:
            # Find first sample for this concentration level
            for name in self.spectra_data.keys():
                if f'{conc_level} H' in name and not name.startswith('BG'):
                    if conc_level not in self.calibration_spectra:
                        raw_spectrum = self.spectra_data[name]['reflectance']
                        corrected_spectrum = raw_spectrum - self.background_spectrum
                        self.calibration_spectra[conc_level] = corrected_spectrum
                        self.calibration_concentrations[conc_level] = self.known_concentrations[conc_level]
                        break
        
        # Create calibration matrix A (samples x wavelengths)
        calibration_matrix = np.array([
            self.calibration_spectra['A'],
            self.calibration_spectra['B'],
            self.calibration_spectra['C']
        ])
        
        # ILS model: P = (A * A^T)^(-1) * A
        # where A is calibration matrix (samples x wavelengths)
        A_T = calibration_matrix.T  # (wavelengths x samples)
        AAT = calibration_matrix @ A_T  # (samples x samples)
        AAT_inv = inv(AAT)
        self.ils_model = AAT_inv @ calibration_matrix  # (samples x wavelengths)
        
        return calibration_matrix
        
    def analyze_all_samples(self):
        """Analyze all samples using ILS method with background subtraction."""
        self.results = {}
        
        for sample_name, data in self.spectra_data.items():
            if sample_name.startswith('BG'):
                continue  # Skip background samples
                
            # Subtract background from the spectrum
            raw_spectrum = data['reflectance']
            corrected_spectrum = raw_spectrum - self.background_spectrum
            
            # Apply ILS analysis: C = P * Y (where Y is the spectrum)
            spectrum_col = corrected_spectrum.reshape(-1, 1)
            concentrations = self.ils_model @ spectrum_col
            
            # Determine true concentration based on sample name
            true_concentration = None
            for conc_level, conc_value in self.known_concentrations.items():
                if f'{conc_level} H' in sample_name:
                    true_concentration = conc_value
                    break
            
            # Calculate predicted concentration (weighted sum)
            predicted_conc = concentrations[0, 0] if concentrations.ndim > 1 else concentrations[0]
            
            # Store results
            self.results[sample_name] = {
                'predicted_concentration': predicted_conc,
                'true_concentration': true_concentration,
                'A_weight': concentrations[0, 0] if concentrations.ndim > 1 else concentrations[0],
                'B_weight': concentrations[1, 0] if concentrations.ndim > 1 else concentrations[1],
                'C_weight': concentrations[2, 0] if concentrations.ndim > 1 else concentrations[2],
                'raw_spectrum': raw_spectrum,
                'corrected_spectrum': corrected_spectrum,
                'error': abs(predicted_conc - true_concentration) if true_concentration else 0,
                'relative_error': abs(predicted_conc - true_concentration) / true_concentration * 100 if true_concentration else 0
            }
            
    def calculate_absorbance(self, reflectance):
        """Calculate absorbance from reflectance."""
        reflectance = np.clip(reflectance, 1e-10, 1.0)
        absorbance_direct = -np.log10(reflectance)
        kubelka_munk = (1 - reflectance)**2 / (2 * reflectance)
        return absorbance_direct, kubelka_munk
        
    def get_results_dataframe(self):
        """Convert results to pandas DataFrame for CSV export."""
        data = []
        for sample_name, result in self.results.items():
            data.append({
                'Sample': sample_name,
                'Predicted_Concentration': result['predicted_concentration'],
                'True_Concentration': result['true_concentration'],
                'Absolute_Error': result['error'],
                'Relative_Error_Percent': result['relative_error'],
                'A_Weight': result['A_weight'],
                'B_Weight': result['B_weight'],
                'C_Weight': result['C_weight']
            })
        return pd.DataFrame(data)

class CLSAnalysisGUI:
    """Main GUI application for CLS & ILS analysis."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CLS & ILS Analysis - BiVO4 Samples")
        self.root.geometry("1400x900")
        
        # Initialize analysis objects
        self.cls_analysis = BiVO4CLSAnalysis()
        self.ils_analysis = BiVO4ILSAnalysis()
        
        # Create GUI components
        self.create_widgets()
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create CLS tabs
        self.create_control_tab()
        self.create_spectra_tab()
        self.create_results_tab()
        self.create_plots_tab()
        
        # Create ILS tabs
        self.create_ils_control_tab()
        self.create_ils_spectra_tab()
        self.create_ils_results_tab()
        self.create_ils_plots_tab()
        
    def create_control_tab(self):
        """Create the CLS control/input tab."""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="CLS Control")
        
        # File selection frame
        file_frame = ttk.LabelFrame(control_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.folder_path = tk.StringVar()
        folder_entry = ttk.Entry(file_frame, textvariable=self.folder_path, width=60)
        folder_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="Browse Folder", command=self.browse_folder)
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        load_btn = ttk.Button(file_frame, text="Load Data", command=self.load_data)
        load_btn.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Controls", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        analyze_btn = ttk.Button(analysis_frame, text="Run CLS Analysis", command=self.run_analysis)
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = ttk.Button(analysis_frame, text="Export Results", command=self.export_results)
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(analysis_frame, text="Clear Results", command=self.clear_results)
        clear_btn.pack(side=tk.LEFT)
        
        # Data info frame
        info_frame = ttk.LabelFrame(control_frame, text="Data Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
    def create_spectra_tab(self):
        """Create the CLS spectra visualization tab."""
        spectra_frame = ttk.Frame(self.notebook)
        self.notebook.add(spectra_frame, text="CLS Spectra")
        
        # Create matplotlib figure
        self.spectra_fig = Figure(figsize=(12, 8), dpi=100)
        self.spectra_canvas = FigureCanvasTkAgg(self.spectra_fig, spectra_frame)
        self.spectra_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.spectra_toolbar = NavigationToolbar2Tk(self.spectra_canvas, spectra_frame)
        self.spectra_toolbar.update()
        
    def create_results_tab(self):
        """Create the CLS results display tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="CLS Results")
        
        # Create treeview for results
        columns = ('Sample', 'A_Conc', 'B_Conc', 'C_Conc', 'Total_Conc', 'Background')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, anchor='center')
        
        # Add scrollbars
        results_scroll_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=results_scroll_y.set, xscrollcommand=results_scroll_x.set)
        
        # Pack widgets
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_plots_tab(self):
        """Create the CLS plots tab."""
        plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(plots_frame, text="CLS Plots")
        
        # Create matplotlib figure for plots
        self.plots_fig = Figure(figsize=(12, 10), dpi=100)
        self.plots_canvas = FigureCanvasTkAgg(self.plots_fig, plots_frame)
        self.plots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.plots_toolbar = NavigationToolbar2Tk(self.plots_canvas, plots_frame)
        self.plots_toolbar.update()
        
    def browse_folder(self):
        """Browse for data folder."""
        folder = filedialog.askdirectory(title="Select folder containing CSV files")
        if folder:
            self.folder_path.set(folder)
            
    def log_message(self, message):
        """Add message to status log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_data(self):
        """Load data from selected folder for CLS analysis."""
        folder = self.folder_path.get()
        if not folder:
            messagebox.showerror("Error", "Please select a folder first!")
            return
            
        try:
            self.log_message("Loading data for CLS analysis...")
            num_files = self.cls_analysis.load_spectra_data(folder)
            self.log_message(f"Successfully loaded {num_files} CSV files")
            
            # Update data info
            self.update_data_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.log_message(f"Error: {str(e)}")
            
    def update_data_info(self):
        """Update data information display."""
        self.info_text.delete(1.0, tk.END)
        
        if not self.cls_analysis.spectra_data:
            self.info_text.insert(tk.END, "No data loaded.")
            return
            
        # Display loaded samples
        self.info_text.insert(tk.END, "Loaded Samples:\n")
        self.info_text.insert(tk.END, "=" * 50 + "\n")
        
        for sample_name, data in self.cls_analysis.spectra_data.items():
            self.info_text.insert(tk.END, f"• {sample_name}\n")
            
        self.info_text.insert(tk.END, f"\nTotal samples: {len(self.cls_analysis.spectra_data)}\n")
        self.info_text.insert(tk.END, f"Wavelength range: {self.cls_analysis.wavelengths[0]}-{self.cls_analysis.wavelengths[-1]} nm\n")
        self.info_text.insert(tk.END, f"Number of wavelengths: {len(self.cls_analysis.wavelengths)}\n")
        
    def run_analysis(self):
        """Run CLS analysis in a separate thread."""
        if not self.cls_analysis.spectra_data:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        def analysis_thread():
            try:
                self.log_message("Creating CLS calibration matrix...")
                self.cls_analysis.create_calibration_matrix()
                self.log_message("CLS calibration matrix created successfully")
                
                self.log_message("Running CLS analysis...")
                self.cls_analysis.analyze_all_samples()
                self.log_message(f"CLS analysis completed for {len(self.cls_analysis.results)} samples")
                
                # Update GUI in main thread
                self.root.after(0, self.update_results_display)
                self.root.after(0, self.update_plots)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"CLS analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.log_message(f"CLS analysis error: {str(e)}"))
                
        # Start analysis in separate thread
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
        
    def update_results_display(self):
        """Update the CLS results treeview."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Add results
        for sample_name, result in self.cls_analysis.results.items():
            values = (
                sample_name,
                f"{result['A_concentration']:.4f}",
                f"{result['B_concentration']:.4f}",
                f"{result['C_concentration']:.4f}",
                f"{result['total_concentration']:.4f}",
                f"{result['background']:.4f}"
            )
            self.results_tree.insert('', tk.END, values=values)
            
    def update_plots(self):
        """Update all plots."""
        self.plot_spectra()
        self.plot_analysis_results()
        
    def plot_spectra(self):
        """Plot CLS reflectance and absorbance spectra."""
        self.spectra_fig.clear()
        
        if not self.cls_analysis.spectra_data:
            return
            
        # Create subplots
        ax1 = self.spectra_fig.add_subplot(2, 2, 1)
        ax2 = self.spectra_fig.add_subplot(2, 2, 2)
        ax3 = self.spectra_fig.add_subplot(2, 2, 3)
        ax4 = self.spectra_fig.add_subplot(2, 2, 4)
        
        colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'BG': 'gray'}
        
        # Plot 1: Raw reflectance spectra
        for sample_name, data in self.cls_analysis.spectra_data.items():
            color = 'gray'
            for conc in colors:
                if sample_name.startswith(conc):
                    color = colors[conc]
                    break
                    
            ax1.plot(data['wavelengths'], data['reflectance'], 
                    color=color, alpha=0.7, label=sample_name, linewidth=2)
            
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Reflectance')
        ax1.set_title('Raw Reflectance Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Background-subtracted spectra
        for sample_name, result in self.cls_analysis.results.items():
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            ax2.plot(self.cls_analysis.wavelengths, result['corrected_spectrum'], 
                    color=color, alpha=0.7, label=sample_name, linewidth=2)
            
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Background-Subtracted Reflectance')
        ax2.set_title('Background-Subtracted Spectra')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Absorbance spectra
        for sample_name, data in self.cls_analysis.spectra_data.items():
            if sample_name.startswith('BG'):
                continue
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            absorbance, _ = self.cls_analysis.calculate_absorbance(data['reflectance'])
            ax3.plot(data['wavelengths'], absorbance, 
                    color=color, alpha=0.7, label=sample_name, linewidth=2)
            
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Absorbance (-log₁₀R)')
        ax3.set_title('Absorbance Spectra')
        ax3.grid(True, alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 4: Kubelka-Munk function
        for sample_name, data in self.cls_analysis.spectra_data.items():
            if sample_name.startswith('BG'):
                continue
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            _, kubelka_munk = self.cls_analysis.calculate_absorbance(data['reflectance'])
            ax4.plot(data['wavelengths'], kubelka_munk, 
                    color=color, alpha=0.7, label=sample_name, linewidth=2)
            
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('F(R) = (1-R)²/(2R)')
        ax4.set_title('Kubelka-Munk Function')
        ax4.grid(True, alpha=0.3)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.spectra_fig.tight_layout()
        self.spectra_canvas.draw()
        
    def plot_analysis_results(self):
        """Plot CLS concentration analysis results."""
        self.plots_fig.clear()
        
        if not self.cls_analysis.results:
            return
            
        # Create subplots
        ax1 = self.plots_fig.add_subplot(2, 2, 1)
        ax2 = self.plots_fig.add_subplot(2, 2, 2)
        ax3 = self.plots_fig.add_subplot(2, 2, 3)
        ax4 = self.plots_fig.add_subplot(2, 2, 4)
        
        # Extract data for plotting
        samples = list(self.cls_analysis.results.keys())
        A_conc = [self.cls_analysis.results[s]['A_concentration'] for s in samples]
        B_conc = [self.cls_analysis.results[s]['B_concentration'] for s in samples]
        C_conc = [self.cls_analysis.results[s]['C_concentration'] for s in samples]
        total_conc = [self.cls_analysis.results[s]['total_concentration'] for s in samples]
        
        # Plot A concentrations
        bars1 = ax1.bar(range(len(samples)), A_conc, color='red', alpha=0.7)
        ax1.set_title('A Component Concentrations')
        ax1.set_ylabel('Concentration')
        ax1.set_xticks(range(len(samples)))
        ax1.set_xticklabels(samples, rotation=45, ha='right')
        ax1.axhline(y=6.67, color='red', linestyle='--', alpha=0.5, label='Expected (6.67%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot B concentrations
        bars2 = ax2.bar(range(len(samples)), B_conc, color='blue', alpha=0.7)
        ax2.set_title('B Component Concentrations')
        ax2.set_ylabel('Concentration')
        ax2.set_xticks(range(len(samples)))
        ax2.set_xticklabels(samples, rotation=45, ha='right')
        ax2.axhline(y=2.0, color='blue', linestyle='--', alpha=0.5, label='Expected (2%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot C concentrations
        bars3 = ax3.bar(range(len(samples)), C_conc, color='green', alpha=0.7)
        ax3.set_title('C Component Concentrations')
        ax3.set_ylabel('Concentration')
        ax3.set_xticks(range(len(samples)))
        ax3.set_xticklabels(samples, rotation=45, ha='right')
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Expected (1%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot total concentrations
        bars4 = ax4.bar(range(len(samples)), total_conc, color='purple', alpha=0.7)
        ax4.set_title('Total Concentrations')
        ax4.set_ylabel('Total Concentration')
        ax4.set_xticks(range(len(samples)))
        ax4.set_xticklabels(samples, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        self.plots_fig.tight_layout()
        self.plots_canvas.draw()
        
    def export_results(self):
        """Export CLS results to CSV file."""
        if not self.cls_analysis.results:
            messagebox.showerror("Error", "No CLS results to export! Run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save CLS Results As"
        )
        
        if filename:
            try:
                df = self.cls_analysis.get_results_dataframe()
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"CLS results exported to {filename}")
                self.log_message(f"CLS results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CLS results: {str(e)}")
                
    def clear_results(self):
        """Clear all CLS results and plots."""
        self.cls_analysis.results = {}
        self.cls_analysis.calibration_matrix = None
        
        # Clear displays
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        self.spectra_fig.clear()
        self.spectra_canvas.draw()
        
        self.plots_fig.clear()
        self.plots_canvas.draw()
        
        self.log_message("CLS results cleared")
###############################################################################################################################################    
    # ===== ILS METHODS =====
    
    def create_ils_control_tab(self):
        """Create the ILS control/input tab."""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="ILS Control")
        
        # File selection frame
        file_frame = ttk.LabelFrame(control_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ils_folder_path = tk.StringVar()
        folder_entry = ttk.Entry(file_frame, textvariable=self.ils_folder_path, width=60)
        folder_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="Browse Folder", command=self.ils_browse_folder)
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        load_btn = ttk.Button(file_frame, text="Load Data", command=self.ils_load_data)
        load_btn.pack(side=tk.LEFT)
        
        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ils_status_text = scrolledtext.ScrolledText(status_frame, height=8, width=80)
        self.ils_status_text.pack(fill=tk.BOTH, expand=True)
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Controls", padding=10)
        analysis_frame.pack(fill=tk.X, padx=10, pady=5)
        
        analyze_btn = ttk.Button(analysis_frame, text="Run ILS Analysis", command=self.run_ils_analysis)
        analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        export_btn = ttk.Button(analysis_frame, text="Export Results", command=self.export_ils_results)
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = ttk.Button(analysis_frame, text="Clear Results", command=self.clear_ils_results)
        clear_btn.pack(side=tk.LEFT)
        
        # Data info frame
        info_frame = ttk.LabelFrame(control_frame, text="Data Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ils_info_text = scrolledtext.ScrolledText(info_frame, height=10, width=80)
        self.ils_info_text.pack(fill=tk.BOTH, expand=True)
    
    def create_ils_spectra_tab(self):
        """Create the ILS spectra visualization tab."""
        spectra_frame = ttk.Frame(self.notebook)
        self.notebook.add(spectra_frame, text="ILS Spectra")
        
        # Create matplotlib figure
        self.ils_spectra_fig = Figure(figsize=(12, 8), dpi=100)
        self.ils_spectra_canvas = FigureCanvasTkAgg(self.ils_spectra_fig, spectra_frame)
        self.ils_spectra_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.ils_spectra_toolbar = NavigationToolbar2Tk(self.ils_spectra_canvas, spectra_frame)
        self.ils_spectra_toolbar.update()
        
    def create_ils_results_tab(self):
        """Create the ILS results display tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="ILS Results")
        
        # Create treeview for results
        columns = ('Sample', 'Predicted', 'True', 'Error', 'Rel_Error%', 'A_Weight', 'B_Weight', 'C_Weight')
        self.ils_results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.ils_results_tree.heading(col, text=col)
            self.ils_results_tree.column(col, width=100, anchor='center')
        
        # Add scrollbars
        results_scroll_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.ils_results_tree.yview)
        results_scroll_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.ils_results_tree.xview)
        self.ils_results_tree.configure(yscrollcommand=results_scroll_y.set, xscrollcommand=results_scroll_x.set)
        
        # Pack widgets
        self.ils_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_ils_plots_tab(self):
        """Create the ILS plots tab."""
        plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(plots_frame, text="ILS Plots")
        
        # Create matplotlib figure for plots
        self.ils_plots_fig = Figure(figsize=(12, 10), dpi=100)
        self.ils_plots_canvas = FigureCanvasTkAgg(self.ils_plots_fig, plots_frame)
        self.ils_plots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.ils_plots_toolbar = NavigationToolbar2Tk(self.ils_plots_canvas, plots_frame)
        self.ils_plots_toolbar.update()
        
    def ils_browse_folder(self):
        """Browse for data folder for ILS."""
        folder = filedialog.askdirectory(title="Select folder containing CSV files for ILS")
        if folder:
            self.ils_folder_path.set(folder)
            
    def ils_log_message(self, message):
        """Add message to ILS status log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.ils_status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.ils_status_text.see(tk.END)
        self.root.update_idletasks()
        
    def ils_load_data(self):
        """Load data from selected folder for ILS analysis."""
        folder = self.ils_folder_path.get()
        if not folder:
            messagebox.showerror("Error", "Please select a folder first!")
            return
            
        try:
            self.ils_log_message("Loading data for ILS analysis...")
            num_files = self.ils_analysis.load_spectra_data(folder)
            self.ils_log_message(f"Successfully loaded {num_files} CSV files")
            
            # Update data info
            self.ils_update_data_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.ils_log_message(f"Error: {str(e)}")
            
    def ils_update_data_info(self):
        """Update ILS data information display."""
        self.ils_info_text.delete(1.0, tk.END)
        
        if not self.ils_analysis.spectra_data:
            self.ils_info_text.insert(tk.END, "No data loaded.")
            return
            
        # Display loaded samples
        self.ils_info_text.insert(tk.END, "Loaded Samples:\n")
        self.ils_info_text.insert(tk.END, "=" * 50 + "\n")
        
        for sample_name, data in self.ils_analysis.spectra_data.items():
            self.ils_info_text.insert(tk.END, f"• {sample_name}\n")
            
        self.ils_info_text.insert(tk.END, f"\nTotal samples: {len(self.ils_analysis.spectra_data)}\n")
        self.ils_info_text.insert(tk.END, f"Wavelength range: {self.ils_analysis.wavelengths[0]}-{self.ils_analysis.wavelengths[-1]} nm\n")
        self.ils_info_text.insert(tk.END, f"Number of wavelengths: {len(self.ils_analysis.wavelengths)}\n")
        
    def run_ils_analysis(self):
        """Run ILS analysis in a separate thread."""
        if not self.ils_analysis.spectra_data:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        def analysis_thread():
            try:
                self.ils_log_message("Building ILS calibration model...")
                self.ils_analysis.build_ils_model()
                self.ils_log_message("ILS model built successfully")
                
                self.ils_log_message("Running ILS analysis...")
                self.ils_analysis.analyze_all_samples()
                self.ils_log_message(f"ILS analysis completed for {len(self.ils_analysis.results)} samples")
                
                # Update GUI in main thread
                self.root.after(0, self.ils_update_results_display)
                self.root.after(0, self.ils_update_plots)
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"ILS analysis failed: {str(e)}"))
                self.root.after(0, lambda: self.ils_log_message(f"ILS analysis error: {str(e)}"))
                
        # Start analysis in separate thread
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
        
    def ils_update_results_display(self):
        """Update the ILS results treeview."""
        # Clear existing items
        for item in self.ils_results_tree.get_children():
            self.ils_results_tree.delete(item)
            
        # Add results
        for sample_name, result in self.ils_analysis.results.items():
            values = (
                sample_name,
                f"{result['predicted_concentration']:.4f}",
                f"{result['true_concentration']:.4f}" if result['true_concentration'] else "N/A",
                f"{result['error']:.4f}",
                f"{result['relative_error']:.2f}",
                f"{result['A_weight']:.4f}",
                f"{result['B_weight']:.4f}",
                f"{result['C_weight']:.4f}"
            )
            self.ils_results_tree.insert('', tk.END, values=values)
            
    def ils_update_plots(self):
        """Update all ILS plots."""
        self.ils_plot_spectra()
        self.ils_plot_analysis_results()
        
    def ils_plot_spectra(self):
        """Plot ILS reflectance and absorbance spectra."""
        self.ils_spectra_fig.clear()
        
        if not self.ils_analysis.spectra_data:
            return
            
        # Create subplots
        ax1 = self.ils_spectra_fig.add_subplot(2, 2, 1)
        ax2 = self.ils_spectra_fig.add_subplot(2, 2, 2)
        ax3 = self.ils_spectra_fig.add_subplot(2, 2, 3)
        ax4 = self.ils_spectra_fig.add_subplot(2, 2, 4)
        
        colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'BG': 'gray'}
        
        # Plot 1: Raw reflectance spectra
        for sample_name, data in self.ils_analysis.spectra_data.items():
            color = 'gray'
            for conc in colors:
                if sample_name.startswith(conc):
                    color = colors[conc]
                    break
                    
            ax1.plot(data['wavelengths'], data['reflectance'], 
                    color=color, alpha=0.7, linewidth=2)
            
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Reflectance')
        ax1.set_title('Raw Reflectance Spectra')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Background-subtracted spectra
        for sample_name, result in self.ils_analysis.results.items():
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            ax2.plot(self.ils_analysis.wavelengths, result['corrected_spectrum'], 
                    color=color, alpha=0.7, linewidth=2)
            
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Background-Subtracted Reflectance')
        ax2.set_title('Background-Subtracted Spectra')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Absorbance spectra
        for sample_name, data in self.ils_analysis.spectra_data.items():
            if sample_name.startswith('BG'):
                continue
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            absorbance, _ = self.ils_analysis.calculate_absorbance(data['reflectance'])
            ax3.plot(data['wavelengths'], absorbance, 
                    color=color, alpha=0.7, linewidth=2)
            
        ax3.set_xlabel('Wavelength (nm)')
        ax3.set_ylabel('Absorbance (-log₁₀R)')
        ax3.set_title('Absorbance Spectra')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Kubelka-Munk function
        for sample_name, data in self.ils_analysis.spectra_data.items():
            if sample_name.startswith('BG'):
                continue
            color = 'gray'
            for conc in ['A', 'B', 'C']:
                if f'{conc} H' in sample_name:
                    color = colors[conc]
                    break
                    
            _, kubelka_munk = self.ils_analysis.calculate_absorbance(data['reflectance'])
            ax4.plot(data['wavelengths'], kubelka_munk, 
                    color=color, alpha=0.7, linewidth=2)
            
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('F(R) = (1-R)²/(2R)')
        ax4.set_title('Kubelka-Munk Function')
        ax4.grid(True, alpha=0.3)
        
        self.ils_spectra_fig.tight_layout()
        self.ils_spectra_canvas.draw()
        
    def ils_plot_analysis_results(self):
        """Plot ILS concentration analysis results."""
        self.ils_plots_fig.clear()
        
        if not self.ils_analysis.results:
            return
            
        # Create subplots
        ax1 = self.ils_plots_fig.add_subplot(2, 2, 1)
        ax2 = self.ils_plots_fig.add_subplot(2, 2, 2)
        ax3 = self.ils_plots_fig.add_subplot(2, 2, 3)
        ax4 = self.ils_plots_fig.add_subplot(2, 2, 4)
        
        # Extract data for plotting
        samples = list(self.ils_analysis.results.keys())
        predicted = [self.ils_analysis.results[s]['predicted_concentration'] for s in samples]
        
        # Get samples with true concentrations
        samples_with_true = [s for s in samples if self.ils_analysis.results[s]['true_concentration']]
        true = [self.ils_analysis.results[s]['true_concentration'] for s in samples_with_true]
        pred_for_true = [self.ils_analysis.results[s]['predicted_concentration'] for s in samples_with_true]
        errors = [self.ils_analysis.results[s]['relative_error'] for s in samples_with_true]
        
        # Plot 1: Predicted vs True concentrations
        if true:
            colors_list = []
            for s in samples_with_true:
                if 'A H' in s:
                    colors_list.append('red')
                elif 'B H' in s:
                    colors_list.append('blue')
                elif 'C H' in s:
                    colors_list.append('green')
                else:
                    colors_list.append('gray')
            
            ax1.scatter(true, pred_for_true, c=colors_list, s=100, alpha=0.7)
            min_val = min(min(true), min(pred_for_true))
            max_val = max(max(true), max(pred_for_true))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')
            ax1.set_xlabel('True Concentration (%)')
            ax1.set_ylabel('Predicted Concentration (%)')
            ax1.set_title('ILS Prediction Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted concentrations bar chart
        bars2 = ax2.bar(range(len(samples)), predicted, color='purple', alpha=0.7)
        ax2.set_title('ILS Predicted Concentrations')
        ax2.set_ylabel('Concentration (%)')
        ax2.set_xticks(range(len(samples)))
        ax2.set_xticklabels(samples, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Relative errors
        if errors:
            bars3 = ax3.bar(range(len(samples_with_true)), errors, color='orange', alpha=0.7)
            ax3.set_title('ILS Prediction Errors')
            ax3.set_ylabel('Relative Error (%)')
            ax3.set_xticks(range(len(samples_with_true)))
            ax3.set_xticklabels(samples_with_true, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars3:
                height = bar.get_height()
                ax3.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Component weights
        A_weights = [self.ils_analysis.results[s]['A_weight'] for s in samples]
        B_weights = [self.ils_analysis.results[s]['B_weight'] for s in samples]
        C_weights = [self.ils_analysis.results[s]['C_weight'] for s in samples]
        
        x = np.arange(len(samples))
        width = 0.25
        
        ax4.bar(x - width, A_weights, width, label='A Weight', color='red', alpha=0.7)
        ax4.bar(x, B_weights, width, label='B Weight', color='blue', alpha=0.7)
        ax4.bar(x + width, C_weights, width, label='C Weight', color='green', alpha=0.7)
        
        ax4.set_title('ILS Component Weights')
        ax4.set_ylabel('Weight')
        ax4.set_xticks(x)
        ax4.set_xticklabels(samples, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.ils_plots_fig.tight_layout()
        self.ils_plots_canvas.draw()
        
    def export_ils_results(self):
        """Export ILS results to CSV file."""
        if not self.ils_analysis.results:
            messagebox.showerror("Error", "No ILS results to export! Run analysis first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save ILS Results As"
        )
        
        if filename:
            try:
                df = self.ils_analysis.get_results_dataframe()
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"ILS results exported to {filename}")
                self.ils_log_message(f"ILS results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export ILS results: {str(e)}")
                
    def clear_ils_results(self):
        """Clear all ILS results and plots."""
        self.ils_analysis.results = {}
        self.ils_analysis.ils_model = None
        
        # Clear displays
        for item in self.ils_results_tree.get_children():
            self.ils_results_tree.delete(item)
            
        self.ils_spectra_fig.clear()
        self.ils_spectra_canvas.draw()
        
        self.ils_plots_fig.clear()
        self.ils_plots_canvas.draw()
        
        self.ils_log_message("ILS results cleared")

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = CLSAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
