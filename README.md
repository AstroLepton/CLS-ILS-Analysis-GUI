# CLS & ILS Analysis GUI

A desktop app for analyzing BiVO4 sample concentrations from reflectance spectroscopy data.

## What it does

You point it at a folder of CSV files from your spectrometer, and it figures out the concentration of BiVO4 in each sample using two different methods:

- **CLS (Classical Least Squares)** — builds a calibration model from known standards and fits your unknown samples to it
- **ILS (Inverse Least Squares)** — works in the other direction, using spectra to directly predict concentrations

Both methods work in the 400–700 nm wavelength range and automatically subtract a background reference spectrum.

## What you need

- Python 3
- The following Python packages:

```
pip install pandas numpy matplotlib seaborn
```

tkinter comes with Python on most systems.

## How to run it

```bash
python "cls_and_ils_analysis_gui.py"
```

## How to use it

1. Click **Load Data** and select the folder containing your CSV files
2. Your CSV files should be semicolon-separated and follow this naming convention:
   - `A H...` — high-concentration samples (6.67%)
   - `B H...` — medium-concentration samples (2%)
   - `C H...` — low-concentration samples (1%)
   - `BG www` — background/reference spectrum
3. Click **Run CLS Analysis** or **Run ILS Analysis** (or both)
4. Browse the results in the tabs — spectra plots, concentration plots, and a results table
5. Click **Export CSV** to save the results

## Default concentrations

The calibration standards default to:
- Sample A: 6.67%
- Sample B: 2%
- Sample C: 1%

If your samples use different concentrations, update the `known_concentrations` dictionary near the top of the script.

## Output

- Predicted concentration for each sample (A, B, C components + total)
- Background-corrected reflectance spectra plots
- Absorbance and Kubelka-Munk transform plots
- Exportable CSV table of all results
