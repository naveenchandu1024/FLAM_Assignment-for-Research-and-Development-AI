## Assessment & Submission Details

Final results:

θ = 0.4881  rad (≈ 27.97°)

M = 0.0205

X = 54.6488

L1 = 12730.2516


## The final Latex Equations:
\left(t*\cos(0.4881)-e^{0.0205\left(t\right)}*\sin(0.3t)*\sin(0.4881)+54.6488  ,  \ 42+t*\sin(0.4881)+e^{0.0205\left(t\right)}*\sin(0.3t)*\cos(0.4881)\right)


![image alt](https://github.com/naveenchandu1024/FLAM_Assignment-for-Research-and-Development-AI/blob/main/demos_parametric.png?raw=true)



   


### Assessment Criteria



1. **L1 Distance  ,**  
   The accuracy of the predicted curve is measured using the L1 distance between uniformly sampled points of the actual and predicted curves.  
   A lower L1 value indicates a better fit.  
   In my case, the computed **L1 distance = 12730.2516**, showing a close match for all 1500 samples.


2. **Explanation of Process ,**  
   
   Problem statement: Find the values of unknown variables in the given **parametric equation of a curve**

   ### **Step 0 — Setup**
- Created `curve_fit_assignment/` project folder.
- Added `src/` and `data/` subfolders.
- Installed dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`).

### **Step 1 — Understanding the Model**
The parametric model defines \(x(t)\) and \(y(t)\) as functions of unknowns \( \theta, M, X \).
Goal: Estimate these parameters such that the predicted curve matches the given data.

### **Step 2 — Data Exploration**
Loaded `xy_data.csv` (1500 points). Verified that only (x, y) values were provided.

### **Step 3 — Estimating t (Parameterization)**
Used **non-linear parameterization** based on cumulative arc-length and adaptive scaling.
This ensures smooth and non-uniform spacing along the curve for realistic fitting.

### **Step 4 — Building the Model**
\[
x(t) = t*\cos(\theta) - e^{M|t|}*\sin(0.3t)*\sin(\theta) + X
\]
\[
y(t) = 42 + t*\sin(\theta) + e^{M|t|}*\sin(0.3t)*\cos(\theta)
\]

### **Step 5 — Parameter Estimation**
Optimization done in **two phases**:
1. **Phase A:** Least Squares (L2) — provides initial guess.
2. **Phase B:** L1 minimization — refines for robustness to outliers.

### **Step 6 — L1 Distance Evaluation**
Computed L1 distance between predicted curve and interpolated observed curve using 500 uniform t samples.

### **Step 7 — Visualization**
Plotted the fitted curve vs data points using `visualize.py`.
Result saved as **fit.png** showing close overlap between curves.

### **Step 8 — Final Output**
Optimized parameters stored in `fit_params.npy`.
Final dataset with t-values saved in `xy_data_with_t.csv`.

---

## 🧾 Final LaTeX Equation (for Submission)

\left(t*\cos(0.4881)-e^{0.0205\left(t\right)}*\sin(0.3t)*\sin(0.4881)+54.6488  ,  \ 42+t*\sin(0.4881)+e^{0.0205\left(t\right)}*\sin(0.3t)*\cos(0.4881)\right)
