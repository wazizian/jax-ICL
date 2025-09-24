import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    # Define x values from 0 to 4 (avoiding x=0 for log scale)
    x = np.logspace(-3, np.log10(4), 1000)  # From 0.001 to 4 in log scale
    
    # Standard normal distribution N(0,1)
    normal_pdf = stats.norm.pdf(x, loc=0, scale=1)
    
    # Student-t distributions rescaled to have variance 1
    # For Student-t with nu degrees of freedom, variance = nu/(nu-2) for nu > 2
    # To get variance 1, we scale by sqrt((nu-2)/nu)
    
    nus = [3, 5, 10]
    student_pdfs = {}
    
    for nu in nus:
        if nu > 2:
            # Scale factor to make variance = 1
            scale = np.sqrt((nu - 2) / nu)
            student_pdfs[nu] = stats.t.pdf(x, df=nu, scale=scale)
        else:
            # For nu <= 2, variance is undefined, so we just use standard scaling
            student_pdfs[nu] = stats.t.pdf(x, df=nu)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot log(1/f(x)) for each distribution
    plt.loglog(x, 1/normal_pdf, label='N(0,1)', linewidth=2)
    
    colors = ['red', 'green', 'orange']
    for i, nu in enumerate(nus):
        plt.loglog(x, 1/student_pdfs[nu], 
                  label=f'Student-t (ν={nu}, var=1)', 
                  linewidth=2, 
                  color=colors[i])
    
    plt.xlabel('x')
    plt.ylabel('log(1/f(x))')
    plt.title('Log Inverse Probability Density Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.001, 4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
