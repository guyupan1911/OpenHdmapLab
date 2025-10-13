#!/usr/bin/env python3
"""
Test script to verify matplotlib plotting works correctly
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib backend based on environment
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
    print("Using non-interactive backend (Agg)")
else:
    matplotlib.use('TkAgg')  # Use interactive backend if display is available
    print("Using interactive backend (TkAgg)")

# Create a simple test plot
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Test Plot')
ax.legend()
ax.grid(True)

# Save the plot
plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
print("Test plot saved as 'test_plot.png'")

# Try to display the plot if possible
try:
    plt.show()
    print("Plot displayed successfully")
except Exception as e:
    print(f"Display not available: {e}")
    print("Plot saved to file instead")

print("Matplotlib backend:", matplotlib.get_backend())
print("Current working directory:", os.getcwd())

