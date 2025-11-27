"""
Simple test script: canvas with 4 random points forming a quad.
Click to add more points and see if they're inside/outside.
Press 'r' to randomize the 4 quad points.
Press 'q' to quit.
"""

import tkinter as tk
import random
import numpy as np
import cv2


class QuadTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quad Test - Click to test points, R to randomize")
        
        self.canvas_size = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='#1a1a2e')
        self.canvas.pack(padx=10, pady=10)
        
        self.label = tk.Label(self.root, text="Click anywhere to test if point is inside quad", font=("Arial", 12))
        self.label.pack(pady=5)
        
        # Generate initial 4 random points
        self.quad_points = []
        self.test_points = []
        self.randomize_quad()
        
        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("r", lambda e: self.randomize_quad())
        self.root.bind("q", lambda e: self.root.quit())
        
    def randomize_quad(self):
        """Generate 4 random points and order them as a quad."""
        margin = 100
        self.quad_points = [
            (random.randint(margin, self.canvas_size - margin),
             random.randint(margin, self.canvas_size - margin))
            for _ in range(4)
        ]
        # Order clockwise from centroid
        self.quad_points = self.order_clockwise(self.quad_points)
        self.test_points = []
        self.redraw()
        
    def order_clockwise(self, pts):
        """Order points clockwise from centroid."""
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        
        def angle_from_center(p):
            return np.arctan2(p[1] - cy, p[0] - cx)
        
        return sorted(pts, key=angle_from_center)
    
    def point_in_or_near_polygon(self, p, polygon, threshold=5):
        """Check if point is inside OR on/near the boundary of polygon."""
        contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        dist = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
        return dist >= -threshold
    
    def is_convex_quad(self, pts):
        """Check if 4 points form a convex quad.
        
        A quad is convex if NO point is inside the triangle formed by the other 3.
        """
        for i in range(4):
            # Triangle from the other 3 points
            triangle = [pts[j] for j in range(4) if j != i]
            # Check if point i is inside that triangle
            if self.point_in_triangle(pts[i], triangle):
                return False
        return True
    
    def point_in_triangle(self, p, triangle):
        """Check if point p is inside triangle using barycentric coords."""
        contour = np.array(triangle, dtype=np.float32).reshape(-1, 1, 2)
        dist = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
        return dist > 0  # Strictly inside (not on boundary)
    
    def on_click(self, event):
        """Test if clicked point is inside the quad."""
        p = (event.x, event.y)
        inside = self.point_in_or_near_polygon(p, self.quad_points)
        self.test_points.append((p, inside))
        self.redraw()
        
        status = "INSIDE" if inside else "OUTSIDE"
        self.label.config(text=f"Point ({p[0]}, {p[1]}): {status}")
    
    def redraw(self):
        self.canvas.delete("all")
        
        # Check if quad is convex
        is_convex = self.is_convex_quad(self.quad_points) if len(self.quad_points) == 4 else False
        
        # Draw quad as filled polygon
        if len(self.quad_points) == 4:
            flat = [coord for pt in self.quad_points for coord in pt]
            fill_color = '#4a4a6a' if is_convex else '#6a3a3a'
            outline_color = '#00ff88' if is_convex else '#ff4444'
            self.canvas.create_polygon(flat, fill=fill_color, outline=outline_color, width=3)
        
        # Draw quad corners
        for i, (x, y) in enumerate(self.quad_points):
            color = '#00ff88' if is_convex else '#ff4444'
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill=color, outline='white', width=2)
            self.canvas.create_text(x, y-20, text=f"P{i+1}", fill='white', font=("Arial", 10, "bold"))
        
        # Draw test points
        for (x, y), inside in self.test_points:
            color = '#00ff00' if inside else '#ff4444'
            self.canvas.create_oval(x-6, y-6, x+6, y+6, fill=color, outline='white', width=1)
        
        # Draw convexity status
        status_text = "CONVEX ✓" if is_convex else "NON-CONVEX ✗"
        status_color = '#00ff88' if is_convex else '#ff4444'
        self.canvas.create_text(self.canvas_size//2, 50, text=status_text, 
                               fill=status_color, font=("Arial", 14, "bold"))
        
        # Draw instructions
        self.canvas.create_text(self.canvas_size//2, 20, 
                               text="R = randomize quad | Click = test point | Q = quit",
                               fill='#888888', font=("Arial", 10))
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = QuadTest()
    app.run()

