"""HOLO -- a zero-dependency augmented reality engine in pure Python.

Pipeline overview::

    scene.render_frame()      synthesise a photo-like frame + ground-truth pose
        -> filters/morph      grayscale, blur, adaptive threshold
        -> contours           connected components + boundary tracing
        -> geom               polygon approximation -> candidate quads
        -> marker.decode      perspective unwarp -> bit grid -> dictionary ID
        -> pose               homography -> 6-DoF pose -> LM refinement
        -> render             draw 3D objects anchored to the marker
        -> png / gif          encode the result

Nothing outside the Python standard library is imported anywhere in this
package (``zlib`` handles DEFLATE for PNG/GIF, ``math`` does the trig).
"""

__version__ = "1.0.0"
