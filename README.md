# starDetection
Detection stars with harris corner, implementing harris corner from scratch and detecting stars as discontinuities in images

![starDetection](https://user-images.githubusercontent.com/22850002/157164344-5c7d5824-d1e2-4c9b-950b-ce902c856db7.gif)

```
def getHarrisCorners(x, k, kernel_size):
    """
    s1: implement sliding window
    s2: on every window differentiate image in x-axis and y-axis (dx, dy)
    s3: stack dx and dy and get covariance matirx S # structural tensor
    s4: calcualte harris response which is det(S) - k * sq(trace(S)); where k is precision factor
    s5: take top N% of corrects which will stars
    algorithm can be improved by adding non maxima supression
    """
    # k -> precision
    # kernel_size -> aperture
    res = np.zeros_like(x).astype('f4')
    center = kernel_size // 2
    dy, dx = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3), cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    nRow, nCol = x.shape
    for r in np.arange(nRow - kernel_size + 1):
        for c in np.arange(nCol - kernel_size + 1):
            sdy, sdx = dy[r:r + kernel_size, c:c + kernel_size], dx[r:r + kernel_size, c:c + kernel_size]
            a = np.stack([sdy.ravel(), sdx.ravel()], axis=1)
            (sxx, sxy), (syx, syy) = a.T.dot(a)  # gradient covariance
            det = sxx * syy - sxy * syx
            trace = sxx + syy
            harrisResponse = det - k * (trace ** 2)
            res[r + center, c + center] = harrisResponse
    return res
```
   
