// Noise map: sigma for the sigma-based filters, and the VST's scale.

#ifndef DN_SIGMA_H
#define DN_SIGMA_H

#include "dn_patch.h"

#ifdef __cplusplus
extern "C" {
#endif

// The estimation grid: one patch every DN_SIGMA_STRIDE voxels per axis, 1/64 of
// the voxels.
#define DN_SIGMA_STRIDE 4

// Patch size k for the estimation pass: the smallest odd k with k^3 >= 2*nvol,
// so the noise bulk is well populated, but never below `extent` nor above
// `cap` (a cube must fit the image; a sphere need not).
int dn_sigma_extent(int nvol, int extent, int cap);

// Estimate sigma (Gavish & Donoho Eq. 26) at patches on a coarse grid, smooth
// across the grid and interpolate to every voxel.  g describes the estimation
// patch.  Returns an nvox3d map (0 where nothing could be estimated), or NULL on
// failure, already reported.
float *dn_sigma_map(const dn_geom *g, const float *img, int nthreads);

// Replace every non-positive entry with the median positive one, for a caller
// that divides by sigma everywhere.  Returns 0 on success, 1 if the map holds no
// estimate at all (left alone: noiseless data), or -1 on allocation failure.
int dn_sigma_fill(float *map, size_t n);

#ifdef __cplusplus
}
#endif

#endif // DN_SIGMA_H
