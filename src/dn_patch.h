// patch geometry and the single-patch denoising kernel.

#ifndef DN_PATCH_H
#define DN_PATCH_H

#include <stdint.h>

#include "dn_eig.h"

// Element type of the patch buffer; see the note on `pt` below.
#ifdef DN_USE_ACCELERATE
typedef double dn_pt_t;
#else
typedef float dn_pt_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// How retained components are weighted.
enum {
	DN_FILTER_OPTTHRESH = 0, // Gavish & Donoho 2014 Eq. 4: unknown-sigma hard threshold
	DN_FILTER_OPTSHRINK,     // Gavish & Donoho 2017: Frobenius-optimal shrinkage
	DN_FILTER_TRUNCATE       // hard cut at the noise bulk edge, sigma from the map
};

// Patch shape.  A sphere is measured in mm, so it stays round on anisotropic
// voxels; at the image edge it is the M nearest voxels inside the image.
enum { DN_SHAPE_CUBE = 0, DN_SHAPE_SPHERE };

// Image and patch geometry, plus the per-run settings the kernel reads (filter,
// shell means, sigma map).  Everything here is constant for a whole run: every
// patch holds exactly M voxels -- the cube shifts inwards at an edge, the sphere
// takes the M nearest voxels inside the image -- so beta, and therefore omega,
// never varies from voxel to voxel and is computed exactly once.
typedef struct {
	int nx, ny, nz;      // spatial dimensions
	int nvol;            // volumes (N)
	int shape;           // DN_SHAPE_*
	int extent;          // cube: side length k (odd); sphere: 0
	int m;               // voxels per patch (M), M > N: k^3, or the sphere's count
	double dx, dy, dz;   // sphere: voxel spacing, mm
	double r2;           // sphere: squared radius, mm^2
	size_t nvox3d;       // nx*ny*nz
	double beta;         // (N-G)/M
	double omega;        // Eq. (4) coefficient
	double sigma_scale;  // 1 / sqrt(M * mu_beta); sigma = median_s * sigma_scale
	// Per-shell demeaning, or ngroups = 0 for none.  group[j] is volume j's shell,
	// -1 if not demeaned; mean[s*nvox3d + v] is voxel v's mean over shell s.
	int ngroups;
	const int *group;
	const float *mean;
	// DN_FILTER_*.  The two sigma-based filters read sigma_map[voxel], falling
	// back to the patch's own median estimate where the map is not positive.
	int filter;
	const float *sigma_map;
} dn_geom;

// Fill geometry for an image, a patch shape and a size k (odd): a k^3 cube, or
// the smallest whole-shell ball of at least k^3 voxels, measured in mm with the
// given voxel spacing (positive and finite).  ngroups is the number of demeaned
// shells (0 for none); the caller sets the per-run fields (group, mean, filter,
// sigma_map).  Returns 0 on success; on
// failure the reason has already been reported.
int dn_geom_init(dn_geom *g, int shape, int nx, int ny, int nz, int nvol,
                 double dx, double dy, double dz, int extent, int ngroups);

// The default extent: the smallest odd k with k*k*k > nvol.  Always found for
// nvol <= DN_MAX_VOL; dn.h enforces that at compile time.
int dn_auto_extent(int nvol);

// Per-worker scratch.  One arena per thread; nothing here is shared.
typedef struct {
	// float on the portable path, and that is exact rather than a compromise:
	// every value in here is copied straight out of a float32 image, so widening
	// it at the point of use gives bit-for-bit the same operands a double array
	// would have held, at half the footprint -- 185 kB against 370 kB at 138x343,
	// which is what the blocked Gram loop's inner passes stream.
	//
	// DOUBLE under Accelerate, because BLAS has no mixed-precision syrk: dsyrk
	// wants the operand in the precision it accumulates in.  Paying the wider
	// gather to reach it is measured as worth it; see the Gram kernel.
	dn_pt_t *pt;     // n*m, patch transposed: pt[j*m + i] = voxel i, volume j
	double *gram;    // n*n
	double *evecs;   // n*n
	double *evals;   // n
	double *s;       // n, singular values ascending
	double *yc;      // n, the centre voxel's own vector
	double *acc;     // n, projection accumulator (kept in double, cast once)
	double *wt;      // n, weight of each retained component
	int32_t *offset; // cube: m flat offsets from the patch origin
	// Sphere: every offset out to twice the radius, nearest first, as flat
	// offsets and as (x, y, z) for the bounds test; idx holds the M selected.
	int32_t *near;
	int16_t *near_xyz;
	int nnear;
	int hx, hy, hz;  // largest |x|, |y|, |z| among the first M: the interior test
	int32_t *idx;
	// Set by every gather: row i of the patch is image voxel base + rows[i].
	size_t base;
	const int32_t *rows;
	double *xhat;    // n*m, full-patch estimate, pt's layout; see dn_work_full
	double *proj;    // m
	dn_eig *eig;
} dn_work;

dn_work *dn_work_create(const dn_geom *g);

// Add the scratch only dn_denoise_patch needs (an n*m double block).  Non-zero
// on allocation failure; dn_work_free releases it either way.
int dn_work_full(dn_work *w, const dn_geom *g);

void dn_work_free(dn_work *w);

// Denoise the voxel at spatial index (ix, iy, iz).
//
//   img       float32, VOLUME-MAJOR: img[voxel + volume*nvox3d]
//   out       N denoised values for this voxel (caller-provided, length nvol),
//             or NULL to stop once sigma is known
//   sigma     noise level used for this patch
//   rank      number of retained components
//
// Returns 0 on success, non-zero if the eigensolver failed to converge.
int dn_denoise_voxel(const dn_geom *g, dn_work *w, const float *img,
                     int ix, int iy, int iz,
                     float *out, float *sigma, uint16_t *rank);

// Furthest any patch row can lie from its anchor voxel, per axis, in voxels --
// including the edge-shifted cube and the edge-grown sphere.
void dn_patch_reach(const dn_geom *g, int reach[3]);

// Select the patch anchored at (ix, iy, iz) without reading any data: sets
// w->base and w->rows exactly as the gather would.  Returns 0, or non-zero if a
// sphere could not be filled (which dn_work_create rules out).
int dn_patch_rows(const dn_geom *g, dn_work *w, int ix, int iy, int iz);

// Denoise the WHOLE patch anchored at (ix, iy, iz), for overlapping-patch
// averaging: w->xhat (see dn_work_full) receives the estimate of every row,
// demeaned (shell means are the caller's to restore), and w->base/w->rows locate
// the rows.  *empty is set for an all-zero patch, which leaves xhat unset and
// should contribute nothing.  Returns 0 on success, non-zero if the eigensolver
// failed.
int dn_denoise_patch(const dn_geom *g, dn_work *w, const float *img,
                     int ix, int iy, int iz,
                     float *sigma, uint16_t *rank, int *empty);

// Voxel vox's mean over volume j's shell, which demeaning removed and an output
// must get back; 0 for a volume that was not demeaned.
static inline double dn_shell_mean(const dn_geom *g, size_t vox, int j) {
	const int shell = g->ngroups ? g->group[j] : -1;
	return shell < 0 ? 0.0 : (double)g->mean[(size_t)shell * g->nvox3d + vox];
}

#ifdef __cplusplus
}
#endif

#endif // DN_PATCH_H
