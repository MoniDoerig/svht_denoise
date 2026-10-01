// patch geometry and the single-patch denoising kernel.
//
// The kernel implements Eq. (4) of Gavish & Donoho 2014 on the Casorati matrix
// of a cubic patch: M patch voxels by N volumes, threshold at
// omega(beta) * median singular value, keep the surviving components.
//
// Two shortcuts make this much cheaper than it looks, and neither changes the
// answer:
//
//  1. Hard thresholding is an orthogonal projection, Xhat = Y V_r V_r', so only
//     the RIGHT singular vectors are needed.  Those are the eigenvectors of the
//     N x N Gram matrix Y'Y, whose eigenvalues are the squared singular values.
//     So a 125 x 102 SVD becomes a 102 x 102 symmetric eigenproblem.
//  2. Only the CENTRE voxel of the patch is written out, so only one row of Xhat
//     is needed.  The projector V_r V_r' is never formed: the centre row is
//     pushed through the retained eigenvectors in two multiplies.
//
// Layout note: the patch is stored TRANSPOSED (volume-major, pt[j*m + i]) rather
// than in the Python's voxel-major order.  That makes each Gram entry a dot
// product of two contiguous length-M rows, and makes the gather read each input
// volume in a compact spatial neighbourhood instead of striding across volumes.

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dn.h"
#include "dn_coef.h"
#include "dn_patch.h"

#ifdef DN_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

// M_PI is XSI, not C99, and this builds under a bare _POSIX_C_SOURCE.
#define DN_PI 3.14159265358979323846

// Origin of the patch centred on `coord` along an axis of length `dim`, clamped
// so the patch stays inside the image: near an edge the patch is SHIFTED INWARDS
// rather than truncated or padded, so every voxel gets a full-size patch.
static int dn_patch_origin(int coord, int dim, int extent) {
	int o = coord - extent / 2;
	if (o < 0) o = 0;
	if (o > dim - extent) o = dim - extent;
	return o;
}

int dn_auto_extent(int nvol) {
	for (int k = 3; k <= DN_MAX_EXTENT; k += 2) {
		double cube = (double)k * k * k;   // double avoids overflow in the test
		if (cube > (double)nvol) return k;
	}
	return 0;
}

// Shared tail of both initialisers: g->m, dims and nvol are set.
static int geom_finish(dn_geom *g, int ngroups) {
	const int nvol = g->nvol, m = g->m;
	// Demeaning G shells leaves G exact-zero singular values in every patch.  They
	// are not noise, so the noise model sees N-G columns: counting them would drag
	// the median down and keep too few components.
	g->ngroups = ngroups;
	g->beta = (double)(nvol - ngroups) / (double)m;
	if (!dn_beta_valid(g->beta)) {
		dn_err("internal: aspect ratio %g is out of range\n", g->beta);
		return 1;
	}
	const double mu = dn_mp_median(g->beta);
	g->omega = dn_lambda_star(g->beta) / sqrt(mu);
	if (!(mu > 0.0) || !isfinite(mu) || !isfinite(g->omega) || g->omega <= 0.0) {
		dn_err("internal: threshold coefficient failed for beta=%g\n", g->beta);
		return 1;
	}
	// sigma_hat = median_s / sqrt(max(M,N) * mu); M > N is guaranteed by the callers.
	g->sigma_scale = 1.0 / sqrt((double)m * mu);
	return 0;
}

// One lattice offset of the sphere, with its squared distance in mm.
typedef struct {
	double d2;
	int x, y, z;
} dn_nb;

// Nearest first; ties in a FIXED order, so the edge selection never depends on
// qsort's.
static int nb_cmp(const void *pa, const void *pb) {
	const dn_nb *a = (const dn_nb *)pa, *b = (const dn_nb *)pb;
	if (a->d2 != b->d2) return a->d2 < b->d2 ? -1 : 1;
	if (a->z != b->z) return a->z - b->z;
	if (a->y != b->y) return a->y - b->y;
	return a->x - b->x;
}

// Every offset with |x| <= hx, |y| <= hy, |z| <= hz, sorted nearest first.
static dn_nb *nb_list(double dx, double dy, double dz, int hx, int hy, int hz, size_t *count) {
	const size_t n = (size_t)(2 * hx + 1) * (2 * hy + 1) * (2 * hz + 1);
	dn_nb *l = (dn_nb *)dn_malloc(n, sizeof(dn_nb));
	if (!l) return NULL;
	size_t i = 0;
	for (int z = -hz; z <= hz; z++)
		for (int y = -hy; y <= hy; y++)
			for (int x = -hx; x <= hx; x++) {
				l[i].d2 = (x * dx) * (x * dx) + (y * dy) * (y * dy) + (z * dz) * (z * dz);
				l[i].x = x; l[i].y = y; l[i].z = z;
				i++;
			}
	qsort(l, n, sizeof(dn_nb), nb_cmp);
	*count = n;
	return l;
}

// The smallest whole-shell ball with at least min_m voxels.
static int geom_init_sphere(dn_geom *g, int nx, int ny, int nz, int nvol,
                            double dx, double dy, double dz, int min_m, int ngroups) {
	g->nx = nx; g->ny = ny; g->nz = nz;
	g->nvol = nvol;
	g->shape = DN_SHAPE_SPHERE;
	g->dx = dx; g->dy = dy; g->dz = dz;
	g->nvox3d = (size_t)nx * (size_t)ny * (size_t)nz;
	if (min_m <= nvol) min_m = nvol + 1;
	if ((size_t)min_m > g->nvox3d) {
		dn_err("a %d-voxel patch does not fit a %dx%dx%d image\n", min_m, nx, ny, nz);
		return 1;
	}
	// Flat offsets are int32_t, and reach at most the whole image.
	if (g->nvox3d > 2147483647u) {
		dn_err("image %dx%dx%d is too large for a spherical patch offset table\n", nx, ny, nz);
		return 1;
	}

	// Box radius from the ball volume, doubled until the min_m-th nearest offset
	// sits strictly inside the box, so no offset of that distance can be missing.
	double rb = 2.0 * cbrt(3.0 * min_m * dx * dy * dz / (4.0 * DN_PI));
	for (;;) {
		const int hx = (int)ceil(rb / dx), hy = (int)ceil(rb / dy), hz = (int)ceil(rb / dz);
		size_t n;
		dn_nb *l = nb_list(dx, dy, dz, hx, hy, hz, &n);
		if (!l) return 1;
		const double r2 = n >= (size_t)min_m ? l[min_m - 1].d2 : INFINITY;
		const double inner = fmin(fmin((hx + 1) * dx, (hy + 1) * dy), (hz + 1) * dz);
		if (r2 < inner * inner) {
			size_t m = (size_t)min_m;
			while (m < n && l[m].d2 <= r2) m++;   // whole shells: no partial ties
			free(l);
			if (m > g->nvox3d) {
				dn_err("a %zu-voxel patch does not fit a %dx%dx%d image\n", m, nx, ny, nz);
				return 1;
			}
			g->m = (int)m;
			g->r2 = r2;
			return geom_finish(g, ngroups);
		}
		free(l);
		rb *= 2.0;
	}
}

int dn_geom_init(dn_geom *g, int shape, int nx, int ny, int nz, int nvol,
                 double dx, double dy, double dz, int extent, int ngroups) {
	if (!g) return 1;
	memset(g, 0, sizeof(*g));

	if (extent < 3 || extent > DN_MAX_EXTENT || (extent % 2) == 0) {
		dn_err("patch extent must be an odd number between 3 and %d (got %d)\n",
		       DN_MAX_EXTENT, extent);
		return 1;
	}
	// As many voxels as the cube it replaces, so the shape changes and the size
	// does not.  Size dominates: measured on benchmark/, the fewest shells with
	// M > N (57 voxels there) was 4-5% worse than the 125-voxel cube, while a
	// sphere of 125 was 0.5% better than it.
	if (shape == DN_SHAPE_SPHERE)
		return geom_init_sphere(g, nx, ny, nz, nvol, dx, dy, dz, extent * extent * extent, ngroups);
	g->dx = dx; g->dy = dy; g->dz = dz;   // the cube's are for the aggregator only
	if (extent > nx || extent > ny || extent > nz) {
		dn_err("patch extent %d exceeds an image dimension (%dx%dx%d)\n",
		       extent, nx, ny, nz);
		return 1;
	}
	const int m = extent * extent * extent;
	if (m <= nvol) {
		dn_err("patch extent %d gives %d voxels for %d volumes; this build requires\n",
		       extent, m, nvol);
		dn_err("  more voxels than volumes (M > N). Use a larger -extent: the default\n");
		dn_err("  rule is the smallest odd k with k^3 > N, which here is %d.\n",
		       dn_auto_extent(nvol));
		return 1;
	}

	g->nx = nx; g->ny = ny; g->nz = nz;
	g->nvol = nvol;
	g->extent = extent;
	g->m = m;
	g->nvox3d = (size_t)nx * (size_t)ny * (size_t)nz;

	// dn_work's offset table is int32_t, and the largest entry it must hold is
	// (extent-1)*(nx*ny + nx + 1).  Nothing else rejects a slice big enough to
	// overflow that: dn_image_read checks each dimension against INT32_MAX and
	// the total against SIZE_MAX, neither of which bounds nx*ny.  A truncated
	// offset would go NEGATIVE and read out of bounds in the gather, so refuse
	// the geometry instead.  Unreachable with any real image -- it needs a single
	// slice of over a billion voxels -- but the check is two lines and the
	// alternative failure is silent.
	const double max_off = (double)(extent - 1) *
	                       ((double)nx * (double)ny + (double)nx + 1.0);
	if (max_off > 2147483647.0) {
		dn_err("image slice %dx%d is too large for a %d-voxel patch offset table\n",
		       nx, ny, extent);
		return 1;
	}
	return geom_finish(g, ngroups);
}

// ---------------------------------------------------------------------------
// worker scratch
// ---------------------------------------------------------------------------

static int cmp_d2(const void *pa, const void *pb) {
	const double a = *(const double *)pa, b = *(const double *)pb;
	return (a > b) - (a < b);
}

void dn_patch_reach(const dn_geom *g, int reach[3]) {
	if (g->shape != DN_SHAPE_SPHERE) {
		// Shifted inwards at an edge, a cube reaches k-1 voxels on one side.
		reach[0] = reach[1] = reach[2] = g->extent - 1;
		return;
	}
	// The sphere's worst case is a corner voxel: only the octant x, y, z >= 0 of
	// the box is inside the image, and its M nearest voxels must all lie strictly
	// inside the box, or a nearer voxel outside it would be missed.  Twice the
	// radius suffices when no axis is clamped to the image; a thin image clamps
	// one, and the others must then reach further -- measured, 40x40x1 at the
	// default size failed outright without this.  Grow until the corner fits.
	const int dims[3] = {g->nx, g->ny, g->nz};
	const double sp[3] = {g->dx, g->dy, g->dz};
	// Seeded at twice the radius alone: adding a voxel of the THICKEST axis, as
	// this once did, inflated every in-plane reach on a thick-slice image.
	double rl = 2.0 * sqrt(g->r2);
	for (;;) {
		double bound = INFINITY;
		for (int a = 0; a < 3; a++) {
			reach[a] = (int)fmin(dims[a] - 1, ceil(rl / sp[a]));
			if (reach[a] < dims[a] - 1) bound = fmin(bound, (reach[a] + 1) * sp[a]);
		}
		const size_t n = (size_t)(reach[0] + 1) * (reach[1] + 1) * (reach[2] + 1);
		if (n >= (size_t)g->m) {
			double *d2 = (double *)dn_malloc(n, sizeof(double));
			if (!d2) {
				// The whole image is always a sufficient reach, and tile_grid sizes
				// its tiles from this, so too small would let same-colour tiles race.
				reach[0] = g->nx - 1; reach[1] = g->ny - 1; reach[2] = g->nz - 1;
				return;
			}
			size_t i = 0;
			for (int z = 0; z <= reach[2]; z++)
				for (int y = 0; y <= reach[1]; y++)
					for (int x = 0; x <= reach[0]; x++)
						d2[i++] = (x * sp[0]) * (x * sp[0]) + (y * sp[1]) * (y * sp[1]) + (z * sp[2]) * (z * sp[2]);
			qsort(d2, n, sizeof(double), cmp_d2);
			const double dm = d2[g->m - 1];
			free(d2);
			if (dm < bound * bound) return;
		}
		rl *= 1.5;
	}
}

dn_work *dn_work_create(const dn_geom *g) {
	if (!g) return NULL;
	dn_work *w = (dn_work *)dn_calloc(1, sizeof(dn_work));
	if (!w) return NULL;
	const int m = g->m, n = g->nvol;
	w->pt = (dn_pt_t *)dn_malloc((size_t)n * m, sizeof(dn_pt_t));
	w->gram = (double *)dn_malloc((size_t)n * n, sizeof(double));
	w->evecs = (double *)dn_malloc((size_t)n * n, sizeof(double));
	w->evals = (double *)dn_malloc((size_t)n, sizeof(double));
	w->s = (double *)dn_malloc((size_t)n, sizeof(double));
	w->yc = (double *)dn_malloc((size_t)n, sizeof(double));
	w->acc = (double *)dn_malloc((size_t)n, sizeof(double));
	w->wt = (double *)dn_malloc((size_t)n, sizeof(double));
	w->offset = (int32_t *)dn_malloc((size_t)m, sizeof(int32_t));
	w->idx = (int32_t *)dn_malloc((size_t)m, sizeof(int32_t));
	w->eig = dn_eig_create(n);
	if (!w->pt || !w->gram || !w->evecs || !w->evals || !w->s || !w->yc || !w->acc || !w->wt ||
	    !w->offset || !w->idx || !w->eig) {
		dn_work_free(w);
		return NULL;
	}

	if (g->shape == DN_SHAPE_SPHERE) {
		int h[3];
		dn_patch_reach(g, h);
		const int hx = h[0], hy = h[1], hz = h[2];
		size_t nl = 0;
		dn_nb *l = nb_list(g->dx, g->dy, g->dz, hx, hy, hz, &nl);
		if (l) {
			w->near = (int32_t *)dn_malloc(nl, sizeof(int32_t));
			w->near_xyz = (int16_t *)dn_malloc(nl * 3, sizeof(int16_t));
		}
		if (!l || !w->near || !w->near_xyz || hx > INT16_MAX || hy > INT16_MAX || hz > INT16_MAX) {
			free(l);
			dn_work_free(w);
			return NULL;
		}
		for (size_t i = 0; i < nl; i++) {
			w->near[i] = (int32_t)(l[i].x + (int64_t)l[i].y * g->nx + (int64_t)l[i].z * g->nx * g->ny);
			w->near_xyz[3 * i] = (int16_t)l[i].x;
			w->near_xyz[3 * i + 1] = (int16_t)l[i].y;
			w->near_xyz[3 * i + 2] = (int16_t)l[i].z;
			if (i < (size_t)m) {
				if (abs(l[i].x) > w->hx) w->hx = abs(l[i].x);
				if (abs(l[i].y) > w->hy) w->hy = abs(l[i].y);
				if (abs(l[i].z) > w->hz) w->hz = abs(l[i].z);
			}
		}
		w->nnear = (int)nl;
		free(l);
		return w;
	}

	// Flat spatial offsets of the patch voxels from the patch origin.  Constant
	// for the whole run because every patch is full size.
	//
	// NIfTI orders a volume with X FASTEST: index = x + y*nx + z*nx*ny.  That is
	// the opposite of a C-order flattening, and getting it
	// backwards produces a plausible-looking image denoised along the wrong axes.
	// dx is innermost here so the gather walks contiguous memory.
	const int k = g->extent;
	int t = 0;
	for (int dz = 0; dz < k; dz++)
		for (int dy = 0; dy < k; dy++)
			for (int dx = 0; dx < k; dx++)
				w->offset[t++] = (int32_t)((size_t)dz * g->nx * g->ny + (size_t)dy * g->nx + dx);
	return w;
}

int dn_work_full(dn_work *w, const dn_geom *g) {
	w->xhat = (double *)dn_malloc((size_t)g->nvol * g->m, sizeof(double));
	w->proj = (double *)dn_malloc((size_t)g->m, sizeof(double));
	return !w->xhat || !w->proj;
}

void dn_work_free(dn_work *w) {
	if (!w) return;
	free(w->pt);
	free(w->gram);
	free(w->evecs);
	free(w->evals);
	free(w->s);
	free(w->yc);
	free(w->acc);
	free(w->wt);
	free(w->offset);
	free(w->idx);
	free(w->xhat);
	free(w->proj);
	free(w->near);
	free(w->near_xyz);
	dn_eig_free(w->eig);
	free(w);
}

// ---------------------------------------------------------------------------
// thresholding
// ---------------------------------------------------------------------------

// Median of an ASCENDING array, using the mean of the middle pair for even n --
// the same convention as MATLAB's median() and NumPy's np.median().
static double dn_median_ascending(const double *s, int n) {
	if (n < 1) return 0.0;
	if (n % 2) return s[n / 2];
	return 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

// Number of retained components for ASCENDING singular values.
//
// The rule is s >= omega * median(s), with one refinement: a component whose
// singular value is exactly zero is never counted.  When the threshold is
// positive the two are identical, because s >= threshold > 0 already implies
// s > 0.  They differ only when the median is zero, i.e. when at least half the
// spectrum has collapsed -- and there the extra components are null directions
// of Y, which contribute EXACTLY nothing to the projection (y_c . v_k is an
// entry of Y v_k = 0).  So this changes no denoised value anywhere; it only
// stops the rank map reading N over empty background, and it makes the all-zero
// all-zero patch fall out on its own: every s is zero, so the rank is 0,
// the projection is empty, and sigma is 0.
//
// The `skip` smallest values are the null directions demeaning created, and are
// excluded from both the median and the count.  Skipped by COUNT, not by value:
// the means are float, so those values are tiny rather than exactly zero.
static int dn_retained_count(const double *s, int n, int skip, double omega, double *median_out) {
	const double med = dn_median_ascending(s + skip, n - skip);
	if (median_out) *median_out = med;
	const double thresh = omega * med;
	int r = 0;
	for (int i = n - 1; i >= skip; i--) {
		if (s[i] > 0.0 && s[i] >= thresh) r++;
		else break;   // ascending, so everything below also fails
	}
	return r;
}

// ---------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------

// Choose the patch rows for the voxel at (ix, iy, iz), and its own row
// (centre_out, may be NULL), into w->base/w->rows; reads no data.
static int patch_select(const dn_geom *g, dn_work *w, int ix, int iy, int iz, int *centre_out) {
	const int m = g->m;

	size_t base;
	const int32_t *offset;
	if (g->shape == DN_SHAPE_SPHERE) {
		// Offsets are relative to the voxel itself, which is the nearest, so row 0.
		base = (size_t)ix + (size_t)iy * g->nx + (size_t)iz * g->nx * g->ny;
		if (centre_out) *centre_out = 0;
		if (ix >= w->hx && ix < g->nx - w->hx && iy >= w->hy && iy < g->ny - w->hy &&
		    iz >= w->hz && iz < g->nz - w->hz) {
			offset = w->near;   // interior: the first M are all inside
		} else {
			int k = 0;
			for (int i = 0; i < w->nnear && k < m; i++) {
				const int x = ix + w->near_xyz[3 * i], y = iy + w->near_xyz[3 * i + 1],
				          z = iz + w->near_xyz[3 * i + 2];
				if (x >= 0 && x < g->nx && y >= 0 && y < g->ny && z >= 0 && z < g->nz)
					w->idx[k++] = w->near[i];
			}
			if (k < m) return -1;   // cannot happen; see dn_work_create
			offset = w->idx;
		}
	} else {
		const int ox = dn_patch_origin(ix, g->nx, g->extent);
		const int oy = dn_patch_origin(iy, g->ny, g->extent);
		const int oz = dn_patch_origin(iz, g->nz, g->extent);
		base = (size_t)ox + (size_t)oy * g->nx + (size_t)oz * g->nx * g->ny;
		offset = w->offset;
		// Row of the patch matrix holding the voxel being denoised, in the same
		// (dz, dy, dx) enumeration order used to build w->offset.
		const int cx = ix - ox, cy = iy - oy, cz = iz - oz;
		if (centre_out) *centre_out = (cz * g->extent + cy) * g->extent + cx;
	}

	w->base = base;
	w->rows = offset;
	return 0;
}

// Gather, and report whether anything in the patch was non-zero.  The flag is
// accumulated here rather than by a second pass over w->pt: the gather already
// touches every element, and at 125x102 a second pass is 102 kB of pointless
// traffic on the majority of patches (these datasets arrive brain-masked, so
// most patches are empty).
static int patch_gather_nonzero(const dn_geom *g, dn_work *w, const float *img,
                                int ix, int iy, int iz, int *centre_out) {
	const int m = g->m, n = g->nvol;
	const size_t nvox3d = g->nvox3d;
	if (patch_select(g, w, ix, iy, iz, centre_out)) return -1;
	const size_t base = w->base;
	const int32_t *offset = w->rows;

	// Gather volume by volume.  Each volume is read over a compact
	// neighbourhood; the outer stride over volumes is paid m times, not m*n.
	int nonzero = 0;
	for (int j = 0; j < n; j++) {
		const float *vol = img + (size_t)j * nvox3d + base;
		dn_pt_t *dst = w->pt + (size_t)j * m;
		const int shell = g->ngroups ? g->group[j] : -1;
		if (shell < 0) {
			for (int i = 0; i < m; i++) {
				const dn_pt_t v = vol[offset[i]];
				dst[i] = v;
				nonzero |= (v != 0.0);
			}
		} else {
			// Tested on the RAW value: an all-zero patch has all-zero means too.
			const float *mu = g->mean + (size_t)shell * nvox3d + base;
			for (int i = 0; i < m; i++) {
				const float v = vol[offset[i]];
				dst[i] = (dn_pt_t)((double)v - (double)mu[offset[i]]);
				nonzero |= (v != 0.0f);
			}
		}
	}
	return nonzero;
}

int dn_patch_rows(const dn_geom *g, dn_work *w, int ix, int iy, int iz) {
	return patch_select(g, w, ix, iy, iz, NULL);
}

// Steps 1-4, shared by both outputs: gather, Gram, eigenvalues, threshold and
// component weights.  Returns 1 with *r_out components chosen, 0 for an
// all-zero patch, negative on failure.
static int patch_decompose(const dn_geom *g, dn_work *w, const float *img,
                           int ix, int iy, int iz, int *centre_out, double *sig_out, int *r_out) {
	const int m = g->m, n = g->nvol;

	// 1. Gather.  An entirely zero patch has an entirely zero spectrum, so the
	//    median, the threshold, the rank and the projection are all zero --
	//    so the shortcut is behaviour-preserving (verified byte-identical) and
	//    worth doing: these datasets arrive brain-masked, so most of the volume
	//    is empty background, and each such patch was otherwise paying a full
	//    Gram and eigensolve to arrive at zero.
	const int gathered = patch_gather_nonzero(g, w, img, ix, iy, iz, centre_out);
	if (gathered <= 0) return gathered;

	// 2. Gram = Y' Y, upper triangle then mirrored.
#ifdef DN_USE_ACCELERATE
	// One dsyrk.  Measured 11x the blocked loop below at 138x343 -- Accelerate
	// puts this on the AMX coprocessor, which no amount of NEON in a hand-written
	// loop reaches.  It is why `pt` is double on this path: there is no
	// mixed-precision syrk to feed float operands into a double accumulator, so
	// the wider gather is the price of admission, and it is worth paying.
	//
	// Row-major with lda = m maps directly onto pt's layout, so no repacking.
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, m, 1.0, w->pt, m,
	            0.0, w->gram, n);
	// dsyrk fills one triangle; the eigensolver's full path reads both.
	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			w->gram[(size_t)j * n + i] = w->gram[(size_t)i * n + j];
#else
	// Gram columns computed per pass over a row.  Swept 2, 4 and 8 in the full
	// workload on 100x100x58x138 -- 1127, 795 and 821 seconds of CPU
	// respectively, against 1221 unblocked.  Do not infer this from clang's
	// vectoriser: at 4 and 8 the cost model declines to vectorise the inner loop
	// at all and only 2 keeps the vectorised form, which would pick the width
	// that recovers 8% over the one that recovers 35%.
	#define DN_GRAM_BLOCK 4
	// Four independent accumulators: a single running sum is a serial dependency
	// chain through the FMA latency, which stops the compiler vectorising and
	// leaves most of the unit idle.  The split is a FIXED partition, so the
	// summation order is still identical on every run and at every thread count.
	//
	// Blocked over j on top of that: DN_GRAM_BLOCK columns share one pass over
	// row i, so row i is read once per block instead of once per j.  On THIS
	// path that is where the time went -- blocking alone is 35% of the whole run
	// at 138x343, and the float rows it feeds on are worth 11% with it and
	// nothing without.  (dsyrk above beats this loop 11x where it is available;
	// this is the only implementation everywhere else.)
	//
	// Neither is a numerics change.  The rows widen back to double here, and
	// since they were copied from a float32 image the widened value IS the value
	// a double array would have held; blocking only interleaves independent dot
	// products, leaving the four accumulators and the fixed partition inside each
	// one untouched.  So the output is bit-identical WITHIN this path -- asserted
	// against a preserved binary on 80 M voxels, not assumed.  It is not
	// bit-identical to the dsyrk path above; see the Makefile.
	for (int i = 0; i < n; i++) {
		const float *a = w->pt + (size_t)i * m;
		int j = i;
		for (; j + DN_GRAM_BLOCK <= n; j += DN_GRAM_BLOCK) {
			const float *b[DN_GRAM_BLOCK];
			double s[DN_GRAM_BLOCK][4] = {{0.0}};
			for (int q = 0; q < DN_GRAM_BLOCK; q++) b[q] = w->pt + (size_t)(j + q) * m;
			int v = 0;
			for (; v + 3 < m; v += 4) {
				const double x0 = a[v], x1 = a[v + 1], x2 = a[v + 2], x3 = a[v + 3];
				for (int q = 0; q < DN_GRAM_BLOCK; q++) {
					s[q][0] += x0 * (double)b[q][v];
					s[q][1] += x1 * (double)b[q][v + 1];
					s[q][2] += x2 * (double)b[q][v + 2];
					s[q][3] += x3 * (double)b[q][v + 3];
				}
			}
			for (; v < m; v++) {
				const double x = a[v];
				for (int q = 0; q < DN_GRAM_BLOCK; q++) s[q][0] += x * (double)b[q][v];
			}
			for (int q = 0; q < DN_GRAM_BLOCK; q++) {
				const double acc = (s[q][0] + s[q][1]) + (s[q][2] + s[q][3]);
				w->gram[(size_t)i * n + j + q] = acc;
				w->gram[(size_t)(j + q) * n + i] = acc;
			}
		}
		for (; j < n; j++) {
			const float *b = w->pt + (size_t)j * m;
			double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
			int v = 0;
			for (; v + 3 < m; v += 4) {
				a0 += (double)a[v] * (double)b[v];
				a1 += (double)a[v + 1] * (double)b[v + 1];
				a2 += (double)a[v + 2] * (double)b[v + 2];
				a3 += (double)a[v + 3] * (double)b[v + 3];
			}
			for (; v < m; v++) a0 += (double)a[v] * (double)b[v];
			const double acc = (a0 + a1) + (a2 + a3);
			w->gram[(size_t)i * n + j] = acc;
			w->gram[(size_t)j * n + i] = acc;
		}
	}
#endif

	// 3. Eigenvalues of the Gram are the squared singular values.
	//
	// Eigenvalues indistinguishable from zero are flushed to zero rather than
	// carried as tiny positives.  Forming Y'Y squares the condition number, so
	// the smallest computed eigenvalues of a rank-deficient patch are pure
	// round-off whose sign and magnitude depend on the solver: keeping them made
	// the RANK MAP differ between two independent eigensolvers at 4% of voxels
	// (all in near-empty background) even though the
	// denoised values agreed to 1e-12, because those directions are null and
	// contribute nothing to the projection.  The cut is the standard numerical
	// rank tolerance, relative to the largest eigenvalue.
	// Only the eigenvalues are needed to reach the threshold decision; the
	// eigenvectors are fetched below, and only the ones that survive it.
	if (dn_eig_values(w->eig, w->gram, w->evals)) return -1;
	const double lam_max = w->evals[n - 1];   // ascending
	const double lam_tol = (lam_max > 0.0) ? lam_max * (double)n * DBL_EPSILON : 0.0;
	for (int i = 0; i < n; i++)
		w->s[i] = (w->evals[i] > lam_tol) ? sqrt(w->evals[i]) : 0.0;

	// 4. Threshold, and a weight for each retained component.
	const size_t vox = (size_t)ix + (size_t)iy * g->nx + (size_t)iz * g->nx * g->ny;
	double med = 0.0;
	int r = dn_retained_count(w->s, n, g->ngroups, g->omega, &med);
	double sig = med * g->sigma_scale;
	if (g->filter != DN_FILTER_OPTTHRESH) {
		if (g->sigma_map && g->sigma_map[vox] > 0.0f) sig = g->sigma_map[vox];
		// Noise singular values of an M x N' matrix of variance sig^2 end at
		// sig*sqrt(M)*(1 + sqrt(beta)); y is s in units of sig*sqrt(M).
		const double unit = sig * sqrt((double)m), edge = 1.0 + sqrt(g->beta);
		r = 0;
		for (int i = n - 1; i >= g->ngroups && w->s[i] > edge * unit; i--) r++;
		for (int k = n - r; k < n; k++) {
			const double y = w->s[k] / unit;
			double wk = 1.0;
			// sigma 0 -- a noiseless patch, whose median singular value is 0 --
			// makes y infinite and eta/y Inf/Inf: keep every positive component whole.
			if (g->filter == DN_FILTER_OPTSHRINK && unit > 0.0) {
				// eta(y) = sqrt((y^2 - beta - 1)^2 - 4 beta) / y, Gavish & Donoho
				// 2017 Eq. 7; the weight is eta/y, clamped against round-off above 1.
				const double t = y * y - g->beta - 1.0;
				wk = sqrt(fmax(t * t - 4.0 * g->beta, 0.0)) / (y * y);
				if (wk > 1.0) wk = 1.0;
			}
			w->wt[k] = wk;
		}
	} else {
		for (int k = n - r; k < n; k++) w->wt[k] = 1.0;
	}

	*sig_out = sig;
	*r_out = r;
	return 1;
}

int dn_denoise_voxel(const dn_geom *g, dn_work *w, const float *img,
                     int ix, int iy, int iz,
                     float *out, float *sigma, uint16_t *rank) {
	const int m = g->m, n = g->nvol;

	int centre = 0, r = 0;
	double sig = 0.0;
	const int rc0 = patch_decompose(g, w, img, ix, iy, iz, &centre, &sig, &r);
	if (rc0 < 0) return 1;
	if (rc0 == 0) {
		if (out)
			for (int j = 0; j < n; j++) out[j] = 0.0f;
		if (sigma) *sigma = 0.0f;
		if (rank) *rank = 0;
		return 0;
	}
	const size_t vox = (size_t)ix + (size_t)iy * g->nx + (size_t)iz * g->nx * g->ny;

	if (sigma) *sigma = (float)sig;
	if (rank) *rank = (uint16_t)r;
	if (!out) return 0;

	// 5. Project the centre row onto the retained subspace, without ever
	//    forming the N x N projector.  The accumulator stays in double and is
	//    rounded to float32 exactly once, at the end.
	if (r <= 0) {
		for (int j = 0; j < n; j++) out[j] = (float)dn_shell_mean(g, vox, j);
		return 0;
	}

	// Now, and only now, do we know which eigenvectors matter.
	if (dn_eig_vectors(w->eig, n - r, r, w->evecs)) return 1;

	double *yc = w->yc, *acc = w->acc;
	for (int j = 0; j < n; j++) {
		yc[j] = (double)w->pt[(size_t)j * m + centre];
		acc[j] = 0.0;
	}
	const double *v = w->evecs;
	for (int k = n - r; k < n; k++) {
		double dot = 0.0;
		for (int j = 0; j < n; j++) dot += yc[j] * v[(size_t)j * n + k];
		dot *= w->wt[k];   // exactly 1.0 under a hard threshold, so bit-neutral there
		for (int j = 0; j < n; j++) acc[j] += dot * v[(size_t)j * n + k];
	}
	for (int j = 0; j < n; j++) out[j] = (float)(acc[j] + dn_shell_mean(g, vox, j));
	return 0;
}

int dn_denoise_patch(const dn_geom *g, dn_work *w, const float *img,
                     int ix, int iy, int iz,
                     float *sigma, uint16_t *rank, int *empty) {
	const int m = g->m, n = g->nvol;
	int r = 0;
	double sig = 0.0;
	const int rc0 = patch_decompose(g, w, img, ix, iy, iz, NULL, &sig, &r);
	if (rc0 < 0) return 1;
	*empty = (rc0 == 0);
	if (sigma) *sigma = (float)sig;
	if (rank) *rank = (uint16_t)r;
	if (*empty) return 0;

	// Xhat = Y V diag(w) V', one retained component at a time: proj = Y v_k is a
	// length-M column, and each volume's row gains w_k v_jk proj.  2*M*N per
	// component, against M*N^2 for the Gram, so it is never the bottleneck.
	double *xhat = w->xhat, *proj = w->proj;
	for (size_t i = 0; i < (size_t)n * m; i++) xhat[i] = 0.0;
	if (r <= 0) return 0;
	if (dn_eig_vectors(w->eig, n - r, r, w->evecs)) return 1;
	const double *v = w->evecs;
	for (int k = n - r; k < n; k++) {
		for (int i = 0; i < m; i++) proj[i] = 0.0;
		for (int j = 0; j < n; j++) {
			const double vjk = v[(size_t)j * n + k];
			const dn_pt_t *row = w->pt + (size_t)j * m;
			for (int i = 0; i < m; i++) proj[i] += (double)row[i] * vjk;
		}
		for (int j = 0; j < n; j++) {
			const double c = w->wt[k] * v[(size_t)j * n + k];
			double *xr = xhat + (size_t)j * m;
			for (int i = 0; i < m; i++) xr[i] += c * proj[i];
		}
	}
	return 0;
}
