// Noise map: sigma for the sigma-based filters, and the VST's scale.
//
// Estimation and denoising want opposite patches: a stable median needs a large,
// well-populated noise bulk, while a spatially specific reconstruction needs a
// small patch.  So sigma is estimated once, from large patches on a coarse grid
// (1/64 of the voxels), and the denoising pass reads it from the map instead of
// trusting the median of its own small patch.

#include <math.h>
#include <stdlib.h>

#include "dn.h"
#include "dn_run.h"
#include "dn_sigma.h"


int dn_sigma_extent(int nvol, int extent, int cap) {
	int k = 3;
	while ((double)k * k * k < 2.0 * nvol) k += 2;
	while (k > cap) k -= 2;
	return k < extent ? extent : k;
}

typedef struct {
	int c0, n;   // first centre, centre count
} dn_axis;

static dn_axis grid_axis(int dim) {
	const int s = DN_SIGMA_STRIDE;
	dn_axis a;
	a.c0 = s / 2 < (dim - 1) / 2 ? s / 2 : (dim - 1) / 2;
	a.n = 1 + (dim - 1 - a.c0) / s;
	return a;
}

// [1 2 1] along one axis of the grid, skipping neighbours past the edge.
static void smooth_axis(double *x, double *tmp, const int n[3], int axis) {
	const size_t stride = axis == 0 ? 1 : axis == 1 ? (size_t)n[0] : (size_t)n[0] * n[1];
	const size_t total = (size_t)n[0] * n[1] * n[2];
	for (size_t i = 0; i < total; i++) {
		const int c = (int)((i / stride) % (size_t)n[axis]);
		double v = 2.0 * x[i];
		if (c > 0) v += x[i - stride];
		if (c < n[axis] - 1) v += x[i + stride];
		tmp[i] = v;
	}
	for (size_t i = 0; i < total; i++) x[i] = tmp[i];
}

float *dn_sigma_map(const dn_geom *g, const float *img, int nthreads) {
	const dn_axis ax[3] = {grid_axis(g->nx), grid_axis(g->ny), grid_axis(g->nz)};
	const int n[3] = {ax[0].n, ax[1].n, ax[2].n};
	const size_t ng = (size_t)n[0] * n[1] * n[2];
	const size_t s = DN_SIGMA_STRIDE;

	float *map = (float *)dn_calloc(g->nvox3d, sizeof(float));
	size_t *list = (size_t *)dn_malloc(ng, sizeof(size_t));
	double *num = (double *)dn_malloc(ng, sizeof(double));
	double *den = (double *)dn_malloc(ng, sizeof(double));
	double *tmp = (double *)dn_malloc(ng, sizeof(double));
	if (!map || !list || !num || !den || !tmp) goto fail;

	size_t i = 0;
	for (int z = 0; z < n[2]; z++)
		for (int y = 0; y < n[1]; y++)
			for (int x = 0; x < n[0]; x++)
				list[i++] = (ax[0].c0 + x * s) + (ax[1].c0 + y * s) * (size_t)g->nx +
				            (ax[2].c0 + z * s) * (size_t)g->nx * g->ny;

	dn_run r = {0};
	r.g = g;
	r.img = img;
	r.noise = map;
	r.list = list;
	r.n_work = ng;
	r.nthreads = nthreads;
	if (dn_run_execute(&r)) goto fail;

	// Normalised smoothing: an all-zero patch has no estimate, and must neither
	// count nor pull its neighbours towards zero.
	for (i = 0; i < ng; i++) {
		const double v = map[list[i]];
		den[i] = v > 0.0 ? 1.0 : 0.0;
		num[i] = v > 0.0 ? v : 0.0;
	}
	for (int a = 0; a < 3; a++) {
		smooth_axis(num, tmp, n, a);
		smooth_axis(den, tmp, n, a);
	}
	for (i = 0; i < ng; i++) num[i] = den[i] > 0.0 ? num[i] / den[i] : 0.0;

	// Trilinear onto every voxel, again weighting only grid points that hold one.
	for (int z = 0; z < g->nz; z++) {
		for (int y = 0; y < g->ny; y++) {
			for (int x = 0; x < g->nx; x++) {
				const int c[3] = {x, y, z};
				int i0[3], i1[3];
				double f[3];
				for (int a = 0; a < 3; a++) {
					double u = (double)(c[a] - ax[a].c0) / (double)s;
					if (u < 0.0) u = 0.0;
					if (u > n[a] - 1) u = n[a] - 1;
					i0[a] = (int)u;
					i1[a] = i0[a] + 1 < n[a] ? i0[a] + 1 : i0[a];
					f[a] = u - i0[a];
				}
				double sw = 0.0, sv = 0.0;
				for (int q = 0; q < 8; q++) {
					const int gx = (q & 1) ? i1[0] : i0[0];
					const int gy = (q & 2) ? i1[1] : i0[1];
					const int gz = (q & 4) ? i1[2] : i0[2];
					const size_t gi = gx + (size_t)gy * n[0] + (size_t)gz * n[0] * n[1];
					if (!(den[gi] > 0.0)) continue;
					const double wq = ((q & 1) ? f[0] : 1.0 - f[0]) *
					                  ((q & 2) ? f[1] : 1.0 - f[1]) *
					                  ((q & 4) ? f[2] : 1.0 - f[2]);
					sw += wq;
					sv += wq * num[gi];
				}
				map[x + (size_t)y * g->nx + (size_t)z * g->nx * g->ny] = sw > 0.0 ? (float)(sv / sw) : 0.0f;
			}
		}
	}
	free(list); free(num); free(den); free(tmp);
	return map;

fail:
	free(map); free(list); free(num); free(den); free(tmp);
	return NULL;
}

static int cmp_float(const void *pa, const void *pb) {
	const float a = *(const float *)pa, b = *(const float *)pb;
	return (a > b) - (a < b);
}

int dn_sigma_fill(float *map, size_t n) {
	size_t k = 0;
	for (size_t i = 0; i < n; i++) k += map[i] > 0.0f;
	if (!k) return 1;   // noiseless: needs no scratch, so cannot fail for lack of it
	float *pos = (float *)dn_malloc(k, sizeof(float));
	if (!pos) return -1;
	k = 0;
	for (size_t i = 0; i < n; i++)
		if (map[i] > 0.0f) pos[k++] = map[i];
	qsort(pos, k, sizeof(float), cmp_float);
	// Mean of the middle pair for an even count, as dn_median_ascending does.
	const float med = (k % 2) ? pos[k / 2] : (float)(0.5 * ((double)pos[k / 2 - 1] + pos[k / 2]));
	free(pos);
	for (size_t i = 0; i < n; i++)
		if (!(map[i] > 0.0f)) map[i] = med;
	return 0;
}
