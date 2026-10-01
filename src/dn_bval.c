// b-value shells, and the per-shell means removed before PCA.
//
// Removing each voxel's mean over each shell takes the shell-mean signal out of
// the Casorati matrix, so the components that remain are closer to pure noise
// and the signal rank drops.  It costs one exact-zero singular value per shell,
// which the threshold must then skip -- see geom_finish in dn_patch.c.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "dn.h"
#include "dn_bval.h"

#define DN_BZERO 10.0      // MRtrix3 BZeroThreshold default
#define DN_SHELL_EPS 80.0  // MRtrix3 DWI_SHELLS_EPSILON

static int cmp_double(const void *a, const void *b) {
	const double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

char *dn_bval_sidecar(const char *input) {
	static const char *ext[] = {".nii.gz", ".nii", ".hdr", ".img"};
	size_t n = strlen(input);
	for (size_t i = 0; i < sizeof ext / sizeof ext[0]; i++) {
		const size_t e = strlen(ext[i]);
		if (n > e && !strcasecmp(input + n - e, ext[i])) { n -= e; break; }
	}
	char *out = (char *)dn_malloc(n + 6, 1);
	if (out) {
		memcpy(out, input, n);
		memcpy(out + n, ".bval", 6);
	}
	return out;
}

int dn_bval_groups(const char *path, int nvol, int *group, double *shell_b, int *shell_n) {
	FILE *f = fopen(path, "r");
	if (!f) {
		dn_err("cannot read b-values '%s'\n", path);
		return -1;
	}
	double *b = (double *)dn_malloc((size_t)nvol, sizeof(double));
	double *sorted = (double *)dn_malloc((size_t)nvol, sizeof(double));
	int *count = (int *)dn_calloc((size_t)nvol, sizeof(int));
	double *sum = (double *)dn_calloc((size_t)nvol, sizeof(double));
	int n = 0, rc = -1;
	if (!b || !sorted || !count || !sum) goto done;

	double v;
	while (fscanf(f, "%lf", &v) == 1) {
		if (n == nvol || !isfinite(v) || v < 0.0) { n = -1; break; }
		b[n++] = v;
	}
	if (n != nvol || !feof(f)) {
		dn_err("'%s' must hold %d non-negative b-values, one per volume\n", path, nvol);
		goto done;
	}

	// Shell boundaries from the sorted list: b=0 is shell 0 when present, and a
	// gap of DN_SHELL_EPS or more between neighbours starts the next.
	memcpy(sorted, b, (size_t)nvol * sizeof(double));
	qsort(sorted, (size_t)nvol, sizeof(double), cmp_double);
	for (int j = 0; j < nvol; j++) {
		int s = 0;
		for (int i = 1; i < nvol && sorted[i] <= b[j]; i++)
			if (sorted[i] > DN_BZERO && (sorted[i - 1] <= DN_BZERO || sorted[i] - sorted[i - 1] >= DN_SHELL_EPS))
				s++;
		group[j] = s;
		count[s]++;
		sum[s] += b[j];
	}

	// A shell of one volume is NOT demeaned: its mean is the volume itself, so it
	// would be zeroed going in and passed through undenoised coming out.
	int ng = 0;
	int *renum = (int *)sorted;   // reuse: nvol ints fit in nvol doubles
	for (int s = 0; s < nvol; s++) {
		renum[s] = -1;
		if (count[s] < 2) continue;
		shell_b[ng] = sum[s] / count[s];
		shell_n[ng] = count[s];
		renum[s] = ng++;
	}
	for (int j = 0; j < nvol; j++) group[j] = renum[group[j]];
	rc = ng;

done:
	fclose(f);
	free(b);
	free(sorted);
	free(count);
	free(sum);
	return rc;
}

float *dn_bval_means(const float *img, size_t nvox3d, int nvol, const int *group, int ngroups) {
	size_t nmean;
	if (dn_mul_size(nvox3d, (size_t)ngroups, &nmean)) return NULL;
	float *mean = (float *)dn_malloc(nmean, sizeof(float));
	double *acc = (double *)dn_malloc(nvox3d, sizeof(double));
	if (!mean || !acc) {
		free(mean); free(acc);
		return NULL;
	}
	// Volume-outer, so every pass streams one contiguous volume.
	for (int g = 0; g < ngroups; g++) {
		int count = 0;
		for (size_t v = 0; v < nvox3d; v++) acc[v] = 0.0;
		for (int j = 0; j < nvol; j++) {
			if (group[j] != g) continue;
			const float *vol = img + (size_t)j * nvox3d;
			for (size_t v = 0; v < nvox3d; v++) acc[v] += vol[v];
			count++;
		}
		float *dst = mean + (size_t)g * nvox3d;
		for (size_t v = 0; v < nvox3d; v++) dst[v] = (float)(acc[v] / count);
	}
	free(acc);
	return mean;
}
