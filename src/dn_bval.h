// b-value shells, and the per-shell means removed before PCA.

#ifndef DN_BVAL_H
#define DN_BVAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Read an FSL .bval and cluster it into shells, MRtrix3's way: b <= 10 is b=0,
// and sorted non-zero b-values share a shell while each is LESS than 80 s/mm^2
// from its neighbour (single linkage; a gap of exactly 80 starts a new shell, as
// MRtrix3's `< bvalue_epsilon` does).
//
//   group[j]  shell of volume j, 0..G-1 in order of increasing b, or -1 for a
//             volume left out of demeaning (a shell of one; see dn_bval.c)
//   shell_b   mean b of each shell (length >= nvol)
//   shell_n   volume count of each shell (length >= nvol)
//
// Returns G >= 0, or -1 if the file cannot be read or does not hold exactly
// nvol finite b-values (already reported).
int dn_bval_groups(const char *path, int nvol, int *group, double *shell_b, int *shell_n);

// The input's sidecar name: a trailing .nii.gz, .nii, .hdr or .img replaced by
// .bval, else .bval appended.  Caller frees; NULL on allocation failure.
char *dn_bval_sidecar(const char *input);

// Per-voxel mean of each shell: mean[g*nvox3d + v].  img is volume-major.
// Returns NULL on allocation failure.
float *dn_bval_means(const float *img, size_t nvox3d, int nvol, const int *group, int ngroups);

#ifdef __cplusplus
}
#endif

#endif // DN_BVAL_H
