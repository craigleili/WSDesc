#include "utils.cuh"

namespace { // kernel space

/* Helper functions */
template <typename scalar_t=float>
__device__ __forceinline__ scalar_t sigmoid(const scalar_t x) {
    return 1. / (1. + exp(-x));
}

template <typename scalar_t=float>
__device__ __forceinline__ scalar_t distance3d(const scalar_t* p0, const scalar_t* p1) {
    const scalar_t d0 = p0[0] - p1[0];
    const scalar_t d1 = p0[1] - p1[1];
    const scalar_t d2 = p0[2] - p1[2];
    return sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

/* Kernel functions */
template <typename scalar_t=float, typename index_t=int>
__global__ void voxelize_fw_cuda_kernel(
    // Inputs
    const scalar_t* __restrict__ points,
    const scalar_t* __restrict__ voxel_centers,
    const scalar_t* __restrict__ voxel_radii,
    const scalar_t               sigma,
    const index_t                num_points,
    const index_t                num_keypoints,
    const index_t                num_voxels,
    const index_t                loops,
    // Outputs
    scalar_t*       __restrict__ voxel_grids
    ) {

    // points:              (N, 3)
    // voxel_centers:       (K, V, 3)
    // voxel_radii:         (K, V)

    // voxel_grids:         (K, V)

    // Global id = num_points (N) * num_voxels (V)
    const index_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= loops) return;

    const index_t pid = gid / num_voxels; // Point id
    const index_t vid = gid % num_voxels; // Voxel id

    const index_t pid3 = pid * 3;
    scalar_t point[3] = {points[pid3], points[pid3 + 1], points[pid3 + 2]};

    for (int64_t kid = 0; kid < num_keypoints; ++kid) { // For each keypoint / voxel grid
        const index_t gvid = kid * num_voxels + vid; // Global voxel id

        const index_t gvid3 = gvid * 3;
        scalar_t center[3] = {voxel_centers[gvid3], voxel_centers[gvid3 + 1], voxel_centers[gvid3 + 2]};

        const scalar_t voxel_radius = voxel_radii[gvid];

        const scalar_t dist_pc = distance3d(point, center);
        const scalar_t dist = dist_pc - voxel_radius;
        const scalar_t sign = dist > 0 ? -1 : 1;
        const scalar_t prob = sigmoid(sign * dist * dist / sigma);
        if (prob < 1e-3) continue;

        atomicMul(&voxel_grids[gvid], 1 - prob);
    }
}

template <typename scalar_t=float, typename index_t=int>
__global__ void voxelize_bw_cuda_kernel(
    // Inputs
    const scalar_t* __restrict__ points,
    const scalar_t* __restrict__ voxel_centers,
    const scalar_t* __restrict__ voxel_radii,
    const scalar_t               sigma,
    const scalar_t* __restrict__ voxel_grids,
    const scalar_t* __restrict__ grad_voxel_grids,
    const index_t                num_points,
    const index_t                num_keypoints,
    const index_t                num_voxels,
    const index_t                loops,
    // Outputs
    scalar_t*       __restrict__ grad_voxel_centers,
    scalar_t*       __restrict__ grad_voxel_radii
    ) {

    // points:              (N, 3)
    // voxel_centers:       (K, V, 3)
    // voxel_radii:         (K, V)
    // voxel_grids:         (K, V), 1 - \Pi_j (1 - P_ij)
    // grad_voxel_grids:    (K, V)

    // grad_voxel_centers:  (K, V, 3)
    // grad_voxel_radii:    (K, V)

    // Global id = num_points (N) * num_voxels (V)
    const index_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= loops) return;

    const index_t pid = gid / num_voxels; // Point id
    const index_t vid = gid % num_voxels; // Voxel id

    const index_t pid3 = pid * 3;
    scalar_t point[3] = {points[pid3], points[pid3 + 1], points[pid3 + 2]};

    for (int64_t kid = 0; kid < num_keypoints; ++kid) { // For each keypoint / voxel grid
        const index_t gvid = kid * num_voxels + vid; // Global voxel id

        const index_t gvid3 = gvid * 3;
        scalar_t center[3] = {voxel_centers[gvid3], voxel_centers[gvid3 + 1], voxel_centers[gvid3 + 2]};

        const scalar_t voxel_radius = voxel_radii[gvid];

        const scalar_t dist_pc = distance3d(point, center);
        const scalar_t dist = dist_pc - voxel_radius;
        const scalar_t sign = dist > 0 ? -1 : 1;
        const scalar_t prob = sigmoid(sign * dist * dist / sigma);
        if (prob < 1e-3) continue;

        // Intermediate vars
        const scalar_t voxel_val = voxel_grids[gvid];
        const scalar_t dvdp = (1 - voxel_val) / max(1 - prob, 1e-6);
        const scalar_t dpdd = prob * (1 - prob) * sign * 2 * dist / sigma;
        const scalar_t dddx = (center[0] - point[0]) / dist_pc;
        const scalar_t dddy = (center[1] - point[1]) / dist_pc;
        const scalar_t dddz = (center[2] - point[2]) / dist_pc;
        const scalar_t dddr = -1;
        const scalar_t dpdx = dpdd * dddx;
        const scalar_t dpdy = dpdd * dddy;
        const scalar_t dpdz = dpdd * dddz;
        const scalar_t dpdr = dpdd * dddr;

        const scalar_t dldv = grad_voxel_grids[gvid];

        atomicAdd(&grad_voxel_centers[gvid3], dldv * dvdp * dpdx);
        atomicAdd(&grad_voxel_centers[gvid3 + 1], dldv * dvdp * dpdy);
        atomicAdd(&grad_voxel_centers[gvid3 + 2], dldv * dvdp * dpdz);
        atomicAdd(&grad_voxel_radii[gvid], dldv * dvdp * dpdr);
    }

}

} // kernel space

void voxelize_fw_cuda(
    // Inputs
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    // Outputs
    torch::Tensor voxel_grids
    ) {

    if (points.dim() != 2) {
        fprintf(stderr,"\nSize of points is incorrect.\n");
        exit(-1);
    }
    if (voxel_centers.dim() != 3) {
        fprintf(stderr,"\nSize of voxel_centers is incorrect.\n");
        exit(-1);
    }
    if (voxel_radii.dim() != 2) {
        fprintf(stderr,"\nSize of voxel_radii is incorrect.\n");
        exit(-1);
    }

    const auto num_points = points.size(0);
    const auto num_keypoints = voxel_centers.size(0);
    const auto num_voxels = voxel_centers.size(1);

    const int64_t loops = num_points * num_voxels;
    const int64_t threads = MAX_THREADS;
    const int64_t blocks  = gpu_blocks(loops, threads);

    voxelize_fw_cuda_kernel<float, int64_t><<<blocks, threads>>>(
        points.data_ptr<float>(),
        voxel_centers.data_ptr<float>(),
        voxel_radii.data_ptr<float>(),
        sigma,
        num_points,
        num_keypoints,
        num_voxels,
        loops,
        voxel_grids.data_ptr<float>()
    );
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}

void voxelize_bw_cuda(
    // Inputs
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids,
    torch::Tensor grad_voxel_grids,
    // Outputs
    torch::Tensor grad_voxel_centers,
    torch::Tensor grad_voxel_radii
    ) {

    if (points.dim() != 2) {
        fprintf(stderr,"\nSize of points is incorrect.\n");
        exit(-1);
    }
    if (voxel_centers.dim() != 3) {
        fprintf(stderr,"\nSize of voxel_centers is incorrect.\n");
        exit(-1);
    }
    if (voxel_radii.dim() != 2) {
        fprintf(stderr,"\nSize of voxel_radii is incorrect.\n");
        exit(-1);
    }
    if (voxel_grids.dim() != 2) {
        fprintf(stderr,"\nSize of voxel_grids is incorrect.\n");
        exit(-1);
    }
    if (grad_voxel_grids.dim() != 2) {
        fprintf(stderr,"\nSize of grad_voxel_grids is incorrect.\n");
        exit(-1);
    }

    const auto num_points = points.size(0);
    const auto num_keypoints = voxel_centers.size(0);
    const auto num_voxels = voxel_centers.size(1);

    const int64_t loops = num_points * num_voxels;
    const int64_t threads = MAX_THREADS;
    const int64_t blocks  = gpu_blocks(loops, threads);

    voxelize_bw_cuda_kernel<float, int64_t><<<blocks, threads>>>(
        points.data_ptr<float>(),
        voxel_centers.data_ptr<float>(),
        voxel_radii.data_ptr<float>(),
        sigma,
        voxel_grids.data_ptr<float>(),
        grad_voxel_grids.data_ptr<float>(),
        num_points,
        num_keypoints,
        num_voxels,
        loops,
        grad_voxel_centers.data_ptr<float>(),
        grad_voxel_radii.data_ptr<float>()
    );
    GPU_ERROR_CHECK(cudaPeekAtLastError());
    GPU_ERROR_CHECK(cudaDeviceSynchronize());
}
