#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations
void voxelize_fw_cuda(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids);

void voxelize_bw_cuda(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids,
    torch::Tensor grad_voxel_grids,
    torch::Tensor grad_voxel_centers,
    torch::Tensor grad_voxel_radii);


// C++ interface
void voxelize_fw(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids) {

    CHECK_INPUT(points);
    CHECK_INPUT(voxel_centers);
    CHECK_INPUT(voxel_radii);
    CHECK_INPUT(voxel_grids);

    return voxelize_fw_cuda(
        points,
        voxel_centers,
        voxel_radii,
        sigma,
        voxel_grids
    );
}

void voxelize_bw(
    torch::Tensor points,
    torch::Tensor voxel_centers,
    torch::Tensor voxel_radii,
    float         sigma,
    torch::Tensor voxel_grids,
    torch::Tensor grad_voxel_grids,
    torch::Tensor grad_voxel_centers,
    torch::Tensor grad_voxel_radii) {

    CHECK_INPUT(points);
    CHECK_INPUT(voxel_centers);
    CHECK_INPUT(voxel_radii);
    CHECK_INPUT(voxel_grids);
    CHECK_INPUT(grad_voxel_grids);
    CHECK_INPUT(grad_voxel_centers);
    CHECK_INPUT(grad_voxel_radii);

    return voxelize_bw_cuda(
        points,
        voxel_centers,
        voxel_radii,
        sigma,
        voxel_grids,
        grad_voxel_grids,
        grad_voxel_centers,
        grad_voxel_radii
    );
}

// Binding to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_fw", &voxelize_fw, "Voxelization forward (CUDA)");
    m.def("voxelize_bw", &voxelize_bw, "Voxelization backward (CUDA)");
}
