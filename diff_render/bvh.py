import torch
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

CUDA_BVH_KERNEL_CODE = r'''
extern "C" __global__
void bvh_intersect_kernel(
    const float3* __restrict__ ray_o,
    const float3* __restrict__ ray_d,
    const float3* __restrict__ box_min,
    const float3* __restrict__ box_max,
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    const int* __restrict__ tri_start,
    const int* __restrict__ tri_count,
    const int* __restrict__ leaf_tris,
    const float3* __restrict__ v0,
    const float3* __restrict__ v1,
    const float3* __restrict__ v2,
    float* __restrict__ out_t,
    float* __restrict__ out_u,
    float* __restrict__ out_v,
    long long* __restrict__ out_idx,
    unsigned char* __restrict__ out_hit,
    int R
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= R) return;

    float3 o = ray_o[idx];
    float3 d = ray_d[idx];
    float3 inv_d = make_float3(
        1.0f / (d.x + (d.x >= 0.0f ? 1e-8f : -1e-8f)),
        1.0f / (d.y + (d.y >= 0.0f ? 1e-8f : -1e-8f)),
        1.0f / (d.z + (d.z >= 0.0f ? 1e-8f : -1e-8f))
    );

    int stack[32];
    int ptr = 0;
    stack[ptr++] = 0;

    float best_t = 1e6f;
    float best_u = 0.0f;
    float best_v = 0.0f;
    long long best_idx = -1;

    while (ptr > 0) {
        int node_id = stack[--ptr];
        float3 bmin = box_min[node_id];
        float3 bmax = box_max[node_id];

        float t1x = (bmin.x - o.x) * inv_d.x;
        float t2x = (bmax.x - o.x) * inv_d.x;
        float t1y = (bmin.y - o.y) * inv_d.y;
        float t2y = (bmax.y - o.y) * inv_d.y;
        float t1z = (bmin.z - o.z) * inv_d.z;
        float t2z = (bmax.z - o.z) * inv_d.z;

        float tnear = fmaxf(fmaxf(fminf(t1x, t2x), fminf(t1y, t2y)), fminf(t1z, t2z));
        float tfar  = fminf(fminf(fmaxf(t1x, t2x), fmaxf(t1y, t2y)), fmaxf(t1z, t2z));

        if (tfar >= tnear && tfar > 0.0f && tnear < best_t) {
            int cnt = tri_count[node_id];
            if (cnt > 0) {
                int st = tri_start[node_id];
                for (int s = 0; s < cnt; ++s) {
                    int tri_id = leaf_tris[st + s];
                    float3 tv0 = v0[tri_id];
                    float3 tv1 = v1[tri_id];
                    float3 tv2 = v2[tri_id];

                    float3 e1 = make_float3(tv1.x - tv0.x, tv1.y - tv0.y, tv1.z - tv0.z);
                    float3 e2 = make_float3(tv2.x - tv0.x, tv2.y - tv0.y, tv2.z - tv0.z);

                    float3 h = make_float3(
                        d.y * e2.z - d.z * e2.y,
                        d.z * e2.x - d.x * e2.z,
                        d.x * e2.y - d.y * e2.x
                    );
                    float a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
                    if (fabsf(a) < 1e-8f) continue;

                    float f = 1.0f / a;
                    float3 sv = make_float3(o.x - tv0.x, o.y - tv0.y, o.z - tv0.z);
                    float u = f * (sv.x * h.x + sv.y * h.y + sv.z * h.z);
                    if (u < 0.0f || u > 1.0f) continue;

                    float3 q = make_float3(
                        sv.y * e1.z - sv.z * e1.y,
                        sv.z * e1.x - sv.x * e1.z,
                        sv.x * e1.y - sv.y * e1.x
                    );
                    float v = f * (d.x * q.x + d.y * q.y + d.z * q.z);
                    if (v < 0.0f || u + v > 1.0f) continue;

                    float thit = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
                    if (thit > 1e-4f && thit < best_t) {
                        best_t = thit;
                        best_u = u;
                        best_v = v;
                        best_idx = tri_id;
                    }
                }
            } else {
                int lc = left_child[node_id];
                int rc = right_child[node_id];
                if (rc >= 0 && ptr < 32) stack[ptr++] = rc;
                if (lc >= 0 && ptr < 32) stack[ptr++] = lc;
            }
        }
    }

    out_t[idx] = best_t;
    out_u[idx] = best_u;
    out_v[idx] = best_v;
    out_idx[idx] = best_idx;
    out_hit[idx] = (best_idx >= 0) ? (unsigned char)1 : (unsigned char)0;
}
'''


class BVHTree:
    """Bounding Volume Hierarchy for ray-triangle acceleration using Surface Area Heuristic."""
    
    _cuda_kernel = None

    def __init__(self, v0, v1, v2, max_leaf_size=4):
        self.max_leaf_size = max_leaf_size
        self.device = v0.device

        v0n = v0.detach().cpu().numpy()
        v1n = v1.detach().cpu().numpy()
        v2n = v2.detach().cpu().numpy()
        self._tri_min = np.minimum(np.minimum(v0n, v1n), v2n)
        self._tri_max = np.maximum(np.maximum(v0n, v1n), v2n)
        self._centroids = (v0n + v1n + v2n) / 3.0

        self._bmin_list, self._bmax_list = [], []
        self._left_list, self._right_list = [], []
        self._tri_start_list, self._tri_count_list, self._leaf_tri_list = [], [], []

        self._build(np.arange(v0n.shape[0]))
        self._finalize()

        if HAS_CUPY and v0.is_cuda and BVHTree._cuda_kernel is None:
            try:
                module = cp.RawModule(code=CUDA_BVH_KERNEL_CODE)
                BVHTree._cuda_kernel = module.get_function("bvh_intersect_kernel")
            except Exception:
                BVHTree._cuda_kernel = False

    def _build(self, indices):
        nid = len(self._bmin_list)
        self._bmin_list.append(self._tri_min[indices].min(axis=0))
        self._bmax_list.append(self._tri_max[indices].max(axis=0))
        self._left_list.append(-1)
        self._right_list.append(-1)
        self._tri_start_list.append(-1)
        self._tri_count_list.append(0)

        n = len(indices)
        if n <= self.max_leaf_size:
            self._tri_start_list[nid] = len(self._leaf_tri_list)
            self._tri_count_list[nid] = n
            self._leaf_tri_list.extend(indices.tolist())
            return nid

        # SAH split evaluation
        bmin, bmax = self._bmin_list[nid], self._bmax_list[nid]
        d = np.maximum(bmax - bmin, 0.0)
        parent_sa = 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
        best_cost, best_axis, best_split = float('inf'), -1, n // 2

        if parent_sa > 1e-12:
            for axis in range(3):
                order = np.argsort(self._centroids[indices, axis])
                s_idx = indices[order]
                s_min, s_max = self._tri_min[s_idx], self._tri_max[s_idx]

                l_min, l_max = np.copy(s_min), np.copy(s_max)
                for i in range(1, n):
                    l_min[i] = np.minimum(l_min[i - 1], s_min[i])
                    l_max[i] = np.maximum(l_max[i - 1], s_max[i])

                r_min, r_max = np.copy(s_min), np.copy(s_max)
                for i in range(n - 2, -1, -1):
                    r_min[i] = np.minimum(r_min[i + 1], s_min[i])
                    r_max[i] = np.maximum(r_max[i + 1], s_max[i])

                ld = np.maximum(l_max[:n - 1] - l_min[:n - 1], 0.0)
                l_sa = 2.0 * (ld[:, 0] * ld[:, 1] + ld[:, 1] * ld[:, 2] + ld[:, 2] * ld[:, 0])
                rd = np.maximum(r_max[1:] - r_min[1:], 0.0)
                r_sa = 2.0 * (rd[:, 0] * rd[:, 1] + rd[:, 1] * rd[:, 2] + rd[:, 2] * rd[:, 0])

                costs = (np.arange(1, n) * l_sa + (n - np.arange(1, n)) * r_sa) / parent_sa
                min_i = np.argmin(costs)
                if costs[min_i] < best_cost:
                    best_cost, best_axis, best_split = costs[min_i], axis, int(min_i) + 1

        if best_axis < 0:
            best_axis = int(np.argmax(bmax - bmin))
            best_split = n // 2

        sorted_idx = indices[np.argsort(self._centroids[indices, best_axis])]
        self._left_list[nid] = self._build(sorted_idx[:best_split])
        self._right_list[nid] = self._build(sorted_idx[best_split:])
        return nid

    def _finalize(self):
        d = self.device
        self.box_min = torch.tensor(np.array(self._bmin_list), dtype=torch.float32, device=d).contiguous()
        self.box_max = torch.tensor(np.array(self._bmax_list), dtype=torch.float32, device=d).contiguous()
        self.left_child = torch.tensor(self._left_list, dtype=torch.int32, device=d).contiguous()
        self.right_child = torch.tensor(self._right_list, dtype=torch.int32, device=d).contiguous()
        self.tri_start = torch.tensor(self._tri_start_list, dtype=torch.int32, device=d).contiguous()
        self.tri_count = torch.tensor(self._tri_count_list, dtype=torch.int32, device=d).contiguous()
        self.is_leaf = (self.tri_count > 0).contiguous()
        self.leaf_tris = torch.tensor(self._leaf_tri_list, dtype=torch.int32, device=d).contiguous()

        del self._bmin_list, self._bmax_list, self._left_list, self._right_list
        del self._tri_start_list, self._tri_count_list, self._leaf_tri_list
        del self._tri_min, self._tri_max, self._centroids

    def intersect(self, ray_o, ray_d, v0, v1, v2):
        R = ray_o.shape[0]
        if R == 0:
            device = ray_o.device
            return (torch.empty(0, device=device), torch.empty(0, device=device),
                    torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.bool, device=device))

        if HAS_CUPY and ray_o.is_cuda and BVHTree._cuda_kernel:
            return self._intersect_cuda(ray_o, ray_d, v0, v1, v2)
        return self._intersect_pytorch(ray_o, ray_d, v0, v1, v2)

    def _intersect_cuda(self, ray_o, ray_d, v0, v1, v2):
        R = ray_o.shape[0]
        device = ray_o.device

        out_t = torch.empty(R, dtype=torch.float32, device=device)
        out_u = torch.empty(R, dtype=torch.float32, device=device)
        out_v = torch.empty(R, dtype=torch.float32, device=device)
        out_idx = torch.empty(R, dtype=torch.long, device=device)
        out_hit = torch.empty(R, dtype=torch.uint8, device=device)

        threads = 256
        blocks = (R + threads - 1) // threads

        BVHTree._cuda_kernel(
            (blocks,), (threads,),
            (
                cp.asarray(ray_o.detach().contiguous()), cp.asarray(ray_d.detach().contiguous()),
                cp.asarray(self.box_min.detach().contiguous()), cp.asarray(self.box_max.detach().contiguous()),
                cp.asarray(self.left_child.detach().contiguous()), cp.asarray(self.right_child.detach().contiguous()),
                cp.asarray(self.tri_start.detach().contiguous()), cp.asarray(self.tri_count.detach().contiguous()),
                cp.asarray(self.leaf_tris.detach().contiguous()),
                cp.asarray(v0.detach().contiguous()), cp.asarray(v1.detach().contiguous()), cp.asarray(v2.detach().contiguous()),
                cp.asarray(out_t.detach().contiguous()), cp.asarray(out_u.detach().contiguous()), cp.asarray(out_v.detach().contiguous()),
                cp.asarray(out_idx.detach().contiguous()), cp.asarray(out_hit.detach().contiguous()),
                R
            )
        )
        return out_t, out_u, out_v, out_idx, out_hit.bool()

    def _intersect_pytorch(self, ray_o, ray_d, v0, v1, v2):
        MAX_STACK = 32
        R = ray_o.shape[0]
        device = ray_o.device

        stack = torch.zeros(R, MAX_STACK, dtype=torch.int32, device=device)
        stack[:, 0] = 0
        ptr = torch.ones(R, dtype=torch.int32, device=device)

        best_t = torch.full((R,), 1e6, dtype=torch.float32, device=device)
        best_u = torch.zeros(R, dtype=torch.float32, device=device)
        best_v = torch.zeros(R, dtype=torch.float32, device=device)
        best_idx = torch.full((R,), -1, dtype=torch.long, device=device)
        hit_any = torch.zeros(R, dtype=torch.bool, device=device)

        inv_d = 1.0 / torch.where(ray_d >= 0, ray_d + 1e-8, ray_d - 1e-8)
        ridx = torch.arange(R, device=device)

        while True:
            active = ptr > 0
            if not active.any():
                break

            pop_pos = (ptr - 1).clamp(min=0).long()
            node_id = stack[ridx, pop_pos].long()
            ptr = torch.where(active, ptr - 1, ptr)

            t1 = (self.box_min[node_id] - ray_o) * inv_d
            t2 = (self.box_max[node_id] - ray_o) * inv_d
            tmin = torch.minimum(t1, t2).amax(dim=-1)
            tmax = torch.maximum(t1, t2).amin(dim=-1)
            box_hit = active & (tmax >= tmin) & (tmax > 0) & (tmin < best_t)

            leaf_mask = box_hit & self.is_leaf[node_id]
            internal_mask = box_hit & ~self.is_leaf[node_id]

            if leaf_mask.any():
                leaf_idx = leaf_mask.nonzero(as_tuple=True)[0]
                l_ray_o, l_ray_d = ray_o[leaf_idx], ray_d[leaf_idx]
                l_node_id, l_best_t = node_id[leaf_idx], best_t[leaf_idx]
                l_start, l_count = self.tri_start[l_node_id], self.tri_count[l_node_id]

                for s in range(self.max_leaf_size):
                    smask = s < l_count
                    tri_id = self.leaf_tris[(l_start + s).clamp(min=0, max=self.leaf_tris.shape[0] - 1).long()].long()

                    tv0, tv1, tv2 = v0[tri_id], v1[tri_id], v2[tri_id]
                    e1, e2 = tv1 - tv0, tv2 - tv0
                    h = torch.linalg.cross(l_ray_d, e2, dim=-1)
                    a = (e1 * h).sum(-1)
                    f = 1.0 / torch.where(a >= 0, a + 1e-8, a - 1e-8)
                    sv = l_ray_o - tv0
                    u = f * (sv * h).sum(-1)
                    q = torch.linalg.cross(sv, e1, dim=-1)
                    vb = f * (l_ray_d * q).sum(-1)
                    t_hit = f * (e2 * q).sum(-1)

                    valid = (smask & (a.abs() > 1e-7) & (u >= 0) & (u <= 1) &
                             (vb >= 0) & (u + vb <= 1) & (t_hit > 1e-4) & (t_hit < l_best_t))

                    if valid.any():
                        update = leaf_idx[valid]
                        best_t[update] = t_hit[valid]
                        best_u[update] = u[valid]
                        best_v[update] = vb[valid]
                        best_idx[update] = tri_id[valid]
                        hit_any[update] = True
                        l_best_t = best_t[leaf_idx]

            if internal_mask.any():
                sel = ridx[internal_mask]
                p = ptr[internal_mask].long()
                stack[sel, p.clamp(max=MAX_STACK - 1)] = self.left_child[node_id[internal_mask]]
                stack[sel, (p + 1).clamp(max=MAX_STACK - 1)] = self.right_child[node_id[internal_mask]]
                ptr[internal_mask] = (p + 2).clamp(max=MAX_STACK).to(torch.int32)

        return best_t, best_u, best_v, best_idx, hit_any