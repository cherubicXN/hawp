import numpy as np
import torch

from shapely.geometry.polygon import LinearRing

try:
    from pycolmap import image_to_world, world_to_image
except:
    pass

### Point-related utils

# Warp a list of points using a homography
def warp_points(points, homography):
    # Convert to homogeneous and in xy format
    new_points = np.concatenate([points[..., [1, 0]],
                                 np.ones_like(points[..., :1])], axis=-1)
    # Warp
    new_points = (homography @ new_points.T).T
    # Convert back to inhomogeneous and hw format
    new_points = new_points[..., [1, 0]] / new_points[..., 2:]
    return new_points


# Mask out the points that are outside of img_size
def mask_points(points, img_size):
    mask = ((points[..., 0] >= 0)
            & (points[..., 0] < img_size[0])
            & (points[..., 1] >= 0)
            & (points[..., 1] < img_size[1]))
    return mask


# Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
# a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate
def keypoints_to_grid(keypoints, img_size):
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points


# Return a 2D matrix indicating the local neighborhood of each point
# for a given threshold and two lists of corresponding keypoints
def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask


### Line-related utils

# Sample n points along lines of shape (num_lines, 2, 2)
def sample_line_points(lines, n):
    line_points_x = np.linspace(lines[:, 0, 0], lines[:, 1, 0], n, axis=-1)
    line_points_y = np.linspace(lines[:, 0, 1], lines[:, 1, 1], n, axis=-1)
    line_points = np.stack([line_points_x, line_points_y], axis=2)
    return line_points


# Return a mask of the valid lines that are within a valid mask of an image
def mask_lines(lines, valid_mask):
    h, w = valid_mask.shape
    int_lines = np.clip(np.round(lines).astype(int), 0, [h - 1, w - 1])
    h_valid = valid_mask[int_lines[:, 0, 0], int_lines[:, 0, 1]]
    w_valid = valid_mask[int_lines[:, 1, 0], int_lines[:, 1, 1]]
    valid = h_valid & w_valid
    return valid


# Return a 2D matrix indicating for each pair of points
# if they are on the same line or not
def get_common_line_mask(line_indices, valid_mask):
    b_size, n_points = line_indices.shape
    common_mask = line_indices[:, :, None] == line_indices[:, None, :]
    common_mask = common_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                           b_size * n_points)
    common_mask = common_mask[valid_mask, :][:, valid_mask]
    return common_mask


# Compute the distances between two sets of lines using the sAP distance
def get_sAP_line_distance(warped_ref_line_seg, target_line_seg):
    dist = (((warped_ref_line_seg[:, None, :, None]
              - target_line_seg[:, None]) ** 2).sum(-1)) ** 0.5
    dist = np.minimum(
        dist[:, :, 0, 0] + dist[:, :, 1, 1],
        dist[:, :, 0, 1] + dist[:, :, 1, 0]
    )
    return dist


# Given a list of line segments and a list of points (2D or 3D coordinates),
# compute the orthogonal projection of all points on all lines.
# This returns the 1D coordinates of the projection on the line,
# as well as the list of orthogonal distances.
def project_point_to_line(line_segs, points):
    # Compute the 1D coordinate of the points projected on the line
    dir_vec = (line_segs[:, 1] - line_segs[:, 0])[:, None]
    coords1d = (((points[None] - line_segs[:, None, 0]) * dir_vec).sum(axis=2)
                / np.linalg.norm(dir_vec, axis=2) ** 2)
    # coords1d is of shape (n_lines, n_points)
    
    # Compute the orthogonal distance of the points to each line
    projection = line_segs[:, None, 0] + coords1d[:, :, None] * dir_vec
    dist_to_line = np.linalg.norm(projection - points[None], axis=2)

    return coords1d, dist_to_line


# Given a list of segments parameterized by the 1D coordinate of the endpoints
# compute the overlap with the segment [0, 1]
def get_segment_overlap(seg_coord1d):
    seg_coord1d = np.sort(seg_coord1d, axis=-1)
    overlap = ((seg_coord1d[..., 1] > 0) * (seg_coord1d[..., 0] < 1)
               * (np.minimum(seg_coord1d[..., 1], 1)
                  - np.maximum(seg_coord1d[..., 0], 0)))
    return overlap


# Compute the symmetrical orthogonal line distance between two sets of lines
# and the average overlapping ratio of both lines.
# Enforce a high line distance for small overlaps.
# This is compatible for nD objects (e.g. both lines in 2D or 3D).
def get_overlap_orth_line_dist(line_seg1, line_seg2, min_overlap=0.5):
    n_lines1, n_lines2 = len(line_seg1), len(line_seg2)

    # Compute the average orthogonal line distance
    coords_2_on_1, line_dists2 = project_point_to_line(
        line_seg1, line_seg2.reshape(n_lines2 * 2, -1))
    line_dists2 = line_dists2.reshape(n_lines1, n_lines2, 2).sum(axis=2)
    coords_1_on_2, line_dists1 = project_point_to_line(
        line_seg2, line_seg1.reshape(n_lines1 * 2, -1))
    line_dists1 = line_dists1.reshape(n_lines2, n_lines1, 2).sum(axis=2)
    line_dists = (line_dists2 + line_dists1.T) / 2

    # Compute the average overlapping ratio
    coords_2_on_1 = coords_2_on_1.reshape(n_lines1, n_lines2, 2)
    overlaps1 = get_segment_overlap(coords_2_on_1)
    coords_1_on_2 = coords_1_on_2.reshape(n_lines2, n_lines1, 2)
    overlaps2 = get_segment_overlap(coords_1_on_2).T
    overlaps = (overlaps1 + overlaps2) / 2

    # Enforce a max line distance for line segments with small overlap
    low_overlaps = overlaps < min_overlap
    line_dists[low_overlaps] = np.amax(line_dists)
    return line_dists


### 3D geometry utils

# Convert from quaternions to rotation matrix
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


# Convert a rotation matrix to quaternions
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# Read the camera intrinsics from a file in COLMAP format
def read_cameras(camera_file, scale_factor=None):
    with open(camera_file, 'r') as f:
        raw_cameras = f.read().rstrip().split('\n')
    raw_cameras = raw_cameras[3:]
    cameras = []
    for c in raw_cameras:
        data = c.split(' ')
        cameras.append({
            "model": data[1],
            "width": int(data[2]),
            "height": int(data[3]),
            "params": np.array(list(map(float, data[4:])))})

    # Optionally scale the intrinsics if the image are resized
    if scale_factor is not None:
        cameras = [scale_intrinsics(c, scale_factor) for c in cameras]
    return cameras


# Adapt the camera intrinsics to an image resize
def scale_intrinsics(intrinsics, scale_factor):
    new_intrinsics = {"model": intrinsics["model"],
                      "width": int(intrinsics["width"] * scale_factor + 0.5),
                      "height": int(intrinsics["height"] * scale_factor + 0.5)
                      }
    params = intrinsics["params"]
    # Adapt the focal length
    params[:2] *= scale_factor
    # Adapt the principal point
    params[2:4] = (params[2:4] * scale_factor + 0.5) - 0.5
    new_intrinsics["params"] = params
    return new_intrinsics


# Project points from 2D to 3D, in (x, y, z) format
def project_2d_to_3d(points, depth, T_local_to_world, intrinsics):
    # Warp to world homogeneous coordinates
    world_points = image_to_world(points[:, [1, 0]],
                                  intrinsics)['world_points']
    world_points *= depth[:, None]
    world_points = np.concatenate([world_points, depth[:, None],
                                   np.ones((len(depth), 1))], axis=1)

    # Warp to the world coordinates
    world_points = (T_local_to_world @ world_points.T).T
    world_points = world_points[:, :3] / world_points[:, 3:]
    return world_points


# Project points from 3D in (x, y, z) format to 2D
def project_3d_to_2d(points, T_world_to_local, intrinsics):
    norm_points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    norm_points = (T_world_to_local @ norm_points.T).T
    norm_points = norm_points[:, :3] / norm_points[:, 3:]
    norm_points = norm_points[:, :2] / norm_points[:, 2:]
    image_points = world_to_image(norm_points, intrinsics)
    image_points = np.stack(image_points['image_points'])[:, [1, 0]]
    return image_points


### Line-ellipse intersection

# Sample n points along ellipses, given as a list of
# tuples (x, c, a, b, theta). Then approximates the 
# ellipse with the output polygon.
def ellipse_polyline(ellipses, n=100):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result


# Compute the intersections between an ellipse a and a line.
def intersect_line_ellipse(a, line):
    ea = LinearRing(a)
    mp = ea.intersection(line)
    if mp.is_empty:
        return np.empty((0, 2))
    elif mp.geom_type == 'Point':
        return np.array([[mp.x, mp.y]])
    elif mp.geom_type == 'MultiPoint':
        return np.stack([[p.x for p in mp], [p.y for p in mp]], axis=-1)
    else:
        raise ValueError('Impossible geometry: ' + mp.geom_type)