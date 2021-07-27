import torch
import numpy as np
import util


def measure_mean_potential(pc: torch.Tensor):
    grid = util.gen_grid().to(pc.device)
    return potential(pc, grid).mean()


def potential(sources, means, eps=1e-5, recursive=True, max_pts=15000):
    """
    Calculate dipole potential
    Args:
        sources: position and dipole moments for the fields sources
        means: positions to calculate potential at

    Returns:
        torch.Tensor meansX3 field at measurement positions
    """

    if recursive:
        def break_by_means():
            mid = int(means.shape[0] / 2)
            return torch.cat([potential(sources, means[:mid], eps, recursive, max_pts),
                              potential(sources, means[mid:], eps, recursive, max_pts)], dim=0)

        def break_by_sources():
            mid = int(sources.shape[0] / 2)
            return potential(sources[:mid], means, eps, recursive, max_pts) \
                   + potential(sources[mid:], means, eps, recursive, max_pts)

        if sources.shape[0] > max_pts and means.shape[0] > max_pts:
            if sources.shape[0] > means.shape[0]:
                return break_by_sources()
            else:
                return break_by_means()

        if sources.shape[0] > max_pts:
            return break_by_sources()

        if means.shape[0] > max_pts:
            return break_by_means()

    p = sources[:, 3:]
    R = sources[:, None, :3] - means[None, :, :3]

    phi = (p[:, None, :] * R).sum(dim=-1)
    phi = phi / (R.norm(dim=-1) ** 3)[:, :]
    phi_total = phi.sum(dim=0)

    phi_total[phi_total.isinf()] = 0
    phi_total[phi_total.isnan()] = 0
    return phi_total


def field_grad(sources, means, eps=1e-5, recursive=True, max_pts=15000):
    """
    Calculate dipole field i.e. potential gradient
    Args:
        sources: position and dipole moments for the fields sources
        means: positions to calculate gradient at

    Returns:
        torch.Tensor meansX3 field at measurement positions
    """

    if recursive:
        def break_by_means():
            mid = int(means.shape[0] / 2)
            return torch.cat([field_grad(sources, means[:mid], eps, recursive, max_pts),
                              field_grad(sources, means[mid:], eps, recursive, max_pts)], dim=0)

        def break_by_sources():
            mid = int(sources.shape[0] / 2)
            return field_grad(sources[:mid], means, eps, recursive, max_pts) \
                   + field_grad(sources[mid:], means, eps, recursive, max_pts)

        if sources.shape[0] > max_pts and means.shape[0] > max_pts:
            if sources.shape[0] > means.shape[0]:
                return break_by_sources()
            else:
                return break_by_means()

        if sources.shape[0] > max_pts:
            return break_by_sources()

        if means.shape[0] > max_pts:
            return break_by_means()

    p = sources[:, 3:]
    R = sources[:, None, :3] - means[None, :, :3]
    R_unit = R / R.norm(dim=-1)[:, :, None]
    E = 3 * (p[:, None, :] * R_unit).sum(dim=-1)[:, :, None] * R_unit - p[:, None, :]
    E = E / (R.norm(dim=-1) ** 3 + eps)[:, :, None]
    E_total = E.sum(dim=0) * -1  # field=(-1)*grad -> flip the sign to get the gradient instead of -gradient
    E_total[E_total.isinf()] = 0
    E_total[E_total.isnan()] = 0
    return E_total


def reference_field(pc1, pc2):
    with torch.no_grad():
        E = field_grad(pc1, pc2, recursive=True)
        if pc2.shape[1] == 3:
            length = E.norm(dim=-1)
            E[length != 0, :] = E[length != 0, :] / length[length != 0, None]
            pc2 = torch.cat([pc2, E], dim=1)
        else:
            interactions = E * pc2[:, 3:]
            interactions = interactions.sum(dim=-1)
            sign = (interactions >= 0).float() * 2 - 1
            pc2[:, 3:] = pc2[:, 3:] * sign[:, None]

        return pc2


def strongest_field_propagation_reps(input_pc, reps, diffuse=False, weights=None):
    input_pc = input_pc.detach()
    with torch.no_grad():
        pts = input_pc

        if weights is not None:
            # factor in the weights for each point by scaling the normals
            weights = weights.clamp(0.1, 1)
            pts[:, 3:] = pts[:, 3:] * weights[:, None]
        device = input_pc.device

        remaining = []

        E = torch.zeros_like(pts[:, :3])
        patches = []
        oriented_pts_mask = torch.zeros(len(pts)).bool()
        non_oriented_pts_mask = torch.zeros(len(pts)).bool()
        for rep, rest in reps:
            remaining.append((rep, rest))
            patches.append(rep)
            non_oriented_pts_mask[rep] = True

        # find the flattest patch to start with
        curv = [util.pca_eigen_values(pts[patch]) for patch in patches]
        min_index = np.array([curv[i][0] for i in range(len(patches))])
        min_index = np.abs(min_index)
        min_index = np.argmin(min_index)

        # calculate the field from the initial patch
        start_patch, rest_patch = remaining.pop(min_index)
        oriented_pts_mask[start_patch] = True
        non_oriented_pts_mask[start_patch] = False
        E[non_oriented_pts_mask] = field_grad(pts[oriented_pts_mask], pts[non_oriented_pts_mask])

        # prop orientation as long as there are remaining unoriented patches
        while len(remaining) > 0:
            # calculate the interaction between the field and all remaining patches
            interaction = [(E[patch] * pts[patch, 3:]).sum(dim=-1).sum() for patch, rest in remaining]

            # orient the patch with the strongest interaction
            max_interaction_index = torch.tensor(interaction).abs().argmax().item()
            patch, rest_patch = remaining.pop(max_interaction_index)
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[patch, 3:] *= -1
                pts[rest_patch, 3:] *= -1
            oriented_pts_mask[patch] = True
            non_oriented_pts_mask[patch] = False

            if diffuse:
                # add the effect of the current patch to *all* other patches
                patch_mask = torch.logical_or(oriented_pts_mask, non_oriented_pts_mask)
                patch_mask[patch] = False
                dE = field_grad(pts[patch], pts[patch_mask])
                E[patch_mask] = E[patch_mask] + dE
            else:
                # add the effect of the current patch only to the *remaining* patches
                dE = field_grad(pts[patch], pts[non_oriented_pts_mask])
                E[non_oriented_pts_mask] = E[non_oriented_pts_mask] + dE

        if diffuse:
            for rep, rest in reps:
                interactions = (E[rep] * pts[rep, 3:]).sum(dim=-1)
                sign = (interactions > 0).float() * 2 - 1
                pts[rep, 3:] = pts[rep, 3:] * sign[:, None]

        E = field_grad(pts[oriented_pts_mask], pts[~oriented_pts_mask])
        interactions = (E * pts[~oriented_pts_mask, 3:]).sum(dim=-1)
        sign = (interactions > 0).float() * 2 - 1
        pts[~oriented_pts_mask, 3:] = pts[~oriented_pts_mask, 3:] * sign[:, None]

        pts = pts.to(device)

        if weights is not None:
            # scale the normal back to unit because of previous weighted scaling
            pts[:, 3:] = pts[:, 3:] / weights[:, None]


def strongest_field_propagation(pts, patches, all_patches, diffuse=False, weights=None):
    with torch.no_grad():
        if weights is not None:
            # factor in the weights for each point by scaling the normals
            weights = weights.clamp(0.1, 1)
            pts[:, 3:] = pts[:, 3:] * weights[:, None]
        device = pts.device

        # initialize remaining
        remaining = []
        pts_mask = torch.zeros(len(pts)).bool()
        zeros = torch.zeros(len(pts)).bool()
        E = torch.zeros_like(pts[:, :3])
        for i in range(len(all_patches)):
            remaining.append((i, all_patches[i]))

        # find the flattest patch to start with
        curv = [util.pca_eigen_values(pts[patch]) for patch in all_patches]
        min_index = np.array([curv[i][0] for i in range(len(all_patches))])
        min_index = np.abs(min_index)
        min_index = np.argmin(min_index)

        # calculate the field from the initial patch
        _, start_patch = remaining.pop(min_index)
        pts_mask[start_patch] = True
        E[~pts_mask] = field_grad(pts[pts_mask], pts[~pts_mask])

        # prop orientation as long as there are remaining unoriented patches
        while len(remaining) > 0:
            # calculate the interaction between the field and all remaining patches
            interaction = [(E[patch] * pts[patch, 3:]).sum(dim=-1).sum() for i, patch in remaining]

            # orient the patch with the strongest interaction
            max_interaction_index = torch.tensor(interaction).abs().argmax().item()
            patch_index, patch = remaining.pop(max_interaction_index)
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[patch, 3:] *= -1
            pts_mask[patch] = True

            if diffuse:
                # add the effect of the current patch to *all* other patches
                patch_mask = zeros.clone()
                patch_mask[patch] = True
                dE = field_grad(pts[patch], pts[~patch_mask])
                E[~patch_mask] = E[~patch_mask] + dE
            else:
                # add the effect of the current patch only to the *remaining* patches
                dE = field_grad(pts[patch], pts[~pts_mask])
                E[~pts_mask] = E[~pts_mask] + dE

        if diffuse:
            for patch in patches:
                patch = patch[1]
                interactions = (E[patch] * pts[patch, 3:]).sum(dim=-1)
                sign = (interactions > 0).float() * 2 - 1
                pts[patch, 3:] = pts[patch, 3:] * sign[:, None]

        pts = pts.to(device)

        if weights is not None:
            # scale the normal back to unit because of previous weighted scaling
            pts[:, 3:] = pts[:, 3:] / weights[:, None]


def strongest_field_propagation_points(pts: torch.Tensor, diffuse=False, starting_point=0):
        device = pts.device
        pts = pts.cuda()
        indx = torch.arange(pts.shape[0]).to(pts.device)

        E = torch.zeros_like(pts[:, :3])
        visited = torch.zeros_like(pts[:, 0]).bool()
        visited[starting_point] = True
        E[~(indx == starting_point)] += field_grad(pts[starting_point:(starting_point + 1)],
                                                  pts[~(indx == starting_point), :3], eps=1e-6)

        # prop orientation as long as there are remaining unoriented points
        while not visited.all():
            # calculate the interaction between the field and all remaining patches
            interaction = (E[~visited] * pts[~visited, 3:]).sum(dim=-1)

            # orient the patch with the strongest interaction
            max_interaction_index = interaction.abs().argmax()
            pts_index = indx[~visited][max_interaction_index]
            # print(f'{patch_index}')
            if interaction[max_interaction_index] < 0:
                pts[pts_index, 3:] *= -1
            visited[pts_index] = True

            E[~(indx == pts_index)] += field_grad(pts[pts_index:(pts_index + 1)],
                                                      pts[~(indx == pts_index), :3], eps=1e-6)

        if diffuse:
            interactions = (E * pts[:, 3:]).sum(dim=-1)
            sign = (interactions > 0).float() * 2 - 1
            pts[:, 3:] = pts[:, 3:] * sign[:, None]

        pts = pts.to(device)