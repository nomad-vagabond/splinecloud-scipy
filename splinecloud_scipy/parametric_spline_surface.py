import requests, json
import numpy as np
from scipy.interpolate import bisplev


class ParametricBivariateSpline:
    """
    Bivariate B-spline surface using explicit knot vectors and control points,
    mirroring the tcck convention of the curve loader.

    Internally stores three scipy-compatible tck tuples — one per output
    coordinate (x, y, z) — and evaluates via bisplev.
    """

    def __init__(self, tu, tv, cp, ku, kv, w=None,
                 log_x=False, log_y=False, log_z=False,
                 flip_yz=False):
        """
        Parameters
        ----------
        tu, tv : 1-D array-like
            Knot vectors along the u and v axes.
        cp : array-like, shape (nu, nv, 3)
            Control-point grid. cp[i, j] = [x, y, z].
        ku, kv : int
            Spline degrees along u and v.
        w : array-like, shape (nu, nv), optional
            NURBS weights (stored for future use; all ≈ 1 in current data).
        log_x, log_y, log_z : bool
            Whether each output axis uses a logarithmic scale.
        flip_yz : bool
            If True, swap the Y and Z channels so that Y becomes the
            dependent variable and Z is treated as an independent
            variable (used for lofted surfaces).
        """
        cp = np.asarray(cp, dtype=float)          # (nu, nv, 3)
        nu, nv, _ = cp.shape

        self.tu = np.asarray(tu, dtype=float)
        self.tv = np.asarray(tv, dtype=float)
        self.ku = int(ku)
        self.kv = int(kv)
        self.w  = np.asarray(w, dtype=float) if w is not None else np.ones((nu, nv))
        self.log_x, self.log_y, self.log_z = log_x, log_y, log_z
        self.flip_yz = flip_yz

        # bisplev expects c as a flat array of length len(tu)*len(tv),
        # laid out as the inner (v) index varying fastest.
        #
        # When flip_yz is True the raw control-point columns are stored
        # as [x, z, y] in the API response, so we swap indices 1 and 2
        # to keep _tck_y / _tck_z semantically correct.
        if flip_yz:
            iy, iz = 2, 1
        else:
            iy, iz = 1, 2

        self._tck_x = (self.tu, self.tv, cp[:, :, 0].ravel(), self.ku, self.kv)
        self._tck_y = (self.tu, self.tv, cp[:, :, iy].ravel(), self.ku, self.kv)
        self._tck_z = (self.tu, self.tv, cp[:, :, iz].ravel(), self.ku, self.kv)

    def __call__(self, u, v, grid=False):
        """
        Evaluate the surface at parameter coordinates (u, v).

        Parameters
        ----------
        u, v : float or array-like
            Parameter values within the knot-vector domain.
        grid : bool
            If True, evaluate on the Cartesian product u × v; returned
            arrays have shape (len(u), len(v)).
            If False (default), evaluate point-wise; u and v must match
            in length.

        Returns
        -------
        x, y, z : ndarrays
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))

        x = bisplev(u, v, self._tck_x)
        y = bisplev(u, v, self._tck_y)
        z = bisplev(u, v, self._tck_z)

        if not grid and u.shape == v.shape:
            # bisplev always returns a 2-D grid; collapse the diagonal
            # for point-wise evaluation
            x = x if x.ndim == 1 else np.diag(x)
            y = y if y.ndim == 1 else np.diag(y)
            z = z if z.ndim == 1 else np.diag(z)

        if self.log_x: x = np.exp(x)
        if self.log_y: y = np.exp(y)
        if self.log_z: z = np.exp(z)

        return x, y, z

    def eval(self, x, y, tol=1e-10, max_iter=50, grid_size=20):
        """
        Find z = S_z(u*, v*) where S_x(u*, v*) = x and S_y(u*, v*) = y.

        Uses Newton's method with an analytical Jacobian (exact partial
        derivatives from bisplev) seeded from a coarse grid search.

        Parameters
        ----------
        x, y : float
            Target coordinates in physical space.
        tol : float
            Convergence tolerance on ||F(u,v)||.
        max_iter : int
            Maximum Newton iterations.
        grid_size : int
            Resolution of the coarse grid used for the initial guess.

        Returns
        -------
        z : float
            Interpolated z value at (x, y).

        Raises
        ------
        ValueError
            If Newton's method fails to converge (point likely outside surface).
        """
        # --- 1. Coarse grid search for initial guess -----------------------
        # Sample the surface on a grid to find the (u,v) cell closest to (x,y)
        u_min, u_max = self.tu[self.ku], self.tu[-(self.ku + 1)]
        v_min, v_max = self.tv[self.kv], self.tv[-(self.kv + 1)]

        u_grid = np.linspace(u_min, u_max, grid_size)
        v_grid = np.linspace(v_min, v_max, grid_size)

        # bisplev with grid=True returns (grid_size, grid_size) arrays
        x_grid = bisplev(u_grid, v_grid, self._tck_x)
        y_grid = bisplev(u_grid, v_grid, self._tck_y)

        dist2 = (x_grid - x) ** 2 + (y_grid - y) ** 2
        i0, j0 = np.unravel_index(np.argmin(dist2), dist2.shape)
        u0, v0 = u_grid[i0], v_grid[j0]

        # --- 2. Newton's method with analytical Jacobian ------------------
        u, v = u0, v0
        for _ in range(max_iter):
            # Residual
            fx = bisplev(u, v, self._tck_x) - x
            fy = bisplev(u, v, self._tck_y) - y

            if abs(fx) < tol and abs(fy) < tol:
                break

            # Analytical Jacobian via bisplev derivative flags
            # dx=1,dy=0 → ∂/∂u;  dx=0,dy=1 → ∂/∂v
            dxdu = bisplev(u, v, self._tck_x, dx=1, dy=0)
            dxdv = bisplev(u, v, self._tck_x, dx=0, dy=1)
            dydu = bisplev(u, v, self._tck_y, dx=1, dy=0)
            dydv = bisplev(u, v, self._tck_y, dx=0, dy=1)

            # 2x2 solve:  J * delta = -F
            J = np.array([[dxdu, dxdv], [dydu, dydv]])
            F = np.array([fx, fy])
            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                # Singular Jacobian — surface is locally degenerate at this point
                break

            u += delta[0]
            v += delta[1]

            # Clamp to valid domain after each step
            u = np.clip(u, u_min, u_max)
            v = np.clip(v, v_min, v_max)
        else:
            raise ValueError(
                f"Newton's method did not converge for (x={x}, y={y}). "
                "Point may be outside the surface domain."
            )

        z = float(bisplev(u, v, self._tck_z))
        if self.log_z:
            z = np.exp(z)
        
        return z


    def _build_interval_map(self):
        """
        Build a 2D map of knot-span cells with precomputed (x, y) bounding
        boxes in physical space.

        Each cell corresponds to one bicubic polynomial piece, defined by
        a span [tu[i], tu[i+1]] x [tv[j], tv[j+1]] in parameter space.
        The physical bounding box is estimated by evaluating S_x and S_y
        at the four corners and the centre of each cell.

        Stored as:
            _imap_u_spans  : (M,)   left edges of u spans
            _imap_v_spans  : (N,)   left edges of v spans
            _imap_u_mid    : (M,)   midpoints of u spans (Newton seed, u)
            _imap_v_mid    : (N,)   midpoints of v spans (Newton seed, v)
            _imap_xmin     : (M, N) minimum x over cell sample points
            _imap_xmax     : (M, N) maximum x over cell sample points
            _imap_ymin     : (M, N) minimum y over cell sample points
            _imap_ymax     : (M, N) maximum y over cell sample points
        """
        tu_unique = np.unique(self.tu)
        tv_unique = np.unique(self.tv)

        u_lo = tu_unique[:-1]
        u_hi = tu_unique[1:]
        v_lo = tv_unique[:-1]
        v_hi = tv_unique[1:]

        u_mid = (u_lo + u_hi) / 2
        v_mid = (v_lo + v_hi) / 2

        M, N = len(u_mid), len(v_mid)

        # Sample each cell at 5 points: 4 corners + centre
        # Shape of each bisplev grid call: (M, N)
        xmin = np.full((M, N),  np.inf)
        xmax = np.full((M, N), -np.inf)
        ymin = np.full((M, N),  np.inf)
        ymax = np.full((M, N), -np.inf)

        sample_u = [u_lo, u_hi, u_lo, u_hi, u_mid]
        sample_v = [v_lo, v_lo, v_hi, v_hi, v_mid]

        for su, sv in zip(sample_u, sample_v):
            x_vals = bisplev(su, sv, self._tck_x)   # (M, N)
            y_vals = bisplev(su, sv, self._tck_y)   # (M, N)
            xmin = np.minimum(xmin, x_vals)
            xmax = np.maximum(xmax, x_vals)
            ymin = np.minimum(ymin, y_vals)
            ymax = np.maximum(ymax, y_vals)

        self._imap_u_spans = u_lo
        self._imap_v_spans = v_lo
        self._imap_u_mid   = u_mid
        self._imap_v_mid   = v_mid
        self._imap_xmin    = xmin
        self._imap_xmax    = xmax
        self._imap_ymin    = ymin
        self._imap_ymax    = ymax


    def _find_candidate_cells(self, x, y):
        """
        Return (u0, v0) seed points for all knot-span cells whose physical
        bounding box contains (x, y).

        Returns a list of (u_mid, v_mid) tuples, ordered by distance from
        the cell centre to (x, y) — best candidate first.
        """
        mask = (
            (self._imap_xmin <= x) & (x <= self._imap_xmax) &
            (self._imap_ymin <= y) & (y <= self._imap_ymax)
        )
        ii, jj = np.where(mask)

        if len(ii) == 0:
            # No cell bounding box contains (x,y) — fall back to nearest centre
            cx = bisplev(self._imap_u_mid, self._imap_v_mid, self._tck_x)
            cy = bisplev(self._imap_u_mid, self._imap_v_mid, self._tck_y)
            dist2 = (cx - x) ** 2 + (cy - y) ** 2
            i0, j0 = np.unravel_index(np.argmin(dist2), dist2.shape)
            return [(self._imap_u_mid[i0], self._imap_v_mid[j0])]

        # Sort matching cells by distance from cell centre to target
        u_cands = self._imap_u_mid[ii]
        v_cands = self._imap_v_mid[jj]
        cx = bisplev(self._imap_u_mid[ii], self._imap_v_mid[jj], self._tck_x)
        cy = bisplev(self._imap_u_mid[ii], self._imap_v_mid[jj], self._tck_y)

        # Extract diagonal — bisplev returns a grid, we want point-wise
        dist2 = np.diag(((cx - x) ** 2 + (cy - y) ** 2))
        order  = np.argsort(dist2)

        return [(u_cands[k], v_cands[k]) for k in order]


    def eval_using_2D_map(self, x, y, tol=1e-10, max_iter=50):
        """
        Find z = S_z(u*, v*) where S_x(u*, v*) = x and S_y(u*, v*) = y,
        using the precomputed 2D interval map to seed Newton's method.

        Tries candidate cells in order of proximity, returning the first
        converged solution. Falls back to the nearest-cell centre if all
        candidates fail.

        Parameters
        ----------
        x, y : float
            Target coordinates in physical space.
        tol : float
            Convergence tolerance on max(|F_x|, |F_y|).
        max_iter : int
            Maximum Newton iterations per candidate.

        Returns
        -------
        z : float
        """
        u_min, u_max = self.tu[self.ku],  self.tu[-(self.ku + 1)]
        v_min, v_max = self.tv[self.kv],  self.tv[-(self.kv + 1)]

        candidates = self._find_candidate_cells(x, y)

        for u0, v0 in candidates:
            u, v = u0, v0
            converged = False

            for _ in range(max_iter):
                fx = float(bisplev(u, v, self._tck_x)) - x
                fy = float(bisplev(u, v, self._tck_y)) - y

                if max(abs(fx), abs(fy)) < tol:
                    converged = True
                    break

                dxdu = float(bisplev(u, v, self._tck_x, dx=1, dy=0))
                dxdv = float(bisplev(u, v, self._tck_x, dx=0, dy=1))
                dydu = float(bisplev(u, v, self._tck_y, dx=1, dy=0))
                dydv = float(bisplev(u, v, self._tck_y, dx=0, dy=1))

                J = np.array([[dxdu, dxdv],
                            [dydu, dydv]])
                F = np.array([fx, fy])

                try:
                    delta = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    break  # singular — try next candidate

                u = np.clip(u + delta[0], u_min, u_max)
                v = np.clip(v + delta[1], v_min, v_max)

            if converged:
                z = float(bisplev(u, v, self._tck_z))
                return np.exp(z) if self.log_z else z

        raise ValueError(
            f"eval_using_2D_map did not converge for (x={x}, y={y}). "
            "Point may be outside the surface domain."
        )