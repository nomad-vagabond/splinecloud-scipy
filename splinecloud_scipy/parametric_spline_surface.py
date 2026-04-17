import requests, json
import numpy as np
from scipy.interpolate import bisplev


class ParametricBivariateSpline:
    """
    Bivariate B-spline surface using explicit knot vectors and control points.

    Internally stores three scipy-compatible tck tuples — one per output
    coordinate (x, y, z) — and evaluates via bisplev.
    """

    def __init__(self, tu, tv, cp, ku, kv, w=None,
                 log_x=False, log_y=False, log_z=False, flip_yz=False):
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
        cp = np.asarray(cp, dtype=float)
        nu, nv, _ = cp.shape

        self.tu = np.asarray(tu, dtype=float)
        self.tv = np.asarray(tv, dtype=float)
        self.ku = int(ku)
        self.kv = int(kv)
        self.w = np.asarray(w, dtype=float) if w is not None else np.ones((nu, nv))
        self.log_x, self.log_y, self.log_z = log_x, log_y, log_z
        self.flip_yz = flip_yz

        # When flip_yz is True swap indices 1 and 2 to keep _tck_y / _tck_z semantically correct.
        if flip_yz:
            iy, iz = 2, 1
        else:
            iy, iz = 1, 2

        self._tck_x = (self.tu, self.tv, cp[:, :, 0].ravel(), self.ku, self.kv)
        self._tck_y = (self.tu, self.tv, cp[:, :, iy].ravel(), self.ku, self.kv)
        self._tck_z = (self.tu, self.tv, cp[:, :, iz].ravel(), self.ku, self.kv)

        self._build_search_grid()

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

    def _build_search_grid(self):
        """
        Pre-evaluate the surface at the centres of all knot spans.
        Called once at construction; result cached for all eval() calls.

        Stores
        ------
        _search_u : (Mu,)   u midpoints of each unique u span
        _search_v : (Mv,)   v midpoints of each unique v span
        _search_x : (Mu, Mv) physical x at each (u_mid, v_mid)
        _search_y : (Mu, Mv) physical y at each (u_mid, v_mid)
        """
        tu_unique = np.unique(self.tu)
        tv_unique = np.unique(self.tv)

        self._search_u = (tu_unique[:-1] + tu_unique[1:]) / 2   # (Mu,)
        self._search_v = (tv_unique[:-1] + tv_unique[1:]) / 2   # (Mv,)

        # bisplev returns (Mu, Mv) grid by default
        self._search_x = bisplev(self._search_u, self._search_v, self._tck_x)
        self._search_y = bisplev(self._search_u, self._search_v, self._tck_y)

    def eval(self, x, y, tol=1e-10, max_iter=50, grid_size=20, threshold=100):
        """
        Evaluate z = S_z(u*, v*) where S_x(u*, v*) = x and S_y(u*, v*) = y.
        Handles both scalar and vector inputs for x and y.

        Parameters
        ----------
        x, y : float or array-like
            Target coordinates in physical space.
        tol : float
            Convergence tolerance on ||F(u,v)||.
        max_iter : int
            Maximum Newton iterations.
        grid_size : int
            Resolution of the coarse grid used for the initial guess (for scalars).
        threshold : int
            The total number of points above which vectorized eval_grid is used.

        Returns
        -------
        result : float or tuple (X, Y, Z)
            If input is scalar, returns z (float).
            If input is vector, returns (X, Y, Z) meshgrids.
        """
        x_is_iter = hasattr(x, '__iter__')
        y_is_iter = hasattr(y, '__iter__')

        if not x_is_iter and not y_is_iter:
            return self.eval_point(x, y, tol=tol, max_iter=max_iter, grid_size=grid_size)

        x_vals = np.atleast_1d(x)
        y_vals = np.atleast_1d(y)

        if len(x_vals) * len(y_vals) >= threshold:
            return self.eval_grid(x_vals, y_vals, tol=tol, max_iter=max_iter)
        else:
            X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
            Z = np.array([[self.eval_point(X[i, j], Y[i, j], tol=tol, max_iter=max_iter, grid_size=grid_size)
                           for j in range(Y.shape[1])]
                          for i in range(X.shape[0])])

            return X, Y, Z

    def eval_point(self, x, y, tol=1e-10, max_iter=50, grid_size=20):
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
        z : float or None (if Newton's method fails to converge)
            Interpolated z value at (x, y).

        """
        # --- 1. Coarse grid search for initial guess -----------------------
        # Sample the surface on a grid to find the (u,v) cell closest to (x,y)

        u_grid = np.linspace(0, 1, grid_size)
        v_grid = np.linspace(0, 1, grid_size)

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
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
        else:
            return None

        z = float(bisplev(u, v, self._tck_z))
        if self.log_z:
            z = np.exp(z)
        
        return z

    def eval_grid(self, x_vals, y_vals, tol=1e-10, max_iter=50):
        """
        Evaluate z = S_z(u*, v*) for a regular grid of (x, y) values,
        using vectorized Newton's method across all points simultaneously.

        Parameters
        ----------
        x_vals : 1-D array of shape (Nx,)
        y_vals : 1-D array of shape (Ny,)

        Returns
        -------
        X, Y, Z : 2-D arrays of shape (Nx, Ny)
        """
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        shape = X.shape
        x_flat = X.ravel()
        y_flat = Y.ravel()
        n = len(x_flat)

        # --- Vectorized initial guess from precomputed search grid ---
        # _search_x/_search_y are (Mu, Mv) grids; flatten for broadcasting
        sx = self._search_x.ravel()          # (Mu*Mv,)
        sy = self._search_y.ravel()
        su = np.repeat(self._search_u, len(self._search_v))
        sv = np.tile(self._search_v, len(self._search_u))

        # For each target point find nearest search-grid cell — (n, Mu*Mv)
        dist2 = (sx[None, :] - x_flat[:, None])**2 + \
                (sy[None, :] - y_flat[:, None])**2   # (n, Mu*Mv)
        best = np.argmin(dist2, axis=1)              # (n,)
        u = su[best].copy()
        v = sv[best].copy()

        # --- Vectorized Newton ---
        # Track which points have not yet converged
        active = np.ones(n, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break

            ua, va = u[active], v[active]
            xa, ya = x_flat[active], y_flat[active]

            # Residuals — one bisplev call per coordinate over active points
            fx = np.array([bisplev(ui, vi, self._tck_x)
                          for ui, vi in zip(ua, va)]) - xa
            fy = np.array([bisplev(ui, vi, self._tck_y)
                          for ui, vi in zip(ua, va)]) - ya

            # Mark converged
            converged = (np.abs(fx) < tol) & (np.abs(fy) < tol)
            active[np.where(active)[0][converged]] = False

            still = ~converged
            if not still.any():
                break

            ua, va = ua[still], va[still]
            fx, fy = fx[still], fy[still]
            xa, ya = xa[still], ya[still]

            # Jacobian entries
            dxdu = np.array([bisplev(ui, vi, self._tck_x, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dxdv = np.array([bisplev(ui, vi, self._tck_x, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])
            dydu = np.array([bisplev(ui, vi, self._tck_y, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dydv = np.array([bisplev(ui, vi, self._tck_y, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])

            # Batch 2x2 solve: det(J) and Cramer's rule — avoids per-point linalg.solve
            det = dxdu * dydv - dxdv * dydu
            safe = np.abs(det) > 1e-14
            du = np.where(safe, ( dydv * (-fx) - dxdv * (-fy)) / det, 0.0)
            dv = np.where(safe, (-dydu * (-fx) + dxdu * (-fy)) / det, 0.0)

            idx = np.where(active)[0]
            u[idx] = np.clip(u[idx] + du, 0, 1)
            v[idx] = np.clip(v[idx] + dv, 0, 1)

        # --- Final evaluation ---
        Z_flat = np.array([bisplev(ui, vi, self._tck_z)
                          for ui, vi in zip(u, v)], dtype=float)
        if self.log_z:
            Z_flat = np.exp(Z_flat)

        Z = Z_flat.reshape(shape)
        
        return X, Y, Z
