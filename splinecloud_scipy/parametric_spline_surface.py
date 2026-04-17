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

    def eval(self, x, y, tol=1e-10, max_iter=50, threshold=100, compute_gradients=False):
        x_is_iter = hasattr(x, '__iter__')
        y_is_iter = hasattr(y, '__iter__')

        if not x_is_iter and not y_is_iter:
            return self.eval_point(x, y, tol=tol, max_iter=max_iter,
                                   compute_gradients=compute_gradients)

        x_vals = np.atleast_1d(x)
        y_vals = np.atleast_1d(y)

        if len(x_vals) * len(y_vals) >= threshold:
            return self.eval_grid(x_vals, y_vals, tol=tol, max_iter=max_iter,
                                  compute_gradients=compute_gradients)
        else:
            X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
            Z = np.zeros_like(X)
            if compute_gradients:
                dZdX = np.zeros_like(X)
                dZdY = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        z, dzdx, dzdy = self.eval_point(
                            X[i, j], Y[i, j], tol=tol, max_iter=max_iter, compute_gradients=True)
                        Z[i, j] = z
                        dZdX[i, j] = dzdx
                        dZdY[i, j] = dzdy
                
                return X, Y, Z, dZdX, dZdY
            else:
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = self.eval_point(X[i, j], Y[i, j], tol=tol, max_iter=max_iter)
                return X, Y, Z

    def eval_point(self, x, y, tol=1e-10, max_iter=50, compute_gradients=False):
        """
        Find z = S_z(u*, v*) where S_x(u*, v*) = x and S_y(u*, v*) = y.

        Parameters
        ----------
        x, y : float
        tol : float
        max_iter : int
        compute_gradients : bool
            If True, also return (dz/dx, dz/dy) via the implicit function theorem.

        Returns
        -------
        z : float, or None if Newton did not converge.
        (dzdx, dzdy) : tuple of float, only if compute_gradients=True.
        """
        # --- Initial guess from search grid ---
        sx = self._search_x.ravel()
        sy = self._search_y.ravel()
        su = np.repeat(self._search_u, len(self._search_v))
        sv = np.tile(self._search_v, len(self._search_u))

        dist2 = (sx - x)**2 + (sy - y)**2
        best  = np.argmin(dist2)
        u, v  = su[best], sv[best]

        # --- Newton's method — keep final Jacobian if needed ---
        J = None
        for _ in range(max_iter):
            fx = float(bisplev(u, v, self._tck_x)) - x
            fy = float(bisplev(u, v, self._tck_y)) - y

            if abs(fx) < tol and abs(fy) < tol:
                break

            dxdu = float(bisplev(u, v, self._tck_x, dx=1, dy=0))
            dxdv = float(bisplev(u, v, self._tck_x, dx=0, dy=1))
            dydu = float(bisplev(u, v, self._tck_y, dx=1, dy=0))
            dydv = float(bisplev(u, v, self._tck_y, dx=0, dy=1))

            J = np.array([[dxdu, dxdv], [dydu, dydv]])
            F = np.array([fx, fy])

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break

            u = np.clip(u + delta[0], 0, 1)
            v = np.clip(v + delta[1], 0, 1)
        else:
            if compute_gradients:
                return None, None, None
            
            return None

        z = float(bisplev(u, v, self._tck_z))

        if not compute_gradients:
            if self.log_z:
                z = np.exp(z)
            return z

        # --- Gradients via implicit function theorem ---
        dzdu = float(bisplev(u, v, self._tck_z, dx=1, dy=0))
        dzdv = float(bisplev(u, v, self._tck_z, dx=0, dy=1))
        grad_uv = np.array([dzdu, dzdv])

        try:
            grad_xy = np.linalg.solve(J, grad_uv)
        except np.linalg.LinAlgError:
            grad_xy = np.array([np.nan, np.nan])

        if self.log_z:
            z = np.exp(z)
            grad_xy *= z
        if self.log_x:
            grad_xy[0] /= x
        if self.log_y:
            grad_xy[1] /= y

        return z, float(grad_xy[0]), float(grad_xy[1])

    def eval_grid(self, x_vals, y_vals, tol=1e-10, max_iter=50, compute_gradients=False):
        """
        Evaluate z over a regular (x, y) grid using vectorized Newton.

        Parameters
        ----------
        x_vals : 1-D array of shape (Nx,)
        y_vals : 1-D array of shape (Ny,)
        tol : float
        max_iter : int
        compute_gradients : bool
            If True, also return dz/dx and dz/dy grids.

        Returns
        -------
        X, Y, Z : 2-D arrays of shape (Nx, Ny)
        dZdX, dZdY : 2-D arrays of shape (Nx, Ny), only if compute_gradients=True.
        """
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        shape = X.shape
        x_flat = X.ravel()
        y_flat = Y.ravel()
        n = len(x_flat)

        # --- Vectorized initial guess ------------------------------------
        sx = self._search_x.ravel()
        sy = self._search_y.ravel()
        su = np.repeat(self._search_u, len(self._search_v))
        sv = np.tile(self._search_v, len(self._search_u))

        dist2 = (sx[None, :] - x_flat[:, None])**2 + \
                (sy[None, :] - y_flat[:, None])**2
        best = np.argmin(dist2, axis=1)
        u = su[best].copy()
        v = sv[best].copy()

        # Store final Jacobian only when gradients are needed
        J_final = np.zeros((n, 2, 2)) if compute_gradients else None

        # --- Vectorized Newton -------------------------------------------
        active = np.ones(n, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break

            idx = np.where(active)[0]
            ua, va = u[idx], v[idx]
            xa, ya = x_flat[idx], y_flat[idx]

            fx = np.array([bisplev(ui, vi, self._tck_x)
                        for ui, vi in zip(ua, va)]) - xa
            fy = np.array([bisplev(ui, vi, self._tck_y)
                        for ui, vi in zip(ua, va)]) - ya

            converged = (np.abs(fx) < tol) & (np.abs(fy) < tol)
            active[idx[converged]] = False

            still = ~converged
            if not still.any():
                break

            ua, va = ua[still], va[still]
            fx, fy = fx[still], fy[still]

            dxdu = np.array([bisplev(ui, vi, self._tck_x, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dxdv = np.array([bisplev(ui, vi, self._tck_x, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])
            dydu = np.array([bisplev(ui, vi, self._tck_y, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dydv = np.array([bisplev(ui, vi, self._tck_y, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])

            if compute_gradients:
                still_idx = idx[still]
                J_final[still_idx, 0, 0] = dxdu
                J_final[still_idx, 0, 1] = dxdv
                J_final[still_idx, 1, 0] = dydu
                J_final[still_idx, 1, 1] = dydv

            det = dxdu * dydv - dxdv * dydu
            safe = np.abs(det) > 1e-14
            du = np.where(safe, ( dydv * (-fx) - dxdv * (-fy)) / det, 0.0)
            dv = np.where(safe, (-dydu * (-fx) + dxdu * (-fy)) / det, 0.0)

            u[idx[still]] = np.clip(u[idx[still]] + du, 0, 1)
            v[idx[still]] = np.clip(v[idx[still]] + dv, 0, 1)

        # --- Final z evaluation ------------------------------------------
        Z_flat = np.array([bisplev(ui, vi, self._tck_z)
                          for ui, vi in zip(u, v)], dtype=float)

        if not compute_gradients:
            if self.log_z:
                Z_flat = np.exp(Z_flat)
            
            return X, Y, Z_flat.reshape(shape)

        # --- Gradients via implicit function theorem ---------------------
        dzdu = np.array([bisplev(ui, vi, self._tck_z, dx=1, dy=0)
                        for ui, vi in zip(u, v)])
        dzdv = np.array([bisplev(ui, vi, self._tck_z, dx=0, dy=1)
                        for ui, vi in zip(u, v)])
        grad_uv = np.stack([dzdu, dzdv], axis=1)        # (n, 2)

        det = J_final[:, 0, 0] * J_final[:, 1, 1] - \
              J_final[:, 0, 1] * J_final[:, 1, 0]
        safe = np.abs(det) > 1e-14

        dzdx = np.where(safe,
                        ( J_final[:, 1, 1] * grad_uv[:, 0] -
                        J_final[:, 0, 1] * grad_uv[:, 1]) / det,
                        np.nan)
        dzdy = np.where(safe,
                        (-J_final[:, 1, 0] * grad_uv[:, 0] +
                        J_final[:, 0, 0] * grad_uv[:, 1]) / det,
                        np.nan)

        if self.log_z:
            Z_flat = np.exp(Z_flat)
            dzdx *= Z_flat
            dzdy *= Z_flat
        if self.log_x:
            dzdx /= x_flat
        if self.log_y:
            dzdy /= y_flat

        return (X, Y, Z_flat.reshape(shape),
                dzdx.reshape(shape), dzdy.reshape(shape))
