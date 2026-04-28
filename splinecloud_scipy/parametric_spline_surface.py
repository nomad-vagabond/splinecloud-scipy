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
            self.log_y = log_z
            self.log_z = log_y
        else:
            iy, iz = 1, 2
            
        self._tck_x = (self.tu, self.tv, cp[:, :, 0].ravel(), self.ku, self.kv)
        self._tck_y = (self.tu, self.tv, cp[:, :, iy].ravel(), self.ku, self.kv)
        self._tck_z = (self.tu, self.tv, cp[:, :, iz].ravel(), self.ku, self.kv)

        self._build_search_grid()

    def __call__(self, u, v):
        """
        Evaluate the surface at parameter coordinates (u, v).

        Parameters
        ----------
        u, v : float or array-like
               Parameter values within the knot-vector domain.

        Returns
        -------
        x, y, z : ndarrays of computed coordinates with shape (len(u), len(v))
                  or scalars if u and v are scalars
        """

        u_in = np.asarray(u, dtype=float)
        v_in = np.asarray(v, dtype=float)
        scalar_input = u_in.ndim == 0 and v_in.ndim == 0

        u_arr = np.atleast_1d(u_in)
        v_arr = np.atleast_1d(v_in)

        x = bisplev(u_arr, v_arr, self._tck_x)
        y = bisplev(u_arr, v_arr, self._tck_y)
        z = bisplev(u_arr, v_arr, self._tck_z)

        if self.log_x: x = np.pow(10, x)
        if self.log_y: y = np.pow(10, y)
        if self.log_z: z = np.pow(10, z)

        if scalar_input:
            return float(x), float(y), float(z)
        
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

        # bisplev returns (Mu, Mv) grid if Mu, Mv > 1, but may return scalar/1D if they are 1.
        # Ensure result is always (Mu, Mv)
        self._search_x = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_x)).reshape(len(self._search_u), len(self._search_v))
        self._search_y = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_y)).reshape(len(self._search_u), len(self._search_v))
        self._search_z = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_z)).reshape(len(self._search_u), len(self._search_v))

    def _bisplev_d2(self, u, v, tck, du=0, dv=0, eps=1e-7):
        """
        Evaluate a second-order (or mixed) derivative of a bivariate spline,
        working around the bisplev constraint that derivative order must be
        strictly less than the spline degree (dx < kx).

        For degree >= 3: uses bisplev directly.
        For degree == 2: the second derivative is piecewise constant;
                         computed via finite differences of the first derivative.
        For degree <= 1: returns 0 (linear or constant spline).

        Parameters
        ----------
        du, dv : int
            Derivative orders along u and v (each 0, 1, or 2, with du+dv <= 2).
        """
        # Pure second derivative w.r.t. u
        if du == 2 and dv == 0:
            if self.ku >= 3:
                return float(bisplev(u, v, tck, dx=2, dy=0))
            elif self.ku >= 2:
                fp = float(bisplev(u + eps, v, tck, dx=1, dy=0))
                fm = float(bisplev(u - eps, v, tck, dx=1, dy=0))
                return (fp - fm) / (2 * eps)
            else:
                return 0.0

        # Pure second derivative w.r.t. v
        if du == 0 and dv == 2:
            if self.kv >= 3:
                return float(bisplev(u, v, tck, dx=0, dy=2))
            elif self.kv >= 2:
                fp = float(bisplev(u, v + eps, tck, dx=0, dy=1))
                fm = float(bisplev(u, v - eps, tck, dx=0, dy=1))
                return (fp - fm) / (2 * eps)
            else:
                return 0.0

        # Mixed derivative d²/dudv
        if du == 1 and dv == 1:
            if self.ku >= 2 and self.kv >= 2:
                return float(bisplev(u, v, tck, dx=1, dy=1))
            else:
                return 0.0

        raise ValueError(f"Unsupported derivative order du={du}, dv={dv}")

    def _compute_boundary_point(self, x, y):
        """
        Find the nearest point on the surface boundary in physical (x, y)
        space, refined with 1D Newton, and return all derivatives needed
        for second-order Taylor extrapolation.

        Returns
        -------
        S_b    : ndarray (3,)  — [x_b, y_b, z_b] at boundary
        dSdu   : ndarray (3,)  — first derivatives w.r.t. u
        dSdv   : ndarray (3,)  — first derivatives w.r.t. v
        d2Sdu2 : ndarray (3,)  — second derivatives w.r.t. u
        d2Sdv2 : ndarray (3,)  — second derivatives w.r.t. v
        d2Sdudv: ndarray (3,)  — mixed derivatives
        du_ext : float         — parameter displacement from boundary (u)
        dv_ext : float         — parameter displacement from boundary (v)
        """
        u_min, u_max = self.tu[self.ku],  self.tu[-(self.ku + 1)]
        v_min, v_max = self.tv[self.kv],  self.tv[-(self.kv + 1)]


        # --- Coarse boundary search --------------------------------------
        n_edge = 100
        u_vals = np.linspace(u_min, u_max, n_edge)
        v_vals = np.linspace(v_min, v_max, n_edge)

        best_dist2 = np.inf
        best_u, best_v = u_min, v_min
        best_edge = None

        edges = [
            ('u_min', np.full(n_edge, u_min), v_vals),
            ('u_max', np.full(n_edge, u_max), v_vals),
            ('v_min', u_vals, np.full(n_edge, v_min)),
            ('v_max', u_vals, np.full(n_edge, v_max)),
        ]

        for edge_name, u_edge, v_edge in edges:
            x_edge = np.array([float(bisplev(ui, vi, self._tck_x))
                            for ui, vi in zip(u_edge, v_edge)])
            y_edge = np.array([float(bisplev(ui, vi, self._tck_y))
                            for ui, vi in zip(u_edge, v_edge)])
            dist2 = (x_edge - x)**2 + (y_edge - y)**2
            idx = np.argmin(dist2)
            if dist2[idx] < best_dist2:
                best_dist2 = dist2[idx]
                best_u, best_v = u_edge[idx], v_edge[idx]
                best_edge = edge_name

        # --- Refine with 1D Newton along the boundary edge ---------------
        u_b, v_b = self._refine_boundary_point(
            x, y, best_u, best_v, best_edge,
            u_min, u_max, v_min, v_max
        )

        # --- Evaluate all derivatives at boundary point ------------------
        def _eval_derivs(tck, u, v):
            val   = float(bisplev(u, v, tck))
            du    = float(bisplev(u, v, tck, dx=1, dy=0))
            dv    = float(bisplev(u, v, tck, dx=0, dy=1))
            du2   = self._bisplev_d2(u, v, tck, du=2, dv=0)
            dv2   = self._bisplev_d2(u, v, tck, du=0, dv=2)
            dudv  = self._bisplev_d2(u, v, tck, du=1, dv=1)
            return val, du, dv, du2, dv2, dudv

        xb, dxdu, dxdv, d2xdu2, d2xdv2, d2xdudv = _eval_derivs(self._tck_x, u_b, v_b)
        yb, dydu, dydv, d2ydu2, d2ydv2, d2ydudv = _eval_derivs(self._tck_y, u_b, v_b)
        zb, dzdu, dzdv, d2zdu2, d2zdv2, d2zdudv = _eval_derivs(self._tck_z, u_b, v_b)

        # --- Solve for (du_ext, dv_ext) to match target (x, y) ----------
        # First-order solve: J * [du, dv]^T = [x - xb, y - yb]^T
        # then refine with second-order correction
        J = np.array([[dxdu, dxdv], [dydu, dydv]])
        rhs = np.array([x - xb, y - yb])

        try:
            d_params = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            d_params = np.zeros(2)

        du_ext, dv_ext = d_params

        S_b     = np.array([xb,      yb,      zb    ])
        dSdu    = np.array([dxdu,    dydu,    dzdu  ])
        dSdv    = np.array([dxdv,    dydv,    dzdv  ])
        d2Sdu2  = np.array([d2xdu2,  d2ydu2,  d2zdu2])
        d2Sdv2  = np.array([d2xdv2,  d2ydv2,  d2zdv2])
        d2Sdudv = np.array([d2xdudv, d2ydudv, d2zdudv])

        return S_b, dSdu, dSdv, d2Sdu2, d2Sdv2, d2Sdudv, du_ext, dv_ext

    def _refine_boundary_point(self, x, y, u0, v0, edge,
            u_min, u_max, v_min, v_max, tol=1e-10, max_iter=50):
        """
        Refine boundary point with 1D Newton along the free parameter
        of the detected edge.
        """
        u, v = u0, v0
        free_u = edge in ('v_min', 'v_max')   # u is free parameter

        for _ in range(max_iter):
            xb = float(bisplev(u, v, self._tck_x))
            yb = float(bisplev(u, v, self._tck_y))

            if free_u:
                dxdt = float(bisplev(u, v, self._tck_x, dx=1, dy=0))
                dydt = float(bisplev(u, v, self._tck_y, dx=1, dy=0))
                d2xdt2 = self._bisplev_d2(u, v, self._tck_x, du=2, dv=0)
                d2ydt2 = self._bisplev_d2(u, v, self._tck_y, du=2, dv=0)
            else:
                dxdt = float(bisplev(u, v, self._tck_x, dx=0, dy=1))
                dydt = float(bisplev(u, v, self._tck_y, dx=0, dy=1))
                d2xdt2 = self._bisplev_d2(u, v, self._tck_x, du=0, dv=2)
                d2ydt2 = self._bisplev_d2(u, v, self._tck_y, du=0, dv=2)

            grad = (xb - x) * dxdt + (yb - y) * dydt
            hess = (dxdt**2 + dydt**2
                    + (xb - x) * d2xdt2
                    + (yb - y) * d2ydt2)

            if abs(hess) < 1e-14 or abs(grad) < tol:
                break

            step = grad / hess
            if free_u:
                u = np.clip(u - step, u_min, u_max)
            else:
                v = np.clip(v - step, v_min, v_max)

        return u, v

    def _extrapolate_point(self, x, y, compute_gradients=False,
            limit_distance=False, limit_consistency=False, limit_steepness=False,
            consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Extrapolate z at (x, y) outside the surface domain using a
        second-order Taylor expansion from the nearest boundary point,
        with reliability checks.

        Parameters
        ----------
        x, y : float
            Target coordinates in internal (log-) space — callers must
            convert physical values via log10() before passing them in.
        compute_gradients : bool
            If True, also return (dz/dx, dz/dy).
        consistency_threshold : float
            Maximum allowed ratio |z2 - z1| / |z2 - z_b| before the
            extrapolation is considered unreliable. Default 0.2 means
            the quadratic correction must be less than 20% of the total
            extrapolated change. Dimensionless and self-calibrating.
        distance_threshold : float
            Maximum allowed extrapolation distance as a fraction of the
            surface's physical diagonal extent. Default 0.5 means the
            query point must be within half the surface's own size.

        Returns
        -------
        z : float or None
            None if any reliability check fails.
        (dzdx, dzdy) : tuple of float or (None, None),
            only when compute_gradients=True.
        """
        _failed = (None, (None, None)) if compute_gradients else None

        S_b, dSdu, dSdv, d2Sdu2, d2Sdv2, d2Sdudv, du, dv = self._compute_boundary_point(x, y)

        J = np.array([[dSdu[0], dSdv[0]], [dSdu[1], dSdv[1]]])
        first_order  = dSdu[2] * du + dSdv[2] * dv
        second_order = 0.5 * d2Sdu2[2] * du**2 + d2Sdudv[2] * du * dv + 0.5 * d2Sdv2[2] * dv**2

        x_b, y_b, z_b = S_b
        z1 = z_b + first_order
        z2 = z_b + first_order + second_order

        x_extent = self._search_x.max() - self._search_x.min()
        y_extent = self._search_y.max() - self._search_y.min()

        if limit_distance:
            # ----------------------------------------------------------------
            # Check 1: normalised distance limit
            # Refuse extrapolation if the query point is far from the boundary
            # relative to the surface's own physical extent.
            # ----------------------------------------------------------------

            surface_diagonal = np.sqrt(x_extent**2 + y_extent**2)
            query_distance = np.sqrt((x - x_b)**2 + (y - y_b)**2)

            if query_distance > distance_threshold * surface_diagonal:
                return _failed

        if limit_consistency:
            # ----------------------------------------------------------------
            # Check 2: Taylor self-consistency
            # Compare first-order (z1) and second-order (z2) predictions.
            # If the quadratic correction is large relative to the total
            # extrapolated change the series is not converging — unreliable.
            # ----------------------------------------------------------------
        
            # Regularise denominator to avoid division by zero when z2 ≈ z_b
            regularisation = 1e-6 * (abs(z_b) + 1.0)
            consistency = abs(z2 - z1) / (abs(z2 - z_b) + regularisation)

            if consistency > consistency_threshold:
                return _failed

        if limit_steepness:
            # ----------------------------------------------------------------
            # Check 3: normalised gradient steepness
            # Refuse if the boundary gradient is anomalously large relative
            # to the surface's own characteristic rate of change.
            # Computed after the distance check to avoid unnecessary work.
            # ----------------------------------------------------------------
        
            z_extent = self._search_z.max() - self._search_z.min()
            g_char_x = z_extent / (x_extent + 1e-14)
            g_char_y = z_extent / (y_extent + 1e-14)
            g_char   = np.sqrt(g_char_x**2 + g_char_y**2)

            # Boundary gradient in physical space via implicit function theorem
            grad_uv = np.array([dSdu[2], dSdv[2]])
            try:
                grad_xy = np.linalg.solve(J, grad_uv)
            except np.linalg.LinAlgError:
                return _failed

            g_boundary = np.sqrt(grad_xy[0]**2 + grad_xy[1]**2)
            steepness = g_boundary / (g_char + 1e-14)

            # Steepness threshold: boundary gradient must not exceed 10x the
            # surface's characteristic gradient. Self-calibrating — surfaces
            # with globally steep gradients get a proportionally larger budget.
            if steepness > 10.0:
                return _failed

        # ----------------------------------------------------------------
        # All checks passed — compute and return extrapolated value
        # ----------------------------------------------------------------
        z = float(z2)

        if self.log_z:
            z = np.pow(10, z)

        if not compute_gradients:
            return z

        # Gradient: use second-order corrected derivatives at (u_b+du, v_b+dv)
        dzdu_ext = dSdu[2] + d2Sdu2[2]  * du + d2Sdudv[2] * dv
        dzdv_ext = dSdv[2] + d2Sdudv[2] * du + d2Sdv2[2]  * dv
        grad_uv_ext = np.array([dzdu_ext, dzdv_ext])

        try:
            grad_xy_ext = np.linalg.solve(J, grad_uv_ext)
        except np.linalg.LinAlgError:
            return _failed

        if self.log_z:
            grad_xy_ext *= z * np.log(10)
        if self.log_x:
            x_phys = np.pow(10, x)
            grad_xy_ext[0] /= (x_phys * np.log(10))
        if self.log_y:
            y_phys = np.pow(10, y)
            grad_xy_ext[1] /= (y_phys * np.log(10))

        return z, (float(grad_xy_ext[0]), float(grad_xy_ext[1]))

    def eval(self, x, y, tol=1e-10, max_iter=50, threshold=100,
             compute_gradients=False, extrapolate=False,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Unified evaluation interface that handles both scalar and vector inputs.

        If x and y are scalars, performs a single-point evaluation.
        If x or y are iterables, performs a grid evaluation.

        Parameters
        ----------
        x, y : float or array-like
            Target coordinates in physical space.
        tol : float
            Convergence tolerance on ||F(u,v)||.
        max_iter : int
            Maximum Newton iterations.
        threshold : int
            The total number of points above which vectorized eval_grid is used.
        compute_gradients : bool
            If True, also return (dz/dx, dz/dy).
        extrapolate : bool
            If True, allow (u, v) to leave the [0, 1] knot domain.
        limit_distance, limit_consistency, limit_steepness : bool
            Reliability checks for extrapolation.
        consistency_threshold, distance_threshold, steepness_threshold : float
            Thresholds for reliability checks.

        Returns
        -------
        result : float, tuple, or multiple ndarrays
            Depending on input type and compute_gradients flag.
        """
        eval_params = dict(
            tol=tol, max_iter=max_iter,
            compute_gradients=compute_gradients, extrapolate=extrapolate,
            limit_distance=limit_distance, limit_consistency=limit_consistency,
            limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
            distance_threshold=distance_threshold, steepness_threshold=steepness_threshold
        )

        scalar_x = not hasattr(x, '__iter__')
        scalar_y = not hasattr(y, '__iter__')

        if scalar_x and scalar_y:
            return self.eval_point(x, y, **eval_params)

        x_vals = np.atleast_1d(x)
        y_vals = np.atleast_1d(y)

        if len(x_vals) * len(y_vals) >= threshold:
            return self.eval_grid(x_vals, y_vals, **eval_params)

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Z = np.zeros_like(X)
        if compute_gradients:
            dZdX = np.zeros_like(X)
            dZdY = np.zeros_like(X)
            for i, j in np.ndindex(X.shape):
                z, (dzdx, dzdy) = self.eval_point(X[i, j], Y[i, j], **eval_params)
                Z[i, j], dZdX[i, j], dZdY[i, j] = z, dzdx, dzdy
            return X, Y, Z, dZdX, dZdY
        else:
            for i, j in np.ndindex(X.shape):
                Z[i, j] = self.eval_point(X[i, j], Y[i, j], **eval_params)
            return X, Y, Z

    def eval_point(self, x, y, tol=1e-10, max_iter=50, compute_gradients=False, extrapolate=False,
                   limit_distance=False, limit_consistency=False, limit_steepness=False,
                   consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Find z = S_z(u*, v*) where S_x(u*, v*) = x and S_y(u*, v*) = y.

        Parameters
        ----------
        x, y : float
            Target coordinates in physical space.
        tol : float
        max_iter : int
        compute_gradients : bool
            If True, also return (dz/dx, dz/dy) via the implicit function theorem.

        Returns
        -------
        z : float, or None if Newton did not converge.
        (dzdx, dzdy) : tuple of float, only if compute_gradients=True.
        """
        # --- Convert to log-space for internal search ---------------------
        x_phys, y_phys = x, y
        if self.log_x: x = np.log10(x)
        if self.log_y: y = np.log10(y)

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
            ## no solution found (point may be ouside the defined domain)
            if extrapolate:
                return self._extrapolate_point(
                    x, y, compute_gradients=compute_gradients,
                    limit_distance=limit_distance, limit_consistency=limit_consistency,
                    limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
                    distance_threshold=distance_threshold, steepness_threshold=steepness_threshold)
            else:
                if compute_gradients:
                    return None, (None, None)
                else:
                    return None

        z = float(bisplev(u, v, self._tck_z))

        if not compute_gradients:
            if self.log_z:
                z = np.pow(10, z)
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
            z = np.pow(10, z)
            grad_xy *= z * np.log(10)
        if self.log_x:
            grad_xy[0] /= (x_phys * np.log(10))
        if self.log_y:
            grad_xy[1] /= (y_phys * np.log(10))

        return z, (float(grad_xy[0]), float(grad_xy[1]))

    def eval_grid(self, x_vals, y_vals, tol=1e-10, max_iter=50, 
                  compute_gradients=False, extrapolate=False,
                  limit_distance=False, limit_consistency=False, limit_steepness=False,
                  consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
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
        # --- Convert to log-space for internal search ---------------------
        x_phys_vals = np.asarray(x_vals, dtype=float)
        y_phys_vals = np.asarray(y_vals, dtype=float)
        if self.log_x: x_vals = np.log10(x_phys_vals)
        if self.log_y: y_vals = np.log10(y_phys_vals)

        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        X_phys, Y_phys = np.meshgrid(x_phys_vals, y_phys_vals, indexing='ij')
        shape = X.shape
        x_flat = X.ravel()
        y_flat = Y.ravel()
        x_phys_flat = X_phys.ravel()
        y_phys_flat = Y_phys.ravel()
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

        # --- Handle non-converged (exterior) points ----------------------
        failed = active  # points that never converged
        extrap_cache = {}  # fi -> (z, dzdx, dzdy) for gradient reuse

        if failed.any():
            if extrapolate:
                failed_idx = np.where(failed)[0]
                for fi in failed_idx:
                    result = self._extrapolate_point(
                        x_flat[fi], y_flat[fi],
                        compute_gradients=compute_gradients,
                        limit_distance=limit_distance, 
                        limit_consistency=limit_consistency,
                        limit_steepness=limit_steepness, 
                        consistency_threshold=consistency_threshold,
                        distance_threshold=distance_threshold, 
                        steepness_threshold=steepness_threshold)
                    if compute_gradients:
                        z_ext, (dzdx_ext, dzdy_ext) = result
                        if z_ext is not None:
                            Z_flat[fi] = np.log10(z_ext) if self.log_z else z_ext
                            extrap_cache[fi] = (dzdx_ext, dzdy_ext)
                        else:
                            Z_flat[fi] = np.nan
                    else:
                        if result is not None:
                            Z_flat[fi] = np.log10(result) if self.log_z else result
                        else:
                            Z_flat[fi] = np.nan
            else:
                Z_flat[failed] = np.nan

        if not compute_gradients:
            if self.log_z:
                Z_flat = np.pow(10, Z_flat)

            return X_phys, Y_phys, Z_flat.reshape(shape)

        # --- Gradients via implicit function theorem ---------------------
        dzdx_flat = np.full(n, np.nan)
        dzdy_flat = np.full(n, np.nan)

        # Converged (interior) points: use Jacobian-based gradients
        conv = ~failed
        if conv.any():
            conv_idx = np.where(conv)[0]
            dzdu = np.array([bisplev(u[i], v[i], self._tck_z, dx=1, dy=0)
                            for i in conv_idx])
            dzdv = np.array([bisplev(u[i], v[i], self._tck_z, dx=0, dy=1)
                            for i in conv_idx])
            grad_uv = np.stack([dzdu, dzdv], axis=1)

            det = (J_final[conv_idx, 0, 0] * J_final[conv_idx, 1, 1] -
                   J_final[conv_idx, 0, 1] * J_final[conv_idx, 1, 0])
            safe = np.abs(det) > 1e-14

            dzdx_flat[conv_idx] = np.where(
                safe,
                ( J_final[conv_idx, 1, 1] * grad_uv[:, 0] -
                  J_final[conv_idx, 0, 1] * grad_uv[:, 1]) / det,
                np.nan)
            dzdy_flat[conv_idx] = np.where(
                safe,
                (-J_final[conv_idx, 1, 0] * grad_uv[:, 0] +
                  J_final[conv_idx, 0, 0] * grad_uv[:, 1]) / det,
                np.nan)

        # Extrapolated points: reuse cached gradients from the first pass
        for fi, (dzdx_ext, dzdy_ext) in extrap_cache.items():
            if dzdx_ext is not None:
                dzdx_flat[fi] = dzdx_ext
                dzdy_flat[fi] = dzdy_ext

        if self.log_z:
            Z_flat = np.pow(10, Z_flat)
            # Only apply log_z correction to converged points — extrapolated
            # gradients already have it baked in from _extrapolate_point
            if conv.any():
                dzdx_flat[conv] *= Z_flat[conv] * np.log(10)
                dzdy_flat[conv] *= Z_flat[conv] * np.log(10)
        # Only apply log_x/log_y correction to converged points —
        # extrapolated gradients already have it baked in from _extrapolate_point
        if self.log_x:
            if conv.any():
                dzdx_flat[conv] /= (x_phys_flat[conv] * np.log(10))
        if self.log_y:
            if conv.any():
                dzdy_flat[conv] /= (y_phys_flat[conv] * np.log(10))

        return (X_phys, Y_phys, Z_flat.reshape(shape),
                dzdx_flat.reshape(shape), dzdy_flat.reshape(shape))
