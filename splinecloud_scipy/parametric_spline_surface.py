import requests, json
import numpy as np
from scipy.interpolate import bisplev


class ParametricBivariateSpline:
    """
    Bivariate B-spline surface using explicit knot vectors and control points.

    Internally stores three scipy-compatible tck tuples — one per output
    coordinate (x1, x2, y) — and evaluates via bisplev.
    """

    def __init__(self, tu, tv, cp, ku, kv, w=None,
                 log_x1=False, log_x2=False, log_y=False):
        """
        Parameters
        ----------
        tu, tv : 1-D array-like
            Knot vectors along the u and v axes.
        cp : array-like, shape (nu, nv, 3)
            Control-point grid. cp[i, j] = [x1, y, x2].
        ku, kv : int
            Spline degrees along u and v.
        w : array-like, shape (nu, nv), optional
            NURBS weights (stored for future use; all ≈ 1 in current data).
        log_x1, log_x2, log_y : bool
            Whether each output axis uses a logarithmic scale.
        """
        cp = np.asarray(cp, dtype=float)
        nu, nv, _ = cp.shape

        self.tu = np.asarray(tu, dtype=float)
        self.tv = np.asarray(tv, dtype=float)
        self.ku = int(ku)
        self.kv = int(kv)
        self.w = np.asarray(w, dtype=float) if w is not None else np.ones((nu, nv))
        self.log_x1, self.log_x2, self.log_y = log_x1, log_x2, log_y

        self._tck_x1 = (self.tu, self.tv, cp[:, :, 0].ravel(), self.ku, self.kv)
        self._tck_y  = (self.tu, self.tv, cp[:, :, 1].ravel(), self.ku, self.kv)
        self._tck_x2 = (self.tu, self.tv, cp[:, :, 2].ravel(), self.ku, self.kv)

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
        x1, x2, y : ndarrays of computed coordinates with shape (len(u), len(v))
                     or scalars if u and v are scalars
        """

        u_in = np.asarray(u, dtype=float)
        v_in = np.asarray(v, dtype=float)
        scalar_input = u_in.ndim == 0 and v_in.ndim == 0

        u_arr = np.atleast_1d(u_in)
        v_arr = np.atleast_1d(v_in)

        x1 = bisplev(u_arr, v_arr, self._tck_x1)
        x2 = bisplev(u_arr, v_arr, self._tck_x2)
        y  = bisplev(u_arr, v_arr, self._tck_y)

        if self.log_x1: x1 = np.pow(10, x1)
        if self.log_x2: x2 = np.pow(10, x2)
        if self.log_y:  y  = np.pow(10, y)

        if scalar_input:
            return float(x1), float(x2), float(y)
        
        return x1, x2, y

    def _build_search_grid(self):
        """
        Pre-evaluate the surface at the centres of all knot spans.
        Called once at construction; result cached for all eval() calls.

        Stores
        ------
        _search_u  : (Mu,)    u midpoints of each unique u span
        _search_v  : (Mv,)    v midpoints of each unique v span
        _search_x1 : (Mu, Mv) physical x1 at each (u_mid, v_mid)
        _search_x2 : (Mu, Mv) physical x2 at each (u_mid, v_mid)
        """
        tu_unique = np.unique(self.tu)
        tv_unique = np.unique(self.tv)

        self._search_u = (tu_unique[:-1] + tu_unique[1:]) / 2   # (Mu,)
        self._search_v = (tv_unique[:-1] + tv_unique[1:]) / 2   # (Mv,)

        # bisplev returns (Mu, Mv) grid if Mu, Mv > 1, but may return scalar/1D if they are 1.
        # Ensure result is always (Mu, Mv)
        self._search_x1 = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_x1)).reshape(len(self._search_u), len(self._search_v))
        self._search_x2 = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_x2)).reshape(len(self._search_u), len(self._search_v))
        self._search_y  = np.atleast_2d(bisplev(self._search_u, self._search_v, self._tck_y)).reshape(len(self._search_u), len(self._search_v))

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

    def _compute_boundary_point(self, x1, x2):
        """
        Find the nearest point on the surface boundary in physical (x1, x2)
        space, refined with 1D Newton, and return all derivatives needed
        for second-order Taylor extrapolation.

        Returns
        -------
        S_b    : ndarray (3,)  — [x1_b, x2_b, y_b] at boundary
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
            x1_edge = np.array([float(bisplev(ui, vi, self._tck_x1))
                            for ui, vi in zip(u_edge, v_edge)])
            x2_edge = np.array([float(bisplev(ui, vi, self._tck_x2))
                            for ui, vi in zip(u_edge, v_edge)])
            dist2 = (x1_edge - x1)**2 + (x2_edge - x2)**2
            idx = np.argmin(dist2)
            if dist2[idx] < best_dist2:
                best_dist2 = dist2[idx]
                best_u, best_v = u_edge[idx], v_edge[idx]
                best_edge = edge_name

        # --- Refine with 1D Newton along the boundary edge ---------------
        u_b, v_b = self._refine_boundary_point(
            x1, x2, best_u, best_v, best_edge,
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

        x1b, dx1du, dx1dv, d2x1du2, d2x1dv2, d2x1dudv = _eval_derivs(self._tck_x1, u_b, v_b)
        x2b, dx2du, dx2dv, d2x2du2, d2x2dv2, d2x2dudv = _eval_derivs(self._tck_x2, u_b, v_b)
        yb,  dydu,  dydv,  d2ydu2,  d2ydv2,  d2ydydv  = _eval_derivs(self._tck_y,  u_b, v_b)

        # --- Solve for (du_ext, dv_ext) to match target (x1, x2) ----------
        # First-order solve: J * [du, dv]^T = [x1 - x1b, x2 - x2b]^T
        # then refine with second-order correction
        J = np.array([[dx1du, dx1dv], [dx2du, dx2dv]])
        rhs = np.array([x1 - x1b, x2 - x2b])

        try:
            d_params = np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            d_params = np.zeros(2)

        du_ext, dv_ext = d_params

        S_b     = np.array([x1b,       x2b,       yb      ])
        dSdu    = np.array([dx1du,     dx2du,     dydu    ])
        dSdv    = np.array([dx1dv,     dx2dv,     dydv    ])
        d2Sdu2  = np.array([d2x1du2,   d2x2du2,   d2ydu2  ])
        d2Sdv2  = np.array([d2x1dv2,   d2x2dv2,   d2ydv2  ])
        d2Sdudv = np.array([d2x1dudv,  d2x2dudv,  d2ydydv ])

        return S_b, dSdu, dSdv, d2Sdu2, d2Sdv2, d2Sdudv, du_ext, dv_ext

    def _refine_boundary_point(self, x1, x2, u0, v0, edge,
            u_min, u_max, v_min, v_max, tol=1e-10, max_iter=50):
        """
        Refine boundary point with 1D Newton along the free parameter
        of the detected edge.
        """
        u, v = u0, v0
        free_u = edge in ('v_min', 'v_max')   # u is free parameter

        for _ in range(max_iter):
            x1b = float(bisplev(u, v, self._tck_x1))
            x2b = float(bisplev(u, v, self._tck_x2))

            if free_u:
                dx1dt = float(bisplev(u, v, self._tck_x1, dx=1, dy=0))
                dx2dt = float(bisplev(u, v, self._tck_x2, dx=1, dy=0))
                d2x1dt2 = self._bisplev_d2(u, v, self._tck_x1, du=2, dv=0)
                d2x2dt2 = self._bisplev_d2(u, v, self._tck_x2, du=2, dv=0)
            else:
                dx1dt = float(bisplev(u, v, self._tck_x1, dx=0, dy=1))
                dx2dt = float(bisplev(u, v, self._tck_x2, dx=0, dy=1))
                d2x1dt2 = self._bisplev_d2(u, v, self._tck_x1, du=0, dv=2)
                d2x2dt2 = self._bisplev_d2(u, v, self._tck_x2, du=0, dv=2)

            grad = (x1b - x1) * dx1dt + (x2b - x2) * dx2dt
            hess = (dx1dt**2 + dx2dt**2
                    + (x1b - x1) * d2x1dt2
                    + (x2b - x2) * d2x2dt2)

            if abs(hess) < 1e-14 or abs(grad) < tol:
                break

            step = grad / hess
            if free_u:
                u = np.clip(u - step, u_min, u_max)
            else:
                v = np.clip(v - step, v_min, v_max)

        return u, v

    def _extrapolate_point(self, x1, x2, compute_gradients=False,
            limit_distance=False, limit_consistency=False, limit_steepness=False,
            consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Extrapolate y at (x1, x2) outside the surface domain using a
        second-order Taylor expansion from the nearest boundary point,
        with reliability checks.

        Parameters
        ----------
        x1, x2 : float
            Target coordinates in internal (log-) space — callers must
            convert physical values via log10() before passing them in.
        compute_gradients : bool
            If True, also return (dy/dx1, dy/dx2).
        consistency_threshold : float
            Maximum allowed ratio |y2 - y1| / |y2 - y_b| before the
            extrapolation is considered unreliable. Default 0.2 means
            the quadratic correction must be less than 20% of the total
            extrapolated change. Dimensionless and self-calibrating.
        distance_threshold : float
            Maximum allowed extrapolation distance as a fraction of the
            surface's physical diagonal extent. Default 0.5 means the
            query point must be within half the surface's own size.

        Returns
        -------
        y : float or None
            None if any reliability check fails.
        (dydx1, dydx2) : tuple of float or (None, None),
            only when compute_gradients=True.
        """
        _failed = (None, (None, None)) if compute_gradients else None

        S_b, dSdu, dSdv, d2Sdu2, d2Sdv2, d2Sdudv, du, dv = self._compute_boundary_point(x1, x2)

        J = np.array([[dSdu[0], dSdv[0]], [dSdu[1], dSdv[1]]])
        first_order  = dSdu[2] * du + dSdv[2] * dv
        second_order = 0.5 * d2Sdu2[2] * du**2 + d2Sdudv[2] * du * dv + 0.5 * d2Sdv2[2] * dv**2

        x1_b, x2_b, y_b = S_b
        y1 = y_b + first_order
        y2 = y_b + first_order + second_order

        x1_extent = self._search_x1.max() - self._search_x1.min()
        x2_extent = self._search_x2.max() - self._search_x2.min()

        if limit_distance:
            # ----------------------------------------------------------------
            # Check 1: normalised distance limit
            # Refuse extrapolation if the query point is far from the boundary
            # relative to the surface's own physical extent.
            # ----------------------------------------------------------------

            surface_diagonal = np.sqrt(x1_extent**2 + x2_extent**2)
            query_distance = np.sqrt((x1 - x1_b)**2 + (x2 - x2_b)**2)

            if query_distance > distance_threshold * surface_diagonal:
                return _failed

        if limit_consistency:
            # ----------------------------------------------------------------
            # Check 2: Taylor self-consistency
            # Compare first-order (y1) and second-order (y2) predictions.
            # If the quadratic correction is large relative to the total
            # extrapolated change the series is not converging — unreliable.
            # ----------------------------------------------------------------
        
            # Regularise denominator to avoid division by zero when y2 ≈ y_b
            regularisation = 1e-6 * (abs(y_b) + 1.0)
            consistency = abs(y2 - y1) / (abs(y2 - y_b) + regularisation)

            if consistency > consistency_threshold:
                return _failed

        if limit_steepness:
            # ----------------------------------------------------------------
            # Check 3: normalised gradient steepness
            # Refuse if the boundary gradient is anomalously large relative
            # to the surface's own characteristic rate of change.
            # Computed after the distance check to avoid unnecessary work.
            # ----------------------------------------------------------------
        
            y_extent = self._search_y.max() - self._search_y.min()
            g_char_x1 = y_extent / (x1_extent + 1e-14)
            g_char_x2 = y_extent / (x2_extent + 1e-14)
            g_char    = np.sqrt(g_char_x1**2 + g_char_x2**2)

            # Boundary gradient in physical space via implicit function theorem
            grad_uv = np.array([dSdu[2], dSdv[2]])
            try:
                grad_x1x2 = np.linalg.solve(J, grad_uv)
            except np.linalg.LinAlgError:
                return _failed

            g_boundary = np.sqrt(grad_x1x2[0]**2 + grad_x1x2[1]**2)
            steepness = g_boundary / (g_char + 1e-14)

            # Steepness threshold: boundary gradient must not exceed 10x the
            # surface's characteristic gradient. Self-calibrating — surfaces
            # with globally steep gradients get a proportionally larger budget.
            if steepness > 10.0:
                return _failed

        # ----------------------------------------------------------------
        # All checks passed — compute and return extrapolated value
        # ----------------------------------------------------------------
        y = float(y2)

        if self.log_y:
            y = np.pow(10, y)

        if not compute_gradients:
            return y

        # Gradient: use second-order corrected derivatives at (u_b+du, v_b+dv)
        dydu_ext = dSdu[2] + d2Sdu2[2]  * du + d2Sdudv[2] * dv
        dydv_ext = dSdv[2] + d2Sdudv[2] * du + d2Sdv2[2]  * dv
        grad_uv_ext = np.array([dydu_ext, dydv_ext])

        try:
            grad_x1x2_ext = np.linalg.solve(J, grad_uv_ext)
        except np.linalg.LinAlgError:
            return _failed

        if self.log_y:
            grad_x1x2_ext *= y * np.log(10)
        if self.log_x1:
            x1_phys = np.pow(10, x1)
            grad_x1x2_ext[0] /= (x1_phys * np.log(10))
        if self.log_x2:
            x2_phys = np.pow(10, x2)
            grad_x1x2_ext[1] /= (x2_phys * np.log(10))

        return y, (float(grad_x1x2_ext[0]), float(grad_x1x2_ext[1]))

    def eval(self, x1, x2, tol=1e-10, max_iter=50, threshold=100,
             compute_gradients=False, extrapolate=False,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Unified evaluation interface that handles both scalar and vector inputs.

        If x1 and x2 are scalars, performs a single-point evaluation.
        If x1 or x2 are iterables, performs a grid evaluation.

        Parameters
        ----------
        x1, x2 : float or array-like
            Target coordinates in physical space.
        tol : float
            Convergence tolerance on ||F(u,v)||.
        max_iter : int
            Maximum Newton iterations.
        threshold : int
            The total number of points above which vectorized eval_grid is used.
        compute_gradients : bool
            If True, also return (dy/dx1, dy/dx2).
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

        scalar_x1 = not hasattr(x1, '__iter__')
        scalar_x2 = not hasattr(x2, '__iter__')

        if scalar_x1 and scalar_x2:
            return self.eval_point(x1, x2, **eval_params)

        x1_vals = np.atleast_1d(x1)
        x2_vals = np.atleast_1d(x2)

        if len(x1_vals) * len(x2_vals) >= threshold:
            return self.eval_grid(x1_vals, x2_vals, **eval_params)

        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        Y = np.zeros_like(X1)
        if compute_gradients:
            dYdX1 = np.zeros_like(X1)
            dYdX2 = np.zeros_like(X1)
            for i, j in np.ndindex(X1.shape):
                y, (dydx1, dydx2) = self.eval_point(X1[i, j], X2[i, j], **eval_params)
                Y[i, j], dYdX1[i, j], dYdX2[i, j] = y, dydx1, dydx2
            return X1, X2, Y, dYdX1, dYdX2
        else:
            for i, j in np.ndindex(X1.shape):
                Y[i, j] = self.eval_point(X1[i, j], X2[i, j], **eval_params)
            return X1, X2, Y

    def eval_point(self, x1, x2, tol=1e-10, max_iter=50, compute_gradients=False, extrapolate=False,
                   limit_distance=False, limit_consistency=False, limit_steepness=False,
                   consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Find y = S_y(u*, v*) where S_x1(u*, v*) = x1 and S_x2(u*, v*) = x2.

        Parameters
        ----------
        x1, x2 : float
            Target coordinates in physical space.
        tol : float
        max_iter : int
        compute_gradients : bool
            If True, also return (dy/dx1, dy/dx2) via the implicit function theorem.

        Returns
        -------
        y : float, or None if Newton did not converge.
        (dydx1, dydx2) : tuple of float, only if compute_gradients=True.
        """
        # --- Convert to log-space for internal search ---------------------
        x1_phys, x2_phys = x1, x2
        if self.log_x1: x1 = np.log10(x1)
        if self.log_x2: x2 = np.log10(x2)

        # --- Knot domain boundaries ---
        u_min, u_max = self.tu[self.ku], self.tu[-(self.ku + 1)]
        v_min, v_max = self.tv[self.kv], self.tv[-(self.kv + 1)]

        # --- Initial guess from search grid ---
        sx1 = self._search_x1.ravel()
        sx2 = self._search_x2.ravel()
        su = np.repeat(self._search_u, len(self._search_v))
        sv = np.tile(self._search_v, len(self._search_u))

        dist2 = (sx1 - x1)**2 + (sx2 - x2)**2
        best  = np.argmin(dist2)
        u, v  = su[best], sv[best]

        # --- Newton's method — keep final Jacobian if needed ---
        # Use relative tolerance to handle large physical-space values
        tol_x1 = tol * (1 + abs(x1))
        tol_x2 = tol * (1 + abs(x2))

        J = None
        for _ in range(max_iter):
            fx1 = float(bisplev(u, v, self._tck_x1)) - x1
            fx2 = float(bisplev(u, v, self._tck_x2)) - x2

            if abs(fx1) < tol_x1 and abs(fx2) < tol_x2:
                break

            dx1du = float(bisplev(u, v, self._tck_x1, dx=1, dy=0))
            dx1dv = float(bisplev(u, v, self._tck_x1, dx=0, dy=1))
            dx2du = float(bisplev(u, v, self._tck_x2, dx=1, dy=0))
            dx2dv = float(bisplev(u, v, self._tck_x2, dx=0, dy=1))

            J = np.array([[dx1du, dx1dv], [dx2du, dx2dv]])
            F = np.array([fx1, fx2])

            try:
                delta = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break

            u = np.clip(u + delta[0], u_min, u_max)
            v = np.clip(v + delta[1], v_min, v_max)
        else:
            ## no solution found (point may be ouside the defined domain)
            if extrapolate:
                return self._extrapolate_point(
                    x1, x2, compute_gradients=compute_gradients,
                    limit_distance=limit_distance, limit_consistency=limit_consistency,
                    limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
                    distance_threshold=distance_threshold, steepness_threshold=steepness_threshold)
            else:
                if compute_gradients:
                    return None, (None, None)
                else:
                    return None

        y = float(bisplev(u, v, self._tck_y))

        if not compute_gradients:
            if self.log_y:
                y = np.pow(10, y)
            return y

        # --- Gradients via implicit function theorem ---
        dydu = float(bisplev(u, v, self._tck_y, dx=1, dy=0))
        dydv = float(bisplev(u, v, self._tck_y, dx=0, dy=1))
        grad_uv = np.array([dydu, dydv])

        try:
            grad_x1x2 = np.linalg.solve(J, grad_uv)
        except np.linalg.LinAlgError:
            grad_x1x2 = np.array([np.nan, np.nan])

        if self.log_y:
            y = np.pow(10, y)
            grad_x1x2 *= y * np.log(10)
        if self.log_x1:
            grad_x1x2[0] /= (x1_phys * np.log(10))
        if self.log_x2:
            grad_x1x2[1] /= (x2_phys * np.log(10))

        return y, (float(grad_x1x2[0]), float(grad_x1x2[1]))

    def eval_grid(self, x1_vals, x2_vals, tol=1e-10, max_iter=50, 
                  compute_gradients=False, extrapolate=False,
                  limit_distance=False, limit_consistency=False, limit_steepness=False,
                  consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10):
        """
        Evaluate y over a regular (x1, x2) grid using vectorized Newton.

        Parameters
        ----------
        x1_vals : 1-D array of shape (Nx1,)
        x2_vals : 1-D array of shape (Nx2,)
        tol : float
        max_iter : int
        compute_gradients : bool
            If True, also return dy/dx1 and dy/dx2 grids.

        Returns
        -------
        X1, X2, Y : 2-D arrays of shape (Nx1, Nx2)
        dYdX1, dYdX2 : 2-D arrays of shape (Nx1, Nx2), only if compute_gradients=True.
        """
        # --- Convert to log-space for internal search ---------------------
        x1_phys_vals = np.asarray(x1_vals, dtype=float)
        x2_phys_vals = np.asarray(x2_vals, dtype=float)
        if self.log_x1: x1_vals = np.log10(x1_phys_vals)
        if self.log_x2: x2_vals = np.log10(x2_phys_vals)

        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        X1_phys, X2_phys = np.meshgrid(x1_phys_vals, x2_phys_vals, indexing='ij')
        shape = X1.shape
        x1_flat = X1.ravel()
        x2_flat = X2.ravel()
        x1_phys_flat = X1_phys.ravel()
        x2_phys_flat = X2_phys.ravel()
        n = len(x1_flat)

        # --- Knot domain boundaries ---
        u_min, u_max = self.tu[self.ku], self.tu[-(self.ku + 1)]
        v_min, v_max = self.tv[self.kv], self.tv[-(self.kv + 1)]

        # --- Vectorized initial guess ------------------------------------
        sx1 = self._search_x1.ravel()
        sx2 = self._search_x2.ravel()
        su = np.repeat(self._search_u, len(self._search_v))
        sv = np.tile(self._search_v, len(self._search_u))

        dist2 = (sx1[None, :] - x1_flat[:, None])**2 + \
                (sx2[None, :] - x2_flat[:, None])**2
        best = np.argmin(dist2, axis=1)
        u = su[best].copy()
        v = sv[best].copy()

        # Store final Jacobian only when gradients are needed
        J_final = np.zeros((n, 2, 2)) if compute_gradients else None

        # --- Vectorized Newton -------------------------------------------
        # Use relative tolerance to handle large physical-space values
        tol_x1 = tol * (1 + np.abs(x1_flat))
        tol_x2 = tol * (1 + np.abs(x2_flat))

        active = np.ones(n, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break

            idx = np.where(active)[0]
            ua, va = u[idx], v[idx]
            x1a, x2a = x1_flat[idx], x2_flat[idx]

            fx1 = np.array([bisplev(ui, vi, self._tck_x1)
                        for ui, vi in zip(ua, va)]) - x1a
            fx2 = np.array([bisplev(ui, vi, self._tck_x2)
                        for ui, vi in zip(ua, va)]) - x2a

            converged = (np.abs(fx1) < tol_x1[idx]) & (np.abs(fx2) < tol_x2[idx])
            active[idx[converged]] = False

            still = ~converged
            if not still.any():
                break

            ua, va = ua[still], va[still]
            fx1, fx2 = fx1[still], fx2[still]

            dx1du = np.array([bisplev(ui, vi, self._tck_x1, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dx1dv = np.array([bisplev(ui, vi, self._tck_x1, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])
            dx2du = np.array([bisplev(ui, vi, self._tck_x2, dx=1, dy=0)
                            for ui, vi in zip(ua, va)])
            dx2dv = np.array([bisplev(ui, vi, self._tck_x2, dx=0, dy=1)
                            for ui, vi in zip(ua, va)])

            if compute_gradients:
                still_idx = idx[still]
                J_final[still_idx, 0, 0] = dx1du
                J_final[still_idx, 0, 1] = dx1dv
                J_final[still_idx, 1, 0] = dx2du
                J_final[still_idx, 1, 1] = dx2dv

            det = dx1du * dx2dv - dx1dv * dx2du
            safe = np.abs(det) > 1e-14
            du = np.where(safe, ( dx2dv * (-fx1) - dx1dv * (-fx2)) / det, 0.0)
            dv = np.where(safe, (-dx2du * (-fx1) + dx1du * (-fx2)) / det, 0.0)

            u[idx[still]] = np.clip(u[idx[still]] + du, u_min, u_max)
            v[idx[still]] = np.clip(v[idx[still]] + dv, v_min, v_max)

        # --- Final y evaluation ------------------------------------------
        Y_flat = np.array([bisplev(ui, vi, self._tck_y)
                          for ui, vi in zip(u, v)], dtype=float)

        # --- Handle non-converged (exterior) points ----------------------
        failed = active  # points that never converged
        extrap_cache = {}  # fi -> (dydx1, dydx2) for gradient reuse

        if failed.any():
            if extrapolate:
                failed_idx = np.where(failed)[0]
                for fi in failed_idx:
                    result = self._extrapolate_point(
                        x1_flat[fi], x2_flat[fi],
                        compute_gradients=compute_gradients,
                        limit_distance=limit_distance, 
                        limit_consistency=limit_consistency,
                        limit_steepness=limit_steepness, 
                        consistency_threshold=consistency_threshold,
                        distance_threshold=distance_threshold, 
                        steepness_threshold=steepness_threshold)
                    if compute_gradients:
                        y_ext, (dydx1_ext, dydx2_ext) = result
                        if y_ext is not None:
                            Y_flat[fi] = np.log10(y_ext) if self.log_y else y_ext
                            extrap_cache[fi] = (dydx1_ext, dydx2_ext)
                        else:
                            Y_flat[fi] = np.nan
                    else:
                        if result is not None:
                            Y_flat[fi] = np.log10(result) if self.log_y else result
                        else:
                            Y_flat[fi] = np.nan
            else:
                Y_flat[failed] = np.nan

        if not compute_gradients:
            if self.log_y:
                Y_flat = np.pow(10, Y_flat)

            return X1_phys, X2_phys, Y_flat.reshape(shape)

        # --- Gradients via implicit function theorem ---------------------
        dydx1_flat = np.full(n, np.nan)
        dydx2_flat = np.full(n, np.nan)

        # Converged (interior) points: use Jacobian-based gradients
        conv = ~failed
        if conv.any():
            conv_idx = np.where(conv)[0]
            dydu = np.array([bisplev(u[i], v[i], self._tck_y, dx=1, dy=0)
                            for i in conv_idx])
            dydv = np.array([bisplev(u[i], v[i], self._tck_y, dx=0, dy=1)
                            for i in conv_idx])
            grad_uv = np.stack([dydu, dydv], axis=1)

            det = (J_final[conv_idx, 0, 0] * J_final[conv_idx, 1, 1] -
                   J_final[conv_idx, 0, 1] * J_final[conv_idx, 1, 0])
            safe = np.abs(det) > 1e-14

            dydx1_flat[conv_idx] = np.where(
                safe,
                ( J_final[conv_idx, 1, 1] * grad_uv[:, 0] -
                  J_final[conv_idx, 0, 1] * grad_uv[:, 1]) / det,
                np.nan)
            dydx2_flat[conv_idx] = np.where(
                safe,
                (-J_final[conv_idx, 1, 0] * grad_uv[:, 0] +
                  J_final[conv_idx, 0, 0] * grad_uv[:, 1]) / det,
                np.nan)

        # Extrapolated points: reuse cached gradients from the first pass
        for fi, (dydx1_ext, dydx2_ext) in extrap_cache.items():
            if dydx1_ext is not None:
                dydx1_flat[fi] = dydx1_ext
                dydx2_flat[fi] = dydx2_ext

        if self.log_y:
            Y_flat = np.pow(10, Y_flat)
            # Only apply log_y correction to converged points — extrapolated
            # gradients already have it baked in from _extrapolate_point
            if conv.any():
                dydx1_flat[conv] *= Y_flat[conv] * np.log(10)
                dydx2_flat[conv] *= Y_flat[conv] * np.log(10)
        # Only apply log_x1/log_x2 correction to converged points —
        # extrapolated gradients already have it baked in from _extrapolate_point
        if self.log_x1:
            if conv.any():
                dydx1_flat[conv] /= (x1_phys_flat[conv] * np.log(10))
        if self.log_x2:
            if conv.any():
                dydx2_flat[conv] /= (x2_phys_flat[conv] * np.log(10))

        return (X1_phys, X2_phys, Y_flat.reshape(shape),
                dydx1_flat.reshape(shape), dydx2_flat.reshape(shape))


