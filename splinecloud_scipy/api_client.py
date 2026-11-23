# -*- coding: utf-8 -*-
import re
import requests, json
import numpy as np

from .parametric_spline import ParametricUnivariateSpline
from .explicit_spline import ExplicitUnivariateSpline, NdBsplExplicit
from .parametric_spline_surface import ParametricBivariateSpline
from .parametric_spline_volume import ParametricSplineVolume


SPLINECLOUD_API_URL = "https://splinecloud.com/api"


SPLINE_FORMS = ("parametric", "explicit")
## lofted surfaces: control net in role order [x1, x2, y] (SplineCloud contract C.3)
PARAMETRIC_SURFACE_VERSION = 1
## extrapolate=True of the legacy surface eval maps to the safest explicit policy
LEGACY_EXTRAPOLATION = "linear"


class _SurfaceMetadataMixin:
    """SplineCloud metadata shared by lofted (parametric) and fitted (explicit) surfaces:
    identification, relation, labels and scales, and loading the underlying data."""

    def _init_metadata(self, data):
        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("description")
        self.form = data.get("form", "parametric")
        self.surface_type = data.get("surface_type")

        # Relational data
        self.subset_uids = data.get("subset_uids", [])
        self.x2_values = data.get("x2_values", [])
        self.curve_uids = data.get("curve_uids", [])
        self.relation_uid = data.get("relation_uid")
        self.relation_name = data.get("relation_name")

        # Labels and scales
        labels = data.get("labels", {}) or {}
        self.x1_label = labels.get("x1")
        self.x2_label = labels.get("x2")
        self.y_label = labels.get("y")
        self.scale_x1 = data.get("scale_x1", "Linear")
        self.scale_x2 = data.get("scale_x2", "Linear")
        self.scale_y = data.get("scale_y", "Linear")

        self.subsets = []

    @property
    def labels(self):
        return [self.x1_label, self.x2_label, self.y_label]

    def load_subsets(self):
        """
        Load datasets (subsets) associated with this surface from the API.

        This method fetches all subsets identified by `subset_uids` and caches
        them in `self.subsets` for subsequent calls.

        Returns
        -------
        list of tuples
            A list of (columns, table) tuples, where each tuple represents
            a loaded subset.
        """
        if self.subsets:
            return self.subsets

        self.subsets = [load_subset(uid) for uid in self.subset_uids]
        return self.subsets


class SplineSurface(_SurfaceMetadataMixin, ParametricBivariateSpline):
    """
    A lofted (parametric) spline surface from the SplineCloud API: a
    ParametricBivariateSpline with SplineCloud metadata. The payload is the parametric
    surface schema v1: control net in role order [x1, x2, y], scales [x1, x2, y] (needs
    splinecloud-scipy >= 1.5; earlier versions read the old [x1, y, x2] order).
    """
    def __init__(self, data):
        sp = data["spline"]
        if sp.get("v") != PARAMETRIC_SURFACE_VERSION or sp.get("form", "parametric") != "parametric":
            raise ValueError("Unsupported lofted surface payload (schema version {}): this "
                             "client reads the parametric surface schema v1".format(sp.get("v")))
        cp = np.array(sp["cp"], dtype=float)   # (nu, nv, 3) = [x1, x2, y]
        w  = np.array(sp["w"],  dtype=float)   # (nu, nv)
        tu = np.array(sp["tu"], dtype=float)
        tv = np.array(sp["tv"], dtype=float)
        ku = int(sp["ku"])
        kv = int(sp["kv"])

        scales = sp.get("scales") or [data.get("scale_x1"), data.get("scale_x2"),
                                      data.get("scale_y")]
        log_x1, log_x2, log_y = (scale == "Logarithmic" for scale in scales)

        ParametricBivariateSpline.__init__(
            self, tu, tv, cp, ku, kv, w=w,
            log_x1=log_x1, log_x2=log_x2, log_y=log_y,
        )
        self._init_metadata(data)

    def load_data(self):
        """
        Load and assemble raw data points from all associated subsets.

        Each subset contains (x1, y) data points. The x2 coordinate for
        each subset is taken from `x2_values`.

        Returns
        -------
        labels : list of str
            A list of [x1, x2, y] coordinate labels.
        data : ndarray
            An (N, 3) array of collected data points.
        """
        if not self.subsets:
            self.load_subsets()

        data = []
        for i, item in enumerate(self.subsets):
            subset_data = item[1]
            for point in subset_data:
                data.append([point[0], self.x2_values[i], point[1]])

        return self.labels, np.array(data)


class ExplicitSplineSurface(_SurfaceMetadataMixin, NdBsplExplicit):
    """
    A fitted (explicit) spline surface y = f(x1, x2) from the SplineCloud API
    (``form == "explicit"``, explicit schema v1 with nx = 2; needs splinecloud-scipy >= 1.5).

    Evaluated directly - no inversion of a parametric map. Drop-in for SplineSurface:
    ``eval(x1, x2)`` keeps the legacy meaning (a scalar for scalar inputs, an outer-product
    grid ``X1, X2, Y`` for arrays). The explicit API is available too: ``f(x1, x2)``
    (pointwise, broadcasting), ``eval_points(points)``, ``eval_grid(x1_vals, x2_vals)``
    (returns Y only), ``grad(points)``, ``slice({axis: value})`` and ``to_parametric()``.

    Fit report: ``method``, ``fit_params``, ``accuracy`` and ``fit_status`` (e.g.
    ``"pending"`` when the relation changed after the fit).
    """

    def __init__(self, data, extrapolation=None):
        payload = data["spline"]
        if payload.get("v") != 1:
            raise ValueError(f"Unsupported explicit spline schema version: {payload.get('v')}")
        if payload.get("nx") != 2 or payload.get("ny", 1) != 1:
            raise ValueError("ExplicitSplineSurface needs an explicit spline with nx = 2, "
                             "ny = 1")
        NdBsplExplicit.__init__(
            self, payload["t"], payload["k"], payload["c"], scales=payload.get("scales"),
            extrapolation=extrapolation or payload.get("extrapolation", "none"))
        self._init_metadata(data)

        self.method = data.get("method")
        self.fit_params = data.get("fit_params")
        self.accuracy = data.get("accuracy")
        self.fit_status = data.get("fit_status")
        self.excluded_points = None     ## set by load_data()
        self.data_format = None         ## set by load_data(): "vector", "matrix" or "stacked"
        self.sources = None             ## set by load_data() for "stacked" relations

    @staticmethod
    def _policy(extrapolate):
        if extrapolate is True:
            return LEGACY_EXTRAPOLATION
        if extrapolate is False:
            return "none"
        return extrapolate              ## None (the object's default) or a policy name

    def eval(self, x1, x2, compute_gradients=False, extrapolate=False, **legacy_options):
        """
        Legacy SplineSurface.eval semantics.

        Scalars give ``y`` (``y, (dy/dx1, dy/dx2)`` with ``compute_gradients``); arrays give
        the outer-product grid ``X1, X2, Y`` (``X1, X2, Y, dYdX1, dYdX2``). Points outside
        the domain give NaN unless ``extrapolate`` is True (linear extrapolation from the
        nearest domain point) or a policy name ("none", "linear", "clamp", "polynomial").
        Options of the parametric evaluator (tol, max_iter, threshold, limit_*) are
        accepted and ignored: explicit evaluation is direct.
        """
        policy = self._policy(extrapolate)

        if not hasattr(x1, '__iter__') and not hasattr(x2, '__iter__'):
            point = np.array([[float(x1), float(x2)]])
            y = float(self.eval_points(point, extrapolate=policy)[0])
            if not compute_gradients:
                return y
            g = self.grad(point)[0]
            return y, (float(g[0]), float(g[1]))

        X1, X2 = np.meshgrid(np.atleast_1d(np.asarray(x1, dtype=float)),
                             np.atleast_1d(np.asarray(x2, dtype=float)), indexing='ij')
        points = np.column_stack([X1.ravel(), X2.ravel()])
        Y = self.eval_points(points, extrapolate=policy).reshape(X1.shape)
        if not compute_gradients:
            return X1, X2, Y
        G = self.grad(points)
        return X1, X2, Y, G[:, 0].reshape(X1.shape), G[:, 1].reshape(X1.shape)

    def eval_point(self, x1, x2, compute_gradients=False, extrapolate=False,
                   **legacy_options):
        """Legacy single-point evaluation (see ``eval``)."""
        return self.eval(float(x1), float(x2), compute_gradients=compute_gradients,
                         extrapolate=extrapolate)

    def load_relation(self):
        """The fitted relation behind this surface (column mapping, scales, point masks), as
        returned by /api/relations/bivariate/fitted/<uid>/."""
        if not self.relation_uid:
            raise ValueError("This surface has no relation to load")
        url = SPLINECLOUD_API_URL + "/relations/bivariate/fitted/{}/".format(self.relation_uid)
        response = requests.get(url, params={"spline": "native"})
        response.raise_for_status()
        return json.loads(response.content)

    def load_data(self):
        """
        Load the data points the surface was fitted to, one ``[x1, x2, y]`` point per row.

        Long-form relations read their mapped columns (x1, x2, y); rows with a gap in one of
        them are skipped. Grid (matrix) relations - x1 down a column, x2 in the column
        headers or in a row, y in the cells - are melted into one point per filled cell,
        row by row; an empty cell is skipped. Stacked relations - several subsets (e.g. the
        contour lines of a chart), each at a constant z - read their columns from every
        subset and take the z role from the subset's z, subset by subset in the relation's
        ``sources`` order (``self.sources``); a subset without a z value is skipped. All
        follow the platform, so the points line up with the relation's exclusion mask,
        stored in ``self.excluded_points`` (None when nothing is excluded).
        ``self.data_format`` is ``"vector"``, ``"matrix"`` or ``"stacked"``.

        Returns
        -------
        labels : list of str
            ``[x1, x2, y]`` labels.
        data : ndarray
            An (N, 3) array of ``[x1, x2, y]`` points.
        """
        relation = self.load_relation()
        self.data_format = relation.get("data_format", "vector")
        mapping = relation["data_mapping"]

        if not self.subsets:
            self.load_subsets()

        if self.data_format == "stacked":
            self.sources = relation.get("sources") or []
            tables = dict(zip(self.subset_uids, self.subsets))
            for source in self.sources:
                ## a subset added to the relation after this surface was fitted
                if source["subset_uid"] not in tables:
                    tables[source["subset_uid"]] = load_subset(source["subset_uid"])
            data = stack_subsets(self.sources, tables, mapping)
        elif self.data_format == "matrix":
            columns, table = self.subsets[0]
            data = melt_grid(columns, table, mapping)
        elif self.data_format == "vector":
            columns, table = self.subsets[0]
            data = _long_form(table, (mapping["x1_idx"], mapping["x2_idx"], mapping["y_idx"]))
        else:
            raise ValueError("Unknown data format '{}'".format(self.data_format))

        excluded = relation.get("excluded_points")
        self.excluded_points = (np.asarray(excluded, dtype=bool)
                                if excluded is not None and len(excluded) == len(data)
                                else None)
        return self.labels, data


def _long_form(table, colnums):
    """The points of a long-form relation: the mapped columns as floats, one row per table
    row with a value in every one of them (rows with a gap are skipped, as on the platform,
    so the points line up with the relation's exclusion mask)."""
    data = np.column_stack([_as_floats(table[:, col], strict=False) for col in colnums])
    return data[np.all(np.isfinite(data), axis=1)]


## the roles of a stacked relation's data_mapping, in point order [x1, x2, y], and the role
## value that stands for the subset's z instead of a column
STACKED_ROLES = ("x1_idx", "x2_idx", "y_idx")
STACKED_Z = "z"


def stack_subsets(sources, tables, mapping):
    """
    Points of a stacked relation, as SplineCloud joins them.

    Parameters
    ----------
    sources : list of dict
        The relation's ``sources``, in point order: ``{"subset_uid", "z", ...}``. A source
        whose ``z`` is None is skipped.
    tables : dict
        ``{subset uid: (columns, table)}`` (as returned by ``load_subset``).
    mapping : dict
        The relation's ``data_mapping``: each of ``x1_idx``, ``x2_idx``, ``y_idx`` is a
        column index of the subsets (they share their columns) or ``"z"``, the subset's z.

    Returns
    -------
    ndarray
        An (N, 3) array of ``[x1, x2, y]``: subset by subset, one point per row with a
        value in every mapped column, the z role filled with the subset's z.
    """
    if sum(mapping[role] == STACKED_Z for role in STACKED_ROLES) != 1:
        raise ValueError("A stacked mapping maps exactly one role to 'z'")
    colnums = [mapping[role] for role in STACKED_ROLES if mapping[role] != STACKED_Z]

    parts = []
    for source in sources:
        if source.get("z") is None:
            continue
        _columns, table = tables[source["subset_uid"]]
        values = _long_form(table, colnums)
        z = np.full(len(values), float(source["z"]))
        columns = iter(values.T)
        parts.append(np.column_stack([z if mapping[role] == STACKED_Z else next(columns)
                                      for role in STACKED_ROLES]))
    return np.concatenate(parts) if parts else np.empty((0, 3))


def _mask(values, n_points):
    """A relation's excluded_points as a bool array, or None (nothing excluded, or a mask
    that no longer matches the data)."""
    if values is None or len(values) != n_points:
        return None
    return np.asarray(values, dtype=bool)


class SplineHyperSurface(NdBsplExplicit):
    """
    A fitted spline hypersurface y = f(x1, ..., xn), n >= 3, from the SplineCloud API
    (``hsf_`` uids; explicit schema v1 with nx >= 3; needs splinecloud-scipy >= 1.5).

    An NdBsplExplicit with SplineCloud metadata. Inputs are in the order of
    ``input_labels`` (the relation's input columns). Evaluation:

    - ``f(x1, ..., xn)``: pointwise, with NumPy broadcasting (a float for scalars)
    - ``eval(points)`` / ``eval_points(points)``: ``(m, n)`` points -> ``(m,)`` values
    - ``eval_grid(x1_vals, ..., xn_vals)``: the outer-product grid, shape ``(n1, ..., nn)``
    - ``grad(points)``, ``slice({axis: value})`` (an axis is an index or an input label)

    Points outside the fitted domain give NaN unless an extrapolation policy is set
    (``extrapolation=`` when loading, or ``extrapolate=`` per call: "none", "linear",
    "clamp", "polynomial"). Logarithmic axes are handled: pass and receive physical values.

    Fit report: ``method``, ``fit_params``, ``accuracy`` and ``fit_status`` (e.g.
    ``"pending"`` when the relation changed after the fit).
    """

    def __init__(self, data, extrapolation=None):
        payload = data["spline"]
        if payload.get("v") != 1:
            raise ValueError(f"Unsupported explicit spline schema version: {payload.get('v')}")
        if (payload.get("nx") or 0) < 3 or payload.get("ny", 1) != 1:
            raise ValueError("SplineHyperSurface needs an explicit spline with nx >= 3, "
                             "ny = 1")
        NdBsplExplicit.__init__(
            self, payload["t"], payload["k"], payload["c"], scales=payload.get("scales"),
            extrapolation=extrapolation or payload.get("extrapolation", "none"))

        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("description")
        self.form = data.get("form", "explicit")
        self.relation_uid = data.get("data_relation_uid")
        self.relation_name = data.get("relation_name")
        self.subset_uids = data.get("subset_uids", [])

        labels = data.get("labels") or {}
        self.input_labels = list(labels.get("inputs") or [None] * self.ndim)
        self.y_label = labels.get("y")

        self.method = data.get("method")
        self.fit_params = data.get("fit_params")
        self.accuracy = data.get("accuracy")
        self.fit_status = data.get("fit_status")

        self.subsets = []
        self.excluded_points = None     ## set by load_data()

    @property
    def nx(self) -> int:
        return self.ndim

    @property
    def labels(self):
        """``[x1 label, ..., xn label, y label]``."""
        return self.input_labels + [self.y_label]

    def axis_index(self, axis) -> int:
        """An input axis by its index or its label."""
        if isinstance(axis, (int, np.integer)):
            if not 0 <= axis < self.ndim:
                raise ValueError(f"Axis {axis} doesn't exist (inputs 0 .. {self.ndim - 1})")
            return int(axis)
        if axis in self.input_labels:
            if self.input_labels.count(axis) > 1:
                raise ValueError(f"Several inputs are labelled {axis!r}: use the axis index")
            return self.input_labels.index(axis)
        raise ValueError(f"No input labelled {axis!r} (inputs: {self.input_labels})")

    def slice(self, fixed):
        """
        Lower-dimensional spline with some inputs fixed, e.g. ``hs.slice({"Pr": 7.0})`` or
        ``hs.slice({1: 7.0, 2: 30.0})``: exact (the fixed axes are contracted with their
        B-spline basis values). One free input gives an ExplicitUnivariateSpline, more an
        NdBsplExplicit; inputs keep their order.
        """
        return NdBsplExplicit.slice(self, {self.axis_index(a): v for a, v in fixed.items()})

    def load_subsets(self):
        """The subsets the hypersurface was fitted to, as (columns, table) tuples (cached)."""
        if not self.subsets:
            self.subsets = [load_subset(uid) for uid in self.subset_uids]
        return self.subsets

    def load_relation(self):
        """The multivariate relation behind this hypersurface (column mapping, scales, point
        masks), as returned by /api/relations/multivariate/fitted/<uid>/."""
        if not self.relation_uid:
            raise ValueError("This hypersurface has no relation to load")
        url = SPLINECLOUD_API_URL + "/relations/multivariate/fitted/{}/".format(
            self.relation_uid)
        response = requests.get(url, params={"spline": "native"})
        response.raise_for_status()
        return json.loads(response.content)

    def load_data(self):
        """
        Load the data points the hypersurface was fitted to, one ``[x1, ..., xn, y]`` point
        per row (rows with a gap in a mapped column are skipped, as on the platform). The
        relation's exclusion mask lines up with them: ``self.excluded_points`` (None when
        nothing is excluded).

        Returns
        -------
        labels : list of str
            ``[x1, ..., xn, y]`` labels.
        data : ndarray
            An (N, n + 1) array of points.
        """
        relation = self.load_relation()
        mapping = relation["data_mapping"]
        columns, table = self.load_subsets()[0]
        data = _long_form(table, list(mapping["input_idx"]) + [mapping["y_idx"]])
        self.excluded_points = _mask(relation.get("excluded_points"), len(data))
        return self.labels, data


class MultivariateRelation:
    """
    A multivariate fitted relation from the SplineCloud API (``drn_`` uids): y = f(x1, ...,
    xn) over one subset, with its hypersurfaces (``self.hypersurfaces``, SplineHyperSurface
    objects, also by name through ``hypersurface(name)``).

    Attributes: ``uid``, ``name``, ``description``, ``nx``, ``input_labels``, ``y_label``,
    ``scales`` (``{"inputs": [...], "y": ...}``), ``data_mapping`` (``{"input_idx": [...],
    "y_idx": n}``, column indexes of the subset), ``excluded_points`` and
    ``datapoint_weights`` (one entry per data point, or None), ``subset_uid``.
    """

    def __init__(self, data, extrapolation=None):
        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("describ")
        self.nx = data.get("nx")
        labels = data.get("labels") or {}
        self.input_labels = list(labels.get("inputs") or [])
        self.y_label = labels.get("y")
        self.scales = data.get("scales") or {}
        self.data_mapping = data.get("data_mapping") or {}
        self.excluded_points = data.get("excluded_points")
        self.datapoint_weights = data.get("datapoint_weights")
        self.subset_uid = (data.get("subset") or {}).get("uid")
        self.hypersurfaces = [SplineHyperSurface(item, extrapolation=extrapolation)
                              for item in data.get("hypersurfaces") or []]
        self._subset = None

    @property
    def labels(self):
        """``[x1 label, ..., xn label, y label]``."""
        return self.input_labels + [self.y_label]

    def hypersurface(self, name):
        """The relation's hypersurface with this name (the first one if several share it)."""
        for item in self.hypersurfaces:
            if item.name == name:
                return item
        raise KeyError(f"No hypersurface named {name!r} "
                       f"(available: {[item.name for item in self.hypersurfaces]})")

    def load_data(self):
        """
        Load the relation's data points, one ``[x1, ..., xn, y]`` point per row (rows with a
        gap in a mapped column are skipped, as on the platform), and the exclusion mask
        lined up with them.

        Returns
        -------
        labels : list of str
        data : ndarray, shape (N, n + 1)
        excluded : ndarray of bool, shape (N,), or None
        """
        if self._subset is None:
            self._subset = load_subset(self.subset_uid)
        columns, table = self._subset
        mapping = self.data_mapping
        data = _long_form(table, list(mapping["input_idx"]) + [mapping["y_idx"]])
        return self.labels, data, _mask(self.excluded_points, len(data))


class SplineVolume(ParametricSplineVolume):
    """
    A lofted spline volume y = f(x1, x2, x3) from the SplineCloud API (``hsf_`` uids with
    ``form == "parametric"``; the parametric volume schema v1, C.7; needs
    splinecloud-scipy >= 1.5): a ParametricSplineVolume with SplineCloud metadata.

    It is lofted through 2-D models (``stations``: ``[{"x3", "surface_uid"}]`` in x3
    order, those of its last build) and reproduces each of them at its x3. Inputs are
    ``[x1, x2, x3]`` (``input_labels``). Evaluation: ``eval(x1, x2, x3)``,
    ``eval_point(...)``, ``slice_at(x3)``; physical values in and out.

    ``fit_status`` is ``"pending"`` when a station changed after the volume was built.
    """

    def __init__(self, data):
        ParametricSplineVolume.__init__(self, data["spline"])
        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("description")
        self.form = data.get("form", "parametric")
        self.relation_uid = data.get("loft_relation_uid")
        self.relation_name = data.get("relation_name")
        labels = data.get("labels") or {}
        self.input_labels = list(labels.get("inputs") or [None] * 3)
        self.y_label = labels.get("y")
        self.stations = list(data.get("stations") or [])
        self.fit_params = data.get("fit_params")
        self.fit_status = data.get("fit_status")
        self._station_surfaces = None

    @property
    def nx(self) -> int:
        return 3

    @property
    def labels(self):
        """``[x1 label, x2 label, x3 label, y label]``."""
        return self.input_labels + [self.y_label]

    @property
    def x3_stations(self):
        """The x3 values of the stations (physical, ascending)."""
        return [station["x3"] for station in self.stations]

    def load_station_surfaces(self):
        """The station surfaces (SplineSurface or ExplicitSplineSurface, in x3 order;
        cached). Their ``load_data()`` gives each station's data points."""
        if self._station_surfaces is None:
            self._station_surfaces = [load_spline_surface(station["surface_uid"])
                                      for station in self.stations]
        return self._station_surfaces

    def load_relation(self):
        """The lofted 3D relation behind this volume (LoftedRelation3D)."""
        if not self.relation_uid:
            raise ValueError("This volume has no relation to load")
        return load_lofted_relation_3d(self.relation_uid)


class LoftedRelation3D:
    """
    A lofted 3D relation from the SplineCloud API (``lr3_`` uids): y = f(x1, x2, x3)
    through 2-D models (lofted or fitted surfaces) at x3 values, with its volume.

    Attributes: ``uid``, ``name``, ``description``, ``input_labels`` (``[x1, x2, x3]``),
    ``y_label``, ``scale_x3``, ``w_degree`` (degree along x3), ``stations`` (``[{"x3_value",
    "surface_uid", "surface_name", "surface_type", "relation_uid"}]`` in x3 order; a
    deleted surface gives None), ``volumes`` (SplineVolume objects; one per relation) and
    ``volume`` (the first, or None).
    """

    def __init__(self, data):
        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("describ")
        labels = data.get("labels") or {}
        self.input_labels = list(labels.get("inputs") or [None] * 3)
        self.y_label = labels.get("y")
        self.scale_x3 = data.get("scale_x3")
        self.w_degree = data.get("w_degree")
        self.stations = list(data.get("stations") or [])
        self.volumes = [SplineVolume(item) for item in data.get("hypersurfaces") or []]

    @property
    def volume(self):
        return self.volumes[0] if self.volumes else None

    @property
    def labels(self):
        """``[x1 label, x2 label, x3 label, y label]``."""
        return self.input_labels + [self.y_label]

    def load_station_surfaces(self):
        """The current stations' surfaces in x3 order (None for a deleted surface)."""
        return [load_spline_surface(station["surface_uid"]) if station.get("surface_uid")
                else None for station in self.stations]


## "<label>=<value>" or "<label>: <value>" grid headers / cells, as SplineCloud reads them
_LABELED_VALUE = re.compile(r"^\s*(?P<label>[^=:]+?)\s*[=:]\s*(?P<value>\S.*?)\s*$")


def _axis_floats(values, what):
    """A grid axis (x1 cells, x2 headers or row) as floats: numbers, or "<label>=<value>" /
    "<label>: <value>" entries that all carry one label (case-insensitive; a label holds a
    letter). Empty entries become NaN; anything else raises ValueError."""
    result = np.full(len(values), np.nan)
    labels = set()
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)) or \
                (isinstance(value, str) and not value.strip()):
            continue
        try:
            result[i] = float(value)
            continue
        except (TypeError, ValueError):
            pass
        match = _LABELED_VALUE.match(str(value))
        if match and re.search(r"[^\W\d_]", match.group("label")):
            try:
                result[i] = float(match.group("value"))
                labels.add(match.group("label").casefold())
                continue
            except ValueError:
                pass
        raise ValueError("{} isn't a number: {!r}".format(what, value))
    if len(labels) > 1:
        raise ValueError("{}s carry different labels: {}".format(what, sorted(labels)))
    return result


def _as_floats(values, strict=True, what="A value"):
    """values as a float array; empty cells (None, NaN, "") become NaN. Any other value that
    isn't a number raises ValueError (strict) or becomes NaN."""
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        try:
            result[i] = float(value)
        except (TypeError, ValueError):
            if strict:
                raise ValueError("{} isn't a number: {!r}".format(what, value))
    return result


def melt_grid(columns, table, mapping):
    """
    Points of a grid (matrix) table, melted to long form as SplineCloud does.

    Parameters
    ----------
    columns : list of str
        Column headers (as returned by ``load_subset``).
    table : ndarray
        The table, rows by columns.
    mapping : dict
        The relation's ``data_mapping``: ``x1_col`` (column of x1 values), ``x2_source``
        (``"header"``: x2 = the y columns' headers; ``"row"``: x2 = the values of row
        ``x2_row``, which is then not a data row) and ``y_cols`` (None = every column but
        x1's).

    Returns
    -------
    ndarray
        An (N, 3) array of ``[x1, x2, y]``: one point per cell with a value whose row has an
        x1 value, row by row, then column by column.
    """
    table = np.asarray(table, dtype=object)
    x1_col, x2_source = mapping["x1_col"], mapping.get("x2_source", "header")
    x2_row = mapping.get("x2_row")
    y_cols = mapping.get("y_cols")
    if y_cols is None:
        y_cols = [c for c in range(len(columns)) if c != x1_col]

    if x2_source == "header":
        x2 = _axis_floats([columns[c] for c in y_cols], what="An x2 header")
    elif x2_source == "row":
        x2 = _axis_floats([table[x2_row, c] for c in y_cols], what="An x2 value")
    else:
        raise ValueError("Unknown x2 source '{}'".format(x2_source))
    if not np.all(np.isfinite(x2)):
        raise ValueError("Every y column needs an x2 value")

    rows = np.arange(table.shape[0])
    if x2_source == "row":
        rows = rows[rows != x2_row]
    x1 = _axis_floats(table[rows, x1_col], what="An x1 value")
    Y = np.column_stack([_as_floats(table[rows, c], what="A cell") for c in y_cols])

    keep = np.isfinite(Y) & np.isfinite(x1)[:, None]
    r_idx, c_idx = np.nonzero(keep)                     ## C order, as on the platform
    return np.column_stack([x1[r_idx], x2[c_idx], Y[r_idx, c_idx]])


# =============================================================================
# HELPERS
# =============================================================================

def fill_array(table, subset, columns, num_rows):
    for ci, col_name in enumerate(columns):
        for ri in range(num_rows):
            table[ri, ci] = subset[col_name][ri]

# =============================================================================
# MAIN CLIENT FUNCTIONS
# =============================================================================

def load_subset(subset_id_or_url):
    url_split = subset_id_or_url.split("/")
    if len(url_split) > 1:
        url = subset_id_or_url
    else:
        subset_id = url_split[-1]
        if "sbt_" not in subset_id or len(subset_id) != 16:
            raise ValueError("Wrong subset id was specified")
        url = SPLINECLOUD_API_URL + "/subsets/{}/".format(subset_id)

    response = requests.get(url)
    subset_raw = json.loads(response.content)['table']
    columns = list(subset_raw.keys())

    # Treat all column data as ordered sequences, ignoring any index keys
    subset = {
        col: list(vals.values()) if isinstance(vals, dict) else vals
        for col, vals in subset_raw.items()
    }

    num_rows = len(list(subset.values())[0])
    table = np.zeros((num_rows, len(columns)))
    try:
        fill_array(table, subset, columns, num_rows)
    except ValueError:
        table = np.empty((num_rows, len(columns)), dtype=object)
        fill_array(table, subset, columns, num_rows)

    return columns, table
        
        
def load_spline(curve_id_or_url):
    url_split = curve_id_or_url.split("/")
    if len(url_split) > 1:
        url = curve_id_or_url
    else:
        curve_id = url_split[-1]
        if "spl_" not in curve_id or len(curve_id) != 16:
            raise ValueError("Wrong curve id was specified")
        url = SPLINECLOUD_API_URL+"/curves/{}/".format(curve_id)

    ## spline=native: explicit curves come in the explicit schema (v1); parametric ones are
    ## unchanged. Servers that don't know the parameter return the legacy parametric form.
    response = requests.get(url, params={"spline": "native"})
    curve = json.loads(response.content)

    form = curve.get('form', 'parametric')
    curve_params = curve['spline']

    if form not in ('explicit', 'parametric'):
        raise ValueError(f"Unsupported spline form '{form}'. "
                         "Upgrade splinecloud-scipy to load this curve.")

    if form == 'explicit' and 'v' in curve_params:
        spline = ExplicitUnivariateSpline.from_json(curve_params)
    else:
        ## parametric curves, and explicit curves from a server without spline=native
        ## support (their exact Greville embedding evaluates to the same y(x))
        t = np.array(curve_params['t'])
        c = np.array(curve_params['c'])
        tcck = t, c[:, 0], c[:, 1], curve_params['k']

        log_x = curve['scale_x'] == "Logarithmic"
        log_y = curve['scale_y'] == "Logarithmic"

        spline = ParametricUnivariateSpline(tcck, log_x=log_x, log_y=log_y)

    spline.form = form
    spline.load_data = lambda: load_subset(curve['subset_uid'])

    return spline

def load_spline_surface(surface_id_or_url, extrapolation=None):
    """
    Fetch a spline surface from the SplineCloud API.

    Accepts either a full URL or a bare surface id (prefix ``srf_``).

    Returns
    -------
    SplineSurface
        for lofted surfaces (``form == "parametric"``), unchanged;
    ExplicitSplineSurface
        for fitted surfaces (``form == "explicit"``). ``extrapolation`` overrides the
        server's suggested default policy for these.

    Raises ValueError for a spline form this version doesn't know.
    """
    url_split = surface_id_or_url.split("/")
    if len(url_split) > 1:
        url = surface_id_or_url
    else:
        surface_id = url_split[-1]
        if "srf_" not in surface_id or len(surface_id) != 16:
            raise ValueError("Wrong surface id was specified")
        url = SPLINECLOUD_API_URL + "/surfaces/{}/".format(surface_id)

    response = requests.get(url, params={"spline": "native"})
    response.raise_for_status()
    data = json.loads(response.content)

    form = data.get("form", "parametric")
    if form not in SPLINE_FORMS:
        raise ValueError(f"Unsupported spline form '{form}'. "
                         "Upgrade splinecloud-scipy to load this surface.")
    if form == "explicit":
        return ExplicitSplineSurface(data, extrapolation=extrapolation)
    return SplineSurface(data)


def _api_url(uid_or_url, prefix, route, what):
    """A full URL as given, or the API URL of a bare uid with the expected prefix."""
    if len(uid_or_url.split("/")) > 1:
        return uid_or_url
    if not uid_or_url.startswith(prefix) or len(uid_or_url) != 16:
        raise ValueError(f"Wrong {what} id was specified (expected '{prefix}' and 16 "
                         "characters)")
    return SPLINECLOUD_API_URL + "/{}/{}/".format(route, uid_or_url)


def load_spline_hypersurface(hypersurface_id_or_url, extrapolation=None):
    """
    Fetch a spline hypersurface y = f(x1, ..., xn), n >= 3, from the SplineCloud API.

    Accepts a full URL or a bare hypersurface id (prefix ``hsf_``). ``extrapolation``
    overrides the server's suggested default policy ("none", "linear", "clamp",
    "polynomial").

    Returns
    -------
    SplineHyperSurface
        for fitted hypersurfaces (``form == "explicit"``, C.1);
    SplineVolume
        for lofted volumes (``form == "parametric"``, C.7; ``extrapolation`` doesn't apply,
        pass ``extrapolate=`` when evaluating).

    Raises ValueError for a spline form this version doesn't know.
    """
    url = _api_url(hypersurface_id_or_url, "hsf_", "hypersurfaces", "hypersurface")
    response = requests.get(url, params={"spline": "native"})
    response.raise_for_status()
    data = json.loads(response.content)

    form = data.get("form", "explicit")
    if form == "explicit":
        return SplineHyperSurface(data, extrapolation=extrapolation)
    if form == "parametric":
        return SplineVolume(data)
    raise ValueError(f"Unsupported hypersurface form '{form}'. "
                     "Upgrade splinecloud-scipy to load this hypersurface.")


def load_multivariate_relation(relation_id_or_url, extrapolation=None):
    """
    Fetch a multivariate fitted relation (prefix ``drn_``) with its hypersurfaces.

    Returns
    -------
    MultivariateRelation
    """
    url = _api_url(relation_id_or_url, "drn_", "relations/multivariate/fitted",
                   "multivariate relation")
    response = requests.get(url, params={"spline": "native"})
    response.raise_for_status()
    return MultivariateRelation(json.loads(response.content), extrapolation=extrapolation)


def load_lofted_relation_3d(relation_id_or_url):
    """
    Fetch a lofted 3D relation (prefix ``lr3_``) with its stations and volume.

    Returns
    -------
    LoftedRelation3D
    """
    url = _api_url(relation_id_or_url, "lr3_", "relations/multivariate/lofted",
                   "lofted 3D relation")
    response = requests.get(url, params={"spline": "native"})
    response.raise_for_status()
    return LoftedRelation3D(json.loads(response.content))


def _uid_from(uid_or_url):
    """The uid at the end of a SplineCloud API or web URL (or the uid itself)."""
    parts = [part for part in uid_or_url.split("?")[0].split("/") if part]
    if not parts:
        raise ValueError("Empty uid or URL")
    return parts[-1]


def load_model(uid_or_url):
    """
    Load a SplineCloud model by its uid or URL, dispatching on the uid prefix: ``spl_`` ->
    curve (load_spline), ``srf_`` -> surface (load_spline_surface), ``hsf_`` ->
    hypersurface or lofted volume (load_spline_hypersurface), ``drn_`` -> multivariate
    relation with its hypersurfaces (load_multivariate_relation), ``lr3_`` -> lofted 3D
    relation with its volume (load_lofted_relation_3d). Data is loaded with load_subset.
    """
    uid = _uid_from(uid_or_url)
    prefix = uid[:4]
    if prefix == "spl_":
        return load_spline(uid_or_url)
    if prefix == "srf_":
        return load_spline_surface(uid_or_url)
    if prefix == "hsf_":
        return load_spline_hypersurface(uid_or_url)
    if prefix == "drn_":
        return load_multivariate_relation(uid_or_url)
    if prefix == "lr3_":
        return load_lofted_relation_3d(uid_or_url)
    if prefix == "sbt_":
        raise ValueError(f"'{uid}' is a subset, not a model: use load_subset")
    raise ValueError(f"Can't load '{uid}': unknown uid prefix '{prefix}' "
                     "(expected spl_, srf_, hsf_, drn_ or lr3_)")
