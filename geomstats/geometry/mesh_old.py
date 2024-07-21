import os
import pickle

import math
import numpy as np
import igl
import scipy
from jax import numpy as jnp

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class _Mesh(Manifold):
    """Private class for general closed manifold described by triangular mesh.
    ----------
    dim : int
        Dimension of the manifold.
    """

    def __init__(self, dim, v, f, trunc=200, cached=None):
        super(_Mesh, self).__init__(
            dim=dim,
            metric=None,
            default_point_type="vector",
            default_coords_type="intrinsic",
        )
        assert dim == 3
        self.v = v
        self.f = f
        self.per_face_normals = igl.per_face_normals(v, f, np.array([]))

        # Pre-compute the eigenvalues and eigenfunctions
        if self.v.shape[0] > 10000 and cached is not None:
            if os.path.isfile(cached):
                with open(cached, 'rb') as ff:
                    eig_val, eig_vec = pickle.load(ff)
                self.eig_val, self.eig_vec = eig_val[:trunc], eig_vec[:,:trunc]
            else:
                self.eig_val, self.eig_vec = self.get_eig(trunc=trunc)
                with open(cached, 'wb') as ff:
                    pickle.dump(obj=(self.eig_val, self.eig_vec), file=ff, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.eig_val, self.eig_vec = self.get_eig(trunc=trunc)

        self.eig_fn_f = igl.average_onto_faces(self.f, self.eig_vec)
        self.grad_eig_fn_f = (igl.grad(v, f) @ self.eig_vec).reshape(-1, 3, trunc, order="F")

    def point_mesh_squared_distance(self, point):
        import pdb; pdb.set_trace()
        # Issue on converting Tracer to np array?
        # point = np.asarray(point)
        return igl.point_mesh_squared_distance(point, self.v, self.f)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the manifold.
        """
        sqrd = self.point_mesh_squared_distance(point)[0]
        # return gs.isclose(np.double(sqrd), 0., atol)
        return gs.isclose(sqrd, 0., atol)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        fidx = self.point_mesh_squared_distance(base_point)[1]
        normals = self.per_face_normals[fidx]
        # if isinstance(vector, torch.Tensor):
        #     normals = torch.from_numpy(normals).float().to(vector.device)
        normals = jnp.array(normals)
        return gs.isclose(self.metric.inner_product(vector, normals), 0., atol)

    # NOTE: project vector to tangent space: torch operations for jacrev computations
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        # TODO: how to compute normal from base_ponit with continuous grad?
        fidx = self.point_mesh_squared_distance(base_point)[1]
        normals = self.per_face_normals[fidx]
        # normals = torch.from_numpy(normals).float().to(vector.device)
        normals = jnp.array(normals)
        coeff = self.metric.inner_product(vector, normals)
        tangent_vec = vector - gs.einsum("...,...j->...j", coeff, normals)
        return tangent_vec
        # return vector

    # NOTE: project vector to tangent space using libigl
    def to_tangent_igl(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        fidx = self.point_mesh_squared_distance(base_point)[1]
        normals = self.per_face_normals[fidx]
        # normals = torch.from_numpy(normals).float().to(vector.device)
        normals = jnp.array(normals)
        coeff = self.metric.inner_product(vector, normals)
        tangent_vec = vector - gs.einsum("...,...j->...j", coeff, normals)
        return tangent_vec

    def projection(self, point):
        """Project a point on the mesh.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim]
            Point projected on the mesh.
        """
        projected_point = self.point_mesh_squared_distance(point)[-1]
        # if isinstance(point, torch.Tensor):
        #     projected_point = torch.from_numpy(projected_point).float().to(point.device)
        return jnp.array(projected_point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : unused

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, state, n_samples=1):
        """Sample in the mesh from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim]
            Points sampled on the mesh.
        """
        # Given as barycentric coordinates and corresponding faces: (nx3, nx1)
        bv, fidx = igl.random_points_on_mesh(n_samples, self.v, self.f)
        bv = bv.reshape(-1, 3)
        samples = (bv[...,None] * self.v[self.f[fidx]]).sum(1)
        # return torch.from_numpy(samples).float().to(device)
        return jnp.array(samples)

    def random_normal_tangent(self, state, base_point, n_samples=1):
        """Sample in the tangent space from the standard normal distribution.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        ambiant_noise = gs.random.normal(state=state, size=(n_samples, self.dim))
        # ambiant_noise = torch.randn((n_samples, self.dim), device=base_point.device)
        # return self.to_tangent(vector=ambiant_noise, base_point=base_point)
        return self.to_tangent_igl(vector=ambiant_noise, base_point=base_point)

    def get_eig(self, trunc):
        l = -igl.cotmatrix(self.v, self.f)
        m = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        # generalized eigenvalue problem
        eig_val, eig_vec = scipy.sparse.linalg.eigsh(l, trunc+1, m, sigma=0, which="LM")
        # Only the non-zero eigenvalues and its eigenvectors
        eig_val, eig_vec = eig_val[1:trunc+1], eig_vec[:,1:trunc+1]

        # Should we normalize the eigenfunction?
        eig_vec = eig_vec / np.sqrt((eig_vec**2).sum(0))
        return eig_val, eig_vec

    def eig_fn(self, fidx):
        return self.eig_fn_f[fidx]

    def grad_eig_fn(self, fidx):
        return self.grad_eig_fn_f[fidx]

    @property
    def log_volume(self):
        return igl.doublearea(self.v, self.f).sum() / 2


    def diff_dist(self, point_a, point_b, wtype='inv'):
        def weighting_fn(z, w):
            if w=='inv':
                return 1./np.abs(z)
            else:
                raise ValueError(f'{w} not valid.')
        d = weighting_fn(self.eig_val, wtype)
        aidx = self.point_mesh_squared_distance(point_a)[1]
        bidx = self.point_mesh_squared_distance(point_b)[1]
        d = d[None,:] * (self.eig_fn(aidx) - self.eig_fn(bidx))**2
        return d.sum(-1) 

    def fidx(self, point):
        return self.point_mesh_squared_distance(point)[1]


class MeshMetric(RiemannianMetric):
    """Class for the Mesh Metric.

    Parameters
    ----------
    dim : int
        Dimension of the Mesh.
    """

    def __init__(self, dim, v, f, trunc, cached):
        super(MeshMetric, self).__init__(dim=dim, signature=(dim, 0))
        self.embedding_metric = EuclideanMetric(dim)
        self._space = _Mesh(dim=dim, v=v, f=f, trunc=trunc, cached=cached)

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        return gs.eye(self.dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim], optional
            Point on the mesh.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        ) #.float()

        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector on the tangent space of the mesh at base point.
        base_point : array-like, shape=[..., dim], optional
            Point on the mesh.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim]
            Point on the mesh.

        Returns
        -------
        exp : array-like, shape=[..., dim]
            Point on the mesh equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        # First project the vector to the tangent space
        fidx = self._space.point_mesh_squared_distance(base_point)[1]
        normals = self._space.per_face_normals[fidx]
        # normals = torch.from_numpy(normals).float().to(tangent_vec.device)
        normals = jnp.array(normals)
        coeff = self.inner_product(tangent_vec, normals)
        tangent_vec = tangent_vec - gs.einsum("...,...j->...j", coeff, normals)
        # NOTE: approx. exp using projection
        return self._space.projection(base_point + tangent_vec)

    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the mesh.
        base_point : array-like, shape=[..., dim]
            Point on the mesh.

        Returns
        -------
        log : array-like, shape=[..., dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        raise NotImplementedError(f'Mesh: log not implemented.')

    def grad(self, func):
        return self.embedding_metric.grad(func)



class Mesh(_Mesh):
    def __init__(self, dim, v, f, trunc=200, cached=None):
        super(Mesh, self).__init__(dim=dim, v=v, f=f, trunc=trunc, cached=cached)
        self.metric = MeshMetric(dim, v, f, trunc, cached)