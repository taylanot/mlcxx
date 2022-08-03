from dolfin import *
import numpy as np
parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
import copy as cp
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


# elastic parameteors
Em = 50e3
num = 0.2
Er = 210e3
nur = 0.3
nphases = len(material_parameters)
sig0 = Constant(250.)  # yield strength

lmbdam = Em*num/(1+num)/(1-2*num)
lmbdar = Er*nur/(1+nur)/(1-2*nur)
mum = Em/2./(1+num)
mur = Er/2./(1+nur)

Etr = Er/100.  # tangent modulus
Hr = Er*Etr/(Er-Etr)  # hardening modulus

Etm = Em/100.  # tangent modulus
Hm = Em*Etm/(Em-Etm)  # hardening modulus


material_parameters = [(Em, num, lmbdam, mum, Etm, Hm), (Er, nur, lmbdar, mur, Etr, Hr)]

a = 1.         # unit cell width
b = sqrt(3.)/2. # unit cell height
c = 0.5        # horizontal offset of top boundary
R = 0.2        # inclusion radius
vol = a*b      # unit cell volume
# we define the unit cell vertices coordinates for later use
vertices = np.array([[0, 0.],
                     [a, 0.],
                     [a+c, b],
                     [c, b]])
fname = "hexag_incl"
mesh = Mesh(fname + ".xml")
subdomains = MeshFunction("size_t", mesh, fname + "_physical_region.xml")
facets = MeshFunction("size_t", mesh, fname + "_facet_region.xml")

class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :]-self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :]-self.vv[1, :] - self.a2) <= self.tol
        
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the 
        # bottom-right or top-left vertices
        return bool((near(x[0], self.vv[0,0] + x[1]*self.a2[0]/self.vv[3,1], self.tol) or 
                     near(x[1], self.vv[0,1] + x[0]*self.a1[1]/self.vv[1,0], self.tol)) and 
                     (not ((near(x[0], self.vv[1,0], self.tol) and near(x[1], self.vv[1,1], self.tol)) or 
                     (near(x[0], self.vv[3,0], self.tol) and near(x[1], self.vv[3,1], self.tol)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.vv[2,0], self.tol) and near(x[1], self.vv[2,1], self.tol): # if on top-right corner
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
        elif near(x[0], self.vv[1,0] + x[1]*self.a2[0]/self.vv[2,1], self.tol): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:   # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]


deg_u = 1
deg_stress = 1
V = VectorFunctionSpace(mesh, "CG", deg_u,constrained_domain=PeriodicDomain(mesh,domainBB,periodicAxes=[0,1], tolerance=1e-10))

We = VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W = FunctionSpace(mesh, We)
W0e = FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W0 = FunctionSpace(mesh, W0e)

sig = Function(W)
sig_old = Function(W)
n_elas = Function(W)
beta = Function(W0)

p = Function(W0, name="Cumulative plastic strain")
u = Function(V, name="Total displacement")
du = Function(V, name="Iteration correction")
Du = Function(V, name="Current increment")
v = TrialFunction(V)
u_ = TestFunction(V)


def eps(v):
    e = sym(grad(v))
    return as_tensor([[e[0, 0], e[0, 1], 0],
                      [e[0, 1], e[1, 1], 0],
                      [0, 0, 0]])

def sigma(eps_el, lmbda):
    return lmbda*tr(eps_el)*Identity(3) + 2*mu*eps_el

def as_3D_tensor(X):
    return as_tensor([[X[0], X[3], 0],
                      [X[3], X[1], 0],
                      [0, 0, X[2]]])


ppos = lambda x: (x+abs(x))/2.

def return_map(deps, old_sig, old_p):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + sigma(deps)
    s = dev(sig_elas)
    sig_eq = sqrt(3/2.*inner(s, s))
    f_elas = sig_eq - sig0 - H*old_p
    dp = ppos(f_elas)/(3*mu+H)
    n_elas = s/sig_eq*ppos(f_elas)/f_elas
    beta = 3*mu*dp/sig_eq
    new_sig = sig_elas-beta*s
    return as_vector([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]]), \
           as_vector([n_elas[0, 0], n_elas[1, 1], n_elas[2, 2], n_elas[0, 1]]), \
           beta, dp

def sigma_tang(e):
    N_elas = as_3D_tensor(n_elas)
    return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*inner(N_elas, e)*N_elas-2*mu*beta*dev(e)

metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
dxm = dx(metadata=metadata)

Eps = Constant(((0.1,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)))

a_Newton = inner(eps(v), sigma_tang(eps(u_)))*dxm
res = -inner(eps(u_), as_3D_tensor(sig))*dxm #+ F_ext(u_)

def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dxm
    b_proj = inner(v, v_)*dxm
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return


file_results = XDMFFile("load.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
P0 = FunctionSpace(mesh, "DG", 0)
p_avg = Function(P0, name="Plastic strain")



Nitermax, tol = 200, 1e-8  # parameters of the Newton-Raphson procedure
Nincr = 1
load_steps = np.linspace(0, 1.1, Nincr+1)[1:]**0.5
results = np.zeros((Nincr+1, 2))


for (i, t) in enumerate(load_steps):
    load = t 
    loading.t = load
    #a = assemble_sys(a_Newton)
    A, Res = assemble_system(a_Newton, res, [])
    nRes0 = Res.norm("l2")
    nRes = nRes0
    Du.interpolate(Constant((0, 0)))
    print("Increment:", str(i+1))
    niter = 0
    while nRes > tol and niter < Nitermax:
        solve(A, du.vector(), Res, "mumps")
        Du.assign(Du+du)
        deps = eps(Du)
        sig_, n_elas_, beta_, dp_ = return_map(deps, sig_old, p)
        local_project(sig_, W, sig)
        local_project(n_elas_, W, n_elas)
        local_project(beta_, W0, beta)
        A, Res = assemble_system(a_Newton, res, [])
        nRes = Res.norm("l2")
        print("    Residual:", nRes)
        niter += 1

