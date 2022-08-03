from dolfin import *    # Library for FEM Analysis
import dolfin
import numpy as np      # Good old numpy  
import matplotlib.pyplot as plt # plot stuff
from petsc4py import PETSc
import os               # Some file management

class Domain():
    """
    * This class reads the mesh file, optionally convert it to xml 
    format (cannot be paralellized) to xml format.
    * Important geometric properties for the periodic boundary 
    conditions are also obtained from the mesh.
    * Just for 2D for now. Have 3D implemented but for the sake of simplicity
    it is way easier to read this.
    """

    def __init__(self, filename):

        # Get the filename 
        self.fname = filename 
        # Root name of the file
        self.name, self.ext =  os.path.splitext(filename) 
        # Read the files with the same root
        self.__read_xml()
        # Get the vertices and the volume of the domain
        self.verticies, self.vol = self.__get_vertices()
        # Get the vertices and the individual volume of the elements
        self.__get_volume()
        
    def __read_xml(self):

        # Read the mesh
        self.mesh = Mesh(self.name+".xml")
        # Dimension of the domain
        self.dim = self.mesh.geometry().dim()
        # Number of elements in the domain
        self.ele_num = self.mesh.num_cells() # Number of elements in the domain

    def __get_vertices(self):
        """
             (x_min,y_max) #-----# (x_max,y_max)
                           |     |
                           |     |
                           |     |
             (x_min,y_min) #-----# (x_max,y_min)

        * This should be self explanatory
        """

        if self.dim == 2:

            x_min = np.min(self.mesh.coordinates()[:,0]) 
            x_max = np.max(self.mesh.coordinates()[:,0]) 
                
            y_min = np.min(self.mesh.coordinates()[:,1]) 
            y_max = np.max(self.mesh.coordinates()[:,1]) 
            
            vert =  np.array([[x_min,y_min], [x_max,y_min], \
                             [x_max,y_max], [x_min,y_max]])

            vol = x_max * y_max

        elif self.dim == 3:
            raise ("Not implimented yet!")

        return vert, vol

    def __get_volume(self):
        """
            Method: Get volume/area of all the elements in a numpy array
        """
        
        self.ele_vol = np.zeros(self.ele_num)
        for i in range(self.ele_num):
            cell = Cell(self.mesh, i)
            self.ele_vol[i] = cell.volume()

class PeriodicBoundary2D(SubDomain):
    """
    Periodic Boundary Conditions implimentation. Given the
    vertices of the RVE it maps left boundary to the right
    boundary and bottom boundary to the top boundary.

                          (top)
                        #-------#
                        ^       |
                (left)  |  RVE  |  (right)
                 [v1]   |       |
                        *------>#
                         (bottom)
                           [v2]

     NOTE: Your RVE should have 0.0 coordinates at the * vertex!
     TO DO: Generalize!
     * A better version is implemented in my repo but this is simpler for
     understanding. You do not need to understand every detail for this. Just
     what it is used for is enough I believe.
    """ 

    def __init__(self, domain, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.v = domain.verticies
        self.v1 = self.v[1,:]-self.v[0,:] # First vector generating periodicity
        self.v2 = self.v[3,:]-self.v[0,:] # Second vector generating periodicity
        # Check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.v[2, :]-self.v[3, :] - self.v1) <= self.tol
        assert np.linalg.norm(self.v[2, :]-self.v[1, :] - self.v2) <= self.tol
        
    def inside(self, x, on_boundary):
        # Return True if on left or bottom boundary AND NOT on one of the 
        # Bottom-right or top-left vertices
        return bool(
            (near(x[0], self.v[0,0] + x[1]*self.v2[0]/self.v[3,1], self.tol) or 
        near(x[1], self.v[0,1] + x[0]*self.v1[1]/self.v[1,0], self.tol)) and 
            (not ((near(x[0], self.v[1,0], self.tol) and \
                    near(x[1], self.v[1,1], self.tol)) or \
                     (near(x[0], self.v[3,0], self.tol) and \
                     near(x[1], self.v[3,1], self.tol))))\
            and on_boundary)

    def map(self, x, y):
        # If on top-right corner
        if near(x[0], self.v[2,0], self.tol) and \
                near(x[1], self.v[2,1], self.tol): 
            y[0] = x[0] - (self.v1[0]+self.v2[0])
            y[1] = x[1] - (self.v1[1]+self.v2[1])
        # If on right boundary
        elif near(x[0], self.v[1,0] + x[1]*self.v2[0]/self.v[2,1], self.tol): 
            y[0] = x[0] - self.v1[0]
            y[1] = x[1] - self.v1[1]
        # Should be on top boundary
        else:   
            y[0] = x[0] - self.v2[0]
            y[1] = x[1] - self.v2[1]

def spit_strain_stress(F11=0.0, F12=0., F21=0., F22=0., t=0, comp='11'): 
    """

    results: is the xdmf file for storing the output field
    F11, F12, F21, F22: Are the Deformation Gradient Components for the
    macroscopic deformation gradient
    t: is the step enumerator
    comp: is the component you want the observe from the deformation gradient
    and the second Piola Kirchhoff stress tensor.

    """
    
    # Initialize your domain
    domain = Domain("domain.xml") 
    # Initialize your FunctionSpaces you would like to solve! It depends on your 
    # problem.
    Ve = VectorElement("CG", domain.mesh.ufl_cell(), 1)
    Re = VectorElement("R", domain.mesh.ufl_cell(), 0)
    W = FunctionSpace(domain.mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary2D(domain,tolerance=1e-10))
    V = FunctionSpace(domain.mesh, Ve)

    # Get your Test and Trial Functions
    v_,lamb_= TestFunctions(W)
    dv, dlamb = TrialFunctions(W)
    w = Function(W)
    u,c = split(w)

    # Establish the Kinematical Relations 
    F_macro_check = np.array([[F11, F12], [F12,F22]]) + np.eye(domain.dim)
    d = len(u)
    I = Identity(d)             # Identity tensor
    F_macro = Constant(((F11, F12), (F12,F22))) + I
    F = variable(grad(u) + F_macro)               # Deformation gradient
    C = F.T*F 
    B = F*F.T 
    #C = variable(C)

    # Invariants of deformation tensors

    J  = det(F)     # Jacobian
    # First-Second-Third invariant of the Cauchy-Green Strain Tensor in order...
    Ic = tr(C)       
    IIc  = 0.5 * (tr(C)**2 - tr(C*C))
    IIIc  = J**2    
    Ic_inv  = tr(inv(C))
    Macro_Green_Strain = 0.5 * (dot(F_macro,F_macro.T)-I)
    Green = 0.5 * (dot(F,F.T)-I)

    # Elastic Moduus and the Poisson's ratio
    E, nu = 200., 0.2
    mu, lmbda, beta = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(nu/(1-2*nu))
    # Strain Energy Density Function for the simple Neo-Hookean Material

    ########## Neo
    psi = mu/2. * (Ic - 2 - 2*ln(J)) + lmbda/2.*(J-1)**2
    #psi = mu/2. * (Ic - 3) - mu * ln(J) + lmbda/2. * (ln(J))**2
    
    ########## BlatzKo [On the Use of the Blatz-Ko consittutive model in nonlinear finite element analysis]
    #f = 1. # -> Valid for most of the rubbers!
    #psi = mu*0.5*(f*(Ic - 3 +2/beta*(J**(-beta)-1))+(1-f)*((IIc/J-3)+2/beta*(J**(beta)-1)))

    ########## SVenant
    #Green = variable(Green)
    #psi = 0.5*lmbda*tr(Green)**2+mu*tr(Green*Green)
    #S = diff(psi, Green)
    #S = lmbda*tr(Green)*I+2*mu*Green
    #sig = 1/J *F*S*F.T
    #P = sig*inv(F.T) 


    ########## Yeoh
    #cs = [0.235, -0.007, 0.0008] 
    #lmbda = 0.1
    #d_s = [1/lmbda,1/lmbda,1/lmbda] 
    #Ic_ = Ic*J**(-2./3.)
    #term1 = sum([2*cs[i-1]*(Ic_-3)**(i-1) for i in range(1,4)])*J**(-2/3)*(I-Ic/3*inv(C))
    #term2 = sum([2*i*d_s[i-1]*(J-1)**(2*i-1) for i in range(1,4)])*J*C

    ##psi = sum([cs[i-1]*(Ic-3)**i + d_s[i-1]*(J-1)**(2*i) for i in range(1,4)])

    #S = term1+term2
    #sig = 1/J *F*S*F.T
    #P = sig*inv(F.T) 

    ########## Gao 1993 -> Change  N and mu to get different behaviours
    #N = 5
    #psi = mu/2.*(Ic**N+Ic_inv**N) + lmbda*(J-1)**2

    ########## Carroll XXX Not working!
    #A, B, C = 0.15, 3.1e7, 0.095
    #psi = A*Ic + B*Ic**4 + C*IIc**2

    ########## Arruda    XXX Not working!
    #cp = [1./2., 1./20., 11./1050., 19./7000., 519./673750.]
    #N = 5
    #lmbda_m =  2.78
    ##Ic *= J**(-2./3.)
    #beta = 1/lmbda_m**2
    ###psi = sum([cp[i-1] * beta**(i-1) * (Ic**i-3**i) for i in range(1,N+1)])
    ###psi *= mu
    ###psi += lmbda/2.*(0.5*(J**2-1)-ln(J))
    #sig = sum([i * cp[i-1] * beta**(i-1) * (Ic**(i-1))*B for i in range(1,N+1)])
    #sig *= mu/2.
    #sig += lmbda*(J-1)*I
    #P = sig*inv(F.T) 

    ########## Mooney-Rivlin XXX Not working!
    #C1 = mu/2.
    #C2 = mu/5.
    #D1 = lmbda/2.
    ##Ic *= J**(-2/3)
    ##IIc *= J**(-2/3)
    ##sig = 2./J*((C1+Ic*C2)*B-2./J*C2*B*B)
    ##sig += (D1*(J-1) - 2./(3.*J) * (C1*Ic+2*C2*IIc))*I
    #psi = C1*(Ic-3) + C2*(IIc-3)  + lmbda*(J-1)**2

    #psi = C1*(Ic-3) + C2*(IIc-3) + D1*(J-1)**2
    #P = sig*inv(F.T) 

    # First Piola-Kirchhoff stress tensor
    P =  diff(psi,F)
    #P = sig*inv(F.T) 

    # Create your variational Problem
    deg = 1
    Form = inner(P,grad(v_))*dx(metadata={'quadrature_degree': deg})
    #F = sum([inner(sigma(dv, i, F_macro), eps(v_))*dx(i) for i in range(nphases)])
    #a, L = lhs(F), rhs(F)
    Form += dot(lamb_,u)*dx(metadata={'quadrature_degree': deg}) + dot(c,v_)*dx(metadata={'quadrature_degree': deg})
    Jac  = derivative(Form, w, TrialFunction(W))
    #solve(Form==0, w, [], J=Jac, solver_parameters={"newton_solver":{"relative_tolerance":1e-6,"absolute_tolerance":1e-6,"relaxation_parameter":1.,"convergence_criterion":'residual','linear_solver':'mumps'}})   
    #solve(Form==0, w, [], solver_parameters={"newton_solver":{"relative_tolerance":1e-6,"absolute_tolerance":1e-6,"relaxation_parameter":1.,"convergence_criterion":'residual','linear_solver':'mumps'}})   
    #print(NonlinearVariationalProblem(Form, w, [], Jac))
    NewtonSolver(Form, w, Jac)
    return w.vector().get_local()
    #A, b = assemble_system(Jac, Form, [])
    #A = assemble(Jac)
    #b = assemble()
    #print(A.array())
    #print(np.linalg.inv(A.array()).dot(b.get_local()))
    #print(b.get_local())

    ##lhs_form, rhs_form = lhs(Form), rhs(Form)
    ###rhs_form = rhs(Form)
    ##print(assemble(lhs_form))
    ##print(assemble(rhs_form).get_local())

def NewtonSolver(G, u, dG):
    tol = 1e-10
    error = 1.
    it = 0
    
    while error > tol and it < 10:
        # Assemble
        dg, g = assemble_system(dG, G)
        
        # PETSc solver
        dGp = dolfin.as_backend_type(dg).mat()
        x, b = dGp.getVecs()
        b.setValues(range(g.size()), g)
        ksp = PETSc.KSP().create()
        ksp.setType('chebyshev')
        ksp.setOperators(dGp)
        ksp.solve(b, x)
        

        error = g.norm('l2')
        print('Iteration:', str(it), '; Error: %3.3e' % error)
        if error < tol:
            break
        
        # Update
        u.vector()[:] = u.vector()[:] - x
        it += 1

(spit_strain_stress(1.))


