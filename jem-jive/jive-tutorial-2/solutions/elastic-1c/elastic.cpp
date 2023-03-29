/*
 * This example program solves a 1-D elasticity problem by applying
 * the finite element method. The calculated displacements in the
 * nodes are saved to the output file "disp.out".
 */

#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/io/FileWriter.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/Array.h>
#include <jive/util/Assignable.h>
#include <jive/util/DenseDofSpace.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/FlexMatrixBuilder.h>
#include <jive/solver/SparseLU.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/geom/Line.h>
#include "Params.h"

// Using declarations for common types.

using jem::idx_t;
using jem::newInstance;
using jem::Ref;
using jem::System;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::util::DofSpace;
using jive::algebra::MatrixBuilder;
using jive::fem::NodeSet;
using jive::fem::ElementSet;

//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------

const char*  DOF_TYPE = "u";

//-----------------------------------------------------------------------
//   makeMesh()
//-----------------------------------------------------------------------

// Creates the finite element mesh. To be precise, this function
// determines the node coordinates and the element connectivities
// (which element is connected to which nodes).

ElementSet                makeMesh

  ( const Params&           params ) // global parameters

{
  using jive::fem::newXNodeSet;
  using jive::fem::newXElementSet;
  using jive::fem::XNodeSet;
  using jive::fem::XElementSet;

  XNodeSet     nodes  = newXNodeSet    ();
  XElementSet  elems  = newXElementSet ( nodes );

  double       dx     = params.length / params.nelm;

  IdxVector    elmnod ( 2 );
  Vector       coords ( 1 );

  // For efficiency:

  nodes.reserve ( params.nnod );
  elems.reserve ( params.nelm );

  for ( idx_t inod = 0; inod < params.nnod; inod++ )
  {
    coords[0] = inod * dx;

    nodes.addNode ( coords );
  }

  for ( idx_t ielm = 0; ielm < params.nelm; ielm++ )
  {
    elmnod[0] = ielm + 0;  // first node
    elmnod[1] = ielm + 1;  // second node

    elems.addElement ( elmnod );
  }

  return elems;
}

//-----------------------------------------------------------------------
//   makeDofs()
//-----------------------------------------------------------------------

// Creates DofSpace and associates DOFs with the nodes.

Ref<DofSpace>             makeDofs

  ( const ElementSet&       elems )

{
  using jive::util::XDofSpace;
  using jive::util::DenseDofSpace;

  NodeSet         nodes = elems.getNodes ();
  Ref<XDofSpace>  dofs  =

    newInstance<DenseDofSpace> ( nodes.getData() );

  idx_t           jtype = dofs->addType ( DOF_TYPE );

  // Just for fun: add the DOFs in reverse order. This should
  // have no effect on the correctness of the code.

  for ( idx_t inod = nodes.size() - 1; inod >= 0; inod-- )
  {
    dofs->addDof ( inod, jtype );
  }

  return dofs;
}

//-----------------------------------------------------------------------
//   assemble()
//-----------------------------------------------------------------------

// Assembles the global stiffness matrix and the right-hand side
// vector.

void                      assemble

  ( MatrixBuilder&          mbld,      // matrix builder
    Vector&                 rhs,       // right-hand side vector
    const DofSpace&         dofs,      // DOFs
    const ElementSet&       elems,     // finite element mesh
    const Params&           params )   // global parameters

{
  using jem::String;
  using jem::numeric::matmul;
  using jive::Cubix;
  using jive::geom::IShape;
  using jive::geom::Line2;

  NodeSet      nodes   = elems.getNodes     ();
  const idx_t  jtype   = dofs .getTypeIndex ( DOF_TYPE );
  String       ischeme = "Gauss1";
  Ref<IShape>  shape   = Line2::getShape    ( "line", ischeme );
  const idx_t  ipCount = shape->ipointCount ();

  Matrix       sfuncs  ( 2, ipCount );
  Cubix        sgrads  ( 1, 2, ipCount );
  Vector       weights ( ipCount );
  Matrix       coords  ( 1, 2 );
  Matrix       elmmat  ( 2, 2 );
  Vector       elmvec  ( 2 );
  IdxVector    elmnod  ( 2 );
  IdxVector    elmdof  ( 2 );

  idx_t        idof;

  // Set the global stiffness matrix and the right-hand side
  // vector to zero first.

  mbld.setToZero ();
  rhs .resize    ( params.nnod );

  rhs = 0.0;

  // Get the shape functions in the integration points. The values of the
  // shape functions do not depend on the node coordinates.

  sfuncs = shape->getShapeFunctions ();

  // Main loop over the elements.

  for ( idx_t ielm = 0; ielm < params.nelm; ielm++ )
  {
    // Get the element nodes and their coordinates.

    elems.getElemNodes  ( elmnod, ielm );
    nodes.getSomeCoords ( coords, elmnod );

    // Get the DOFs associated with the nodes of the element.

    dofs.getDofIndices ( elmdof, elmnod, jtype );

    // Compute the element stiffness matrix and the element force vector
    // by means of numerical integration.

    shape->getShapeGradients ( sgrads, weights, coords );

    elmmat = 0.0;
    elmvec = 0.0;

    for ( idx_t ip = 0; ip < ipCount; ip++ )
    {
      elmmat += weights[ip] * params.young * params.area *

        matmul ( sgrads[ip].transpose(), sgrads[ip] );

      elmvec += weights[ip] * params.tract * sfuncs[ip];
    }

    // Add the element stiffness matrix to the global
    // stiffness matrix and the element force vector to the right-hand
    // side vector.

    mbld.addBlock ( elmdof, elmdof, elmmat );

    rhs[elmdof] += elmvec;
  }

  // Add the force acting on the right end of the bar to the
  // right-hand side vector.

  idof       = dofs.getDofIndex ( params.nnod - 1, jtype );
  rhs[idof] += params.force;

  // Account for the zero displacement in the first node by adding
  // a large value to the first diagonal entry in global
  // stiffness matrix.

  idof = dofs.getDofIndex ( 0, jtype );

  mbld.addValue ( idof, idof, 1.0e24 );
}

//-----------------------------------------------------------------------
//   solve()
//-----------------------------------------------------------------------

// Solves the global system of equations.

void                      solve

  ( Vector&                 lhs,     // solution vector
    MatrixBuilder&          mbld,    // matrix builder
    const Vector&           rhs )    // right-hand side vector

{
  using jem::NIL;
  using jive::solver::Solver;
  using jive::solver::SparseLU;

  // No constraints, so pass NIL as the last argument.

  Ref<Solver>  solver =

    newInstance<SparseLU> ( "solver", mbld.getMatrix(), NIL );

  mbld.updateMatrix ();
  lhs .resize       ( rhs.size() );

  solver->solve ( lhs, rhs );
}

//-----------------------------------------------------------------------
//   saveDisp()
//-----------------------------------------------------------------------

// Saves the displacement in the nodes to an output file.

void                      saveDisp

  ( const Vector&           disp,     // node displacements
    const DofSpace&         dofs,     // DOFs
    const ElementSet&       elems )   // finite element mesh

{
  using jem::io::Writer;
  using jem::io::FileWriter;

  NodeSet      nodes   = elems.getNodes          ();
  Ref<Writer>  out     = newInstance<FileWriter> ( "disp.out" );
  const idx_t  jtype   = dofs.getTypeIndex       ( DOF_TYPE );

  Vector       coords ( 1 );

  *out << "# Column 1: node number.\n";
  *out << "# Column 2: node coordinate [m].\n";
  *out << "# Column 3: node displacement [m].\n\n";

  for ( idx_t inod = 0; inod < disp.size(); inod++ )
  {
    idx_t  idof = dofs.getDofIndex ( inod, jtype );

    nodes.getNodeCoords ( coords, inod );

    *out << inod << ' ' << coords[0];
    *out         << ' ' << disp[idof] << "\n";
  }
}


//-----------------------------------------------------------------------
//   run()
//-----------------------------------------------------------------------

// Executes the main steps in the program.

int run ()
{
  using jive::util::Assignable;
  using jive::algebra::FlexMatrixBuilder;

  Assignable
    <ElementSet>      elems;   // finite element mesh

  Ref<MatrixBuilder>  mbld;    // global stiffness matrix
  Ref<DofSpace>       dofs;    // DOFs

  Vector              disp;    // displacements
  Vector              rhs;     // right-hand side vector

  Params              params;  // global parameters


  System::out() << "\n";
  System::out() << "Reading input parameters ...\n\n";

  params.read ();

  System::out() << "  nodes    : " << params.nnod   << "\n";
  System::out() << "  elements : " << params.nelm   << "\n";
  System::out() << "  force    : " << params.force  << "\n";
  System::out() << "  traction : " << params.tract  << "\n";
  System::out() << "  area     : " << params.area   << "\n";
  System::out() << "  young    : " << params.young  << "\n";
  System::out() << "  length   : " << params.length << "\n\n";

  System::out() << "Creating the finite element mesh ...\n";

  elems = makeMesh ( params );

  System::out() << "Initializing the DOFs ...\n";

  dofs  = makeDofs ( elems );

  System::out() << "Assembling the system of equations ...\n";

  mbld = newInstance<FlexMatrixBuilder> ();

  assemble ( *mbld, rhs, *dofs, elems, params );

  System::out() << "Solving the system of equations ...\n";

  solve ( disp, *mbld, rhs );

  System::out() << "Saving the results ...\n";

  saveDisp ( disp, *dofs, elems );

  System::out() << "Done.\n\n";

  return 0;
}

//-----------------------------------------------------------------------
//   main()
//-----------------------------------------------------------------------

// The main function. This is where the program starts.

int main ()
{
  return System::exec ( run );
}
