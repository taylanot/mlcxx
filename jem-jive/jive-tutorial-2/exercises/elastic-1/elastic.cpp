/*
 * This example program solves a 1-D elasticity problem by applying
 * the finite element method. The calculated displacements in the
 * nodes are saved to the output file "disp.out".
 */

#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/io/FileWriter.h>
#include <jive/Array.h>
#include <jive/algebra/AbstractMatrix.h>
#include <jive/algebra/FlexMatrixBuilder.h>
#include <jive/solver/SparseLU.h>
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
using jive::algebra::MatrixBuilder;

//-----------------------------------------------------------------------
//   makeMesh()
//-----------------------------------------------------------------------

// Creates the finite element mesh. To be precise, this function
// determines the node coordinates and the element connectivities
// (which element is connected to which nodes).

void                      makeMesh

  ( Vector&                 coords,  // node coordinates
    IdxMatrix&              elcon,   // element connectivities
    const Params&           params ) // global parameters

{
  double  dx = params.length / params.nelm;

  coords.resize (    params.nnod );
  elcon .resize ( 2, params.nelm );

  for ( idx_t inod = 0; inod < params.nnod; inod++ )
  {
    coords[inod] = inod * dx;
  }

  for ( idx_t ielm = 0; ielm < params.nelm; ielm++ )
  {
    elcon(0,ielm) = ielm + 0;  // first node
    elcon(1,ielm) = ielm + 1;  // second node
  }
}

//-----------------------------------------------------------------------
//   assemble()
//-----------------------------------------------------------------------

// Assembles the global stiffness matrix and the right-hand side
// vector.

void                      assemble

  ( MatrixBuilder&          mbld,      // matrix builder
    Vector&                 rhs,       // right-hand side vector
    const Vector&           coords,    // node coordinates
    const IdxMatrix&        elcon,     // element connectivities
    const Params&           params )   // global parameters

{
  Matrix     elmmat ( 2, 2 );
  Vector     elmvec ( 2 );
  IdxVector  elmnod ( 2 );
  double     coeff;
  double     dx;

  // Set the global stiffness matrix and the right-hand side
  // vector to zero first.

  mbld.setToZero ();
  rhs .resize    ( params.nnod );

  rhs = 0.0;

  // Main loop over the elements.

  for ( idx_t ielm = 0; ielm < params.nelm; ielm++ )
  {
    // Store the two element nodes in a local array.

    elmnod = elcon[ielm];
    dx     = coords[elmnod[1]] - coords[elmnod[0]];

    // Compute the element stiffness matrix.

    coeff       =  params.young * params.area / dx;
    elmmat(0,0) =  coeff;
    elmmat(0,1) = -coeff;
    elmmat(1,0) = -coeff;
    elmmat(1,1) =  coeff;

    // Add the element stiffness matrix to the global
    // stiffness matrix.

    mbld.addBlock ( elmnod, elmnod, elmmat );

    // Compute the element force vector that incorporates the
    // traction along the bar and add it to the right-hand
    // side vector.

    coeff        = 0.5 * dx * params.tract;
    elmvec[0]    = coeff;
    elmvec[1]    = coeff;
    rhs[elmnod] += elmvec;
  }

  // Add the force acting on the right end of the bar to the
  // right-hand side vector.

  rhs[params.nnod - 1] += params.force;

  // Account for the zero displacement in the first node by adding
  // a large value to the first diagonal entry in global
  // stiffness matrix.

  mbld.addValue ( 0, 0, 1.0e24 );
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
    const Vector&           coords )   // node coordinates

{
  using jem::io::Writer;
  using jem::io::FileWriter;

  Ref<Writer>  out = newInstance<FileWriter> ( "disp.out" );

  *out << "# Column 1: node number.\n";
  *out << "# Column 2: node coordinate [m].\n";
  *out << "# Column 3: node displacement [m].\n\n";

  for ( idx_t inod = 0; inod < disp.size(); inod++ )
  {
    *out << inod << ' ' << coords[inod];
    *out         << ' ' << disp[inod] << "\n";
  }
}


//-----------------------------------------------------------------------
//   run()
//-----------------------------------------------------------------------

// Executes the main steps in the program.

int run ()
{
  using jive::algebra::FlexMatrixBuilder;

  Ref<MatrixBuilder>  mbld;    // global stiffness matrix

  Vector              coords;  // node coordinates
  IdxMatrix           elcon;   // element connectivities
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

  makeMesh ( coords, elcon, params );

  System::out() << "Assembling the system of equations ...\n";

  mbld = newInstance<FlexMatrixBuilder> ();

  assemble ( *mbld, rhs, coords, elcon, params );

  System::out() << "Solving the system of equations ...\n";

  solve ( disp, *mbld, rhs );

  System::out() << "Saving the results ...\n";

  saveDisp ( disp, coords );

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
