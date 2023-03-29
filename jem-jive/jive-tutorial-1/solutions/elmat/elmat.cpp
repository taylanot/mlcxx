
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/geom/Quad.h>

using namespace jem;
using namespace jive;
using namespace jive::geom;

//-----------------------------------------------------------------------
//   makeElmat
//-----------------------------------------------------------------------

void makeElmat ( const IShape&  shape,
                 const Matrix&  coords )
{
  // Assemble the element matrix using the given shape and
  // node coordinate matrix ...

  using jem::numeric::matmul;

  const idx_t  rank      = shape.globalRank  ();
  const idx_t  ipCount   = shape.ipointCount ();
  const idx_t  nodeCount = shape.nodeCount   ();

  Matrix       elmat     ( nodeCount, nodeCount );
  Cubix        grads     ( rank, nodeCount, ipCount );
  Vector       weights   ( ipCount );


  shape.getShapeGradients ( grads, weights, coords );

  elmat = 0.0;

  for ( idx_t ip = 0; ip < ipCount; ip++ )
  {
    elmat += weights[ip] * matmul ( grads[ip].transpose(),
                                    grads[ip] );
  }

  System::out() << "Element matrix:\n" << elmat << "\n";
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  String       ischeme = "Gauss2 * Gauss2";
  String       bscheme = "Gauss2";
  Ref<IShape>  shape   = Quad4::getShape ( "quad", ischeme, bscheme );

  Matrix       coords  ( 2, 4 );


  coords(0,0) = 0.0;
  coords(1,0) = 0.0;
  coords(0,1) = 1.0;
  coords(1,1) = 0.0;
  coords(0,2) = 1.0;
  coords(1,2) = 1.0;
  coords(0,3) = 0.0;
  coords(1,3) = 1.0;

  makeElmat ( *shape, coords );

  // The element matrix should be unaffected by uniform
  // translations.

  coords += 2.0;

  makeElmat ( *shape, coords );

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}

