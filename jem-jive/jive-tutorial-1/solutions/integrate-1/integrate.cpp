
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/geom/Quad.h>

using namespace jem;
using namespace jive;
using namespace jive::fem;
using namespace jive::geom;

//-----------------------------------------------------------------------
//   makeShape
//-----------------------------------------------------------------------

Ref<Shape> makeShape ()
{
  // The interior and boundary integration schemes; the latter is
  // not used here, but it must be specified.

  String  ischeme = "Gauss2 * Gauss2";
  String  bscheme = "Gauss2";

  return Quad4::getShape ( "quad", ischeme, bscheme );
}

//-----------------------------------------------------------------------
//   makeMesh
//-----------------------------------------------------------------------

ElementSet makeMesh ()
{
  const idx_t    NX     = 10;
  const idx_t    NY     = 10;
  const double   DX     = 1.0 / (double) NX;
  const double   DY     = 2.0 / (double) NY;

  XNodeSet       nodes  = newXNodeSet    ();
  XElementSet    elems  = newXElementSet ( nodes );

  IdxVector      inodes ( 4 );
  Vector         xpos   ( 2 );


  // Create the nodes first.

  for ( idx_t j = 0; j <= NY; j++ )
  {
    xpos[1] = j * DY;

    for ( idx_t i = 0; i <= NX; i++ )
    {
      xpos[0] = i * DX;

      nodes.addNode ( xpos );
    }
  }

  // Now create the elements.

  for ( idx_t j = 0; j < NY; j++ )
  {
    for ( idx_t i = 0; i < NX; i++ )
    {
      inodes[0] = i + j * (NX + 1);
      inodes[1] = inodes[0] + 1;
      inodes[2] = inodes[1] + (NX + 1);
      inodes[3] = inodes[2] - 1;

      elems.addElement ( inodes );
    }
  }

  return elems;
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  ElementSet  elems   = makeMesh       ();
  NodeSet     nodes   = elems.getNodes ();
  Ref<Shape>  shape   = makeShape      ();

  IdxVector   inodes  ( 4 );
  Matrix      coords  ( 2, 4 );
  Vector      weights ( shape->ipointCount() );

  double      area    = 0.0;


  for ( idx_t ielem = 0; ielem < elems.size(); ielem++ )
  {
    elems.getElemNodes  ( inodes, ielem );
    nodes.getSomeCoords ( coords, inodes );

    shape->getIntegrationWeights ( weights, coords );

    area += sum ( weights );
  }

  System::out() << "Mesh surface area = " << area << "\n";

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
