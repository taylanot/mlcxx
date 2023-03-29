
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
  XNodeSet       nodes  = newXNodeSet    ();
  XElementSet    elems  = newXElementSet ( nodes );

  // Make the mesh here ...

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


  // Calculate the mesh area here ...

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
