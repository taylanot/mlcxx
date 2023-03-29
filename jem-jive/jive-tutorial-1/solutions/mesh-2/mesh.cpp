
#include <jem/base/System.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>

using namespace jem;
using namespace jive;
using namespace jive::fem;

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------

void print ( const ElementSet& elems )
{
  NodeSet    nodes = elems.getNodes ();

  IdxVector  inodes;
  idx_t inode;
  Vector coords2;
  Matrix     coords;

  for ( idx_t ielem = 0; ielem < elems.size(); ielem++ )
  {
    const idx_t  n = elems.getElemNodeCount ( ielem );

    inodes.resize ( n );
    coords.resize ( nodes.rank(), n );
    coords2.resize ( nodes.rank() );

    elems.getElemNodes  ( inodes, ielem );
    nodes.getSomeCoords ( coords, inodes );
    nodes.getNodeCoords ( coords2 , inodes[0] );

    System::out() << "Element " << elems.getElemID( ielem ) << "\n\n";
    System::out() << "  nodes  : "  << inodes << "\n";
    System::out() << "  coords :\n" << coords << "\n\n";
    System::out() << "  coords2 :\n" << coords2 << "\n\n";
  }
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  const idx_t     NX     = 10;
  const idx_t     NY     = 10;
  const double    DX     = 1.0 / (double) NX;
  const double    DY     = 2.0 / (double) NY;

  XNodeSet        nodes  = newXNodeSet    ();
  XElementSet     elems  = newXElementSet ( nodes );

  IdxVector       inodes ( 4 );
  Vector          xpos   ( 2 );


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

  print ( elems );

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
