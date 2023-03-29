
#include <jem/base/System.h>
#include <jive/util/FlexDofSpace.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>

using namespace jem;
using namespace jive;
using namespace jive::util;
using namespace jive::fem;

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------

void print ( const ElementSet& elems,
             const DofSpace&   dofs )
{
  NodeSet    nodes  = elems.getNodes ();

  IdxVector  jtypes ( dofs.typeCount() );

  IdxVector  inodes;
  IdxVector  idofs;
  Matrix     coords;

  for ( idx_t i = 0; i < jtypes.size(); i++ )
  {
    jtypes[i] = i;
  }

  for ( idx_t ielem = 0; ielem < elems.size(); ielem++ )
  {
    const idx_t  n = elems.getElemNodeCount ( ielem );

    inodes.resize ( n );
    idofs .resize ( n * jtypes.size() );
    coords.resize ( nodes.rank(), n );

    elems.getElemNodes  ( inodes, ielem );
    nodes.getSomeCoords ( coords, inodes );
    dofs .getDofIndices ( idofs,  inodes, jtypes );

    System::out() << "Element " << elems.getElemID( ielem ) << "\n\n";
    System::out() << "  nodes  : "  << inodes << "\n";
    System::out() << "  DOFs   : "  << idofs  << "\n";
    System::out() << "  coords :\n" << coords << "\n\n";
  }
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  const idx_t     NX     = 2;
  const idx_t     NY     = 2;
  const double    DX     = 1.0 / (double) NX;
  const double    DY     = 2.0 / (double) NY;

  XNodeSet        nodes  = newXNodeSet    ();
  XElementSet     elems  = newXElementSet ( nodes );

  IdxVector       jtypes ( 2 );
  IdxVector       inodes ( 4 );
  Vector          xpos   ( 2 );

  Ref<XDofSpace>  dofs;


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
  // Define the DOFs.

  dofs      = newInstance<FlexDofSpace> ( nodes.getData() );
  jtypes[0] = dofs->addType ( "dx" );
  jtypes[1] = dofs->addType ( "dy" );

  for ( idx_t inode = 0; inode < nodes.size(); inode++ )
  {
    dofs->addDofs ( inode, jtypes );
  }



  print ( elems, *dofs );

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
