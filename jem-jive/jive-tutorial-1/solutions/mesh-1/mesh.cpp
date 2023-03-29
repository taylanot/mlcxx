
#include <jem/base/System.h>
#include <jive/util/StdPointSet.h>
#include <jive/util/StdGroupSet.h>

using namespace jem;
using namespace jive;
using namespace jive::util;

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

int run ()
{
  const idx_t     NX     = 10;
  const idx_t     NY     = 10;
  const double    DX     = 1.0 / (double) NX;
  const double    DY     = 2.0 / (double) NY;

  Ref<XPointSet>  nodes;
  Ref<XGroupSet>  elems;

  IdxVector       inodes ( 4 );
  Vector          xpos   ( 2 );


  nodes = newInstance<StdPointSet> ( "nodes", "node" );
  elems = newInstance<StdGroupSet> ( "elems", "element", nodes );

  // Create the nodes first.

  for ( idx_t j = 0; j <= NY; j++ )
  {
    xpos[1] = j * DY;

    for ( idx_t i = 0; i <= NX; i++ )
    {
      xpos[0] = i * DX;

      nodes->addPoint ( xpos );
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

      elems->addGroup ( inodes );
    }
  }

  return 0;
}

//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------

int main ()
{
  return System::exec ( run );
}
