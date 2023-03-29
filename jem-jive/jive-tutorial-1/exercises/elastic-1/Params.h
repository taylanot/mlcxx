#ifndef PARAMS_H
#define PARAMS_H

// This header file is needed for the type jem::idx_t.

#include <jem/defines.h>

//-----------------------------------------------------------------------
//   class Params
//-----------------------------------------------------------------------

class Params
{
 public:

                        Params  ();

  void                  read    ();


 public:

  // Number of elements and number of nodes.

  jem::idx_t            nelm;
  jem::idx_t            nnod;

  // Other parameters:
  //
  //   force  : force [N] applied to the right end of the bar;
  //   tract  : traction [N/m] applied along the bar;
  //   area   : cross-sectional area [m^2];
  //   young  : Young's modulus [Pa];
  //   length : length of the bar [m].

  double                force;
  double                tract;
  double                area;
  double                young;
  double                length;

};

#endif
