
#include <jem/util/Properties.h>
#include "Params.h"

//=======================================================================
//   class Params
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

Params::Params ()
{
  nelm   = 0;
  nnod   = 0;
  force  = 0.0;
  tract  = 0.0;
  area   = 0.0;
  young  = 0.0;
  length = 0.0;
}

//-----------------------------------------------------------------------
//   read()
//-----------------------------------------------------------------------

// Reads the model parameters from the file "elastic.dat".

void Params::read ()
{
  using jem::util::Properties;

  const double  SMALL = 1.0e-20;
  const double  LARGE = 1.0e20;

  Properties    conf;


  conf.parseFile ( "elastic.dat" );

  // The "force" and "tract" parameters are optional; they are
  // zero by default.

  conf.get  ( nelm,   "nelm",   1, 100000 );
  conf.find ( force,  "force" );
  conf.find ( tract,  "tract" );
  conf.get  ( area,   "area",   SMALL, LARGE );
  conf.get  ( young,  "young",  SMALL, LARGE );
  conf.get  ( length, "length", SMALL, LARGE );

  nnod = nelm + 1;
}

