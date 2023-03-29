
#include <cmath>
#include <jem/base/Array.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/fem/XNodeSet.h>
#include "Array.h"
#include "ImpingeModule.h"

using jive::app::Module;

//=======================================================================
//   class ImpingeModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

ImpingeModule::ImpingeModule ( const String& name ) :

  Module ( name )

{}

ImpingeModule::~ImpingeModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status ImpingeModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::ALL;
  using jem::System;
  using jive::fem::XNodeSet;

  Properties  myProps = props.findProps ( myName_ );
  Properties  myConf  = conf .makeProps ( myName_ );
  XNodeSet    nodes   = XNodeSet::get   ( globdat, getContext() );

  Matrix      coords;
  Vector      impsz;
  double      scale;
  double      length;
  double      width;
  double      y0, y1, ym;
  double      x0, x1;


  if ( nodes.rank() != 2 )
  {
    System::warn() << "No impinging applied; wrong number of "
                   << "dimensions.\n";

    return DONE;
  }

  if ( ! myProps.find( impsz, "size", 0.0, 1.0e20 ) )
  {
    // No impingement size; forget about it.

    return DONE;
  }

  myConf.set ( "size", impsz );

  if ( impsz.size() != 2 )
  {
    myProps.propertyError (
      "size",
      "invalid array size; should be two"
    );
  }

  coords.resize    ( 2, nodes.size() );
  nodes .getCoords ( coords );

  x0 = min ( coords(0,ALL) );
  y0 = min ( coords(1,ALL) );
  x1 = max ( coords(0,ALL) );
  y1 = max ( coords(1,ALL) );
  ym = 0.5 * (y0 + y1);

  length = x1 - x0;
  width  = y1 - y0;

  if ( (length   * width    <= 0.0) ||
       (impsz[0] * impsz[1] <= 0.0) )
  {
    // Nothing to do.

    return DONE;
  }

  System::out() << "Applying mesh impingement ...\n";

  scale = (y1 - y0 - impsz[1]) / (y1 - y0);

  for ( idx_t i = 0; i < coords.size(1); i++ )
  {
    double  x = coords(0,i) - x0;
    double  y = coords(1,i) - y0;

    if ( (x >= 0.5 * (length - impsz[0])) &&
         (x <= 0.5 * (length + impsz[0])) )
    {
      double  t = x - 0.5 * (length - impsz[0]);
      double  s = 1.0;

      t /= impsz[0];
      s  = 0.5 * (1.0 + std::cos( 2.0 * t * 3.14159265359 ));
      s  = 1.0 + scale * (s - 1.0);
      y  = ym + (y - ym) * s;

      coords(1,i) = y;
    }
  }

  nodes.setCoords ( coords );

  return DONE;
}
