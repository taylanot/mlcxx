
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/rtti.h>
#include <jem/base/array/Array.h>
#include <jem/util/Dictionary.h>
#include <jem/util/Properties.h>
#include <jem/gl/Vertex.h>


JEM_BEGIN_PACKAGE( gl )


using jem::util::Dictionary;

extern const GLdouble  RADS_PER_DEGREE;
extern const GLdouble  DEGREES_PER_RAD;


//=======================================================================
//   class Vertex
//=======================================================================

//-----------------------------------------------------------------------
//   configure_ (2-D)
//-----------------------------------------------------------------------


bool Vertex::configure_

  ( double&            x,
    double&            y,
    const String&      name,
    const Properties&  props )

{
  Ref<Object>  obj;


  if ( ! props.find( obj, name ) )
  {
    return false;
  }

  if ( isInstance< Array<double> >( obj ) )
  {
    const Array<double>&  tmp = toValue< Array<double> > ( obj );

    if ( tmp.size() != 2 )
    {
      props.propertyError (
        name,
        String::format (
          "invalid array length (%d); should be 2", tmp.size ()
        )
      );
    }

    x = tmp[0];
    y = tmp[1];

    return true;
  }

  if ( isInstance<Dictionary>( obj ) )
  {
    Properties  tmp = props.getProps ( name );

    tmp.get ( x, "x" );
    tmp.get ( y, "y" );

    return true;
  }

  props.propertyError (
    name,
    "property has wrong type; expected a 2-D vertex"
  );

  return false;
}


//-----------------------------------------------------------------------
//   configure_ (3-D vertex)
//-----------------------------------------------------------------------


bool Vertex::configure_

  ( double&            x,
    double&            y,
    double&            z,
    const String&      name,
    const Properties&  props )

{
  Ref<Object>  obj;


  if ( ! props.find( obj, name ) )
  {
    return false;
  }

  if ( isInstance< Array<double> >( obj ) )
  {
    const Array<double>&  tmp = toValue< Array<double> > ( obj );

    if ( tmp.size() != 3 )
    {
      props.propertyError (
        name,
        String::format (
          "invalid array length (%d); should be 3", tmp.size ()
        )
      );
    }

    x = tmp[0];
    y = tmp[1];
    z = tmp[2];

    return true;
  }

  if ( isInstance<Dictionary>( obj ) )
  {
    Properties  tmp = props.getProps ( name );

    tmp.get ( x, "x" );
    tmp.get ( y, "y" );
    tmp.get ( z, "z" );

    return true;
  }

  props.propertyError (
    name,
    "property has wrong type; expected a 3-D vertex"
  );

  return false;
}


//-----------------------------------------------------------------------
//   getConfig_
//-----------------------------------------------------------------------


void Vertex::getConfig_

  ( const String&      name,
    const Properties&  props,
    double             x,
    double             y )

{
  Array<double>  tmp ( 2 );

  tmp[0] = x;
  tmp[1] = y;

  props.set ( name, tmp );
}


void Vertex::getConfig_

  ( const String&      name,
    const Properties&  props,
    double             x,
    double             y,
    double             z )

{
  Array<double>  tmp ( 3 );

  tmp[0] = x;
  tmp[1] = y;
  tmp[2] = z;

  props.set ( name, tmp );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   getAngle
//-----------------------------------------------------------------------


GLdouble            getAngle

  ( const Vertex2d&   e1,
    const Vertex2d&   e2 )

{
  GLdouble  a, s, t, u;

  s =

    std::sqrt ( e1[0] * e1[0] + e1[1] * e1[1] ) *
    std::sqrt ( e2[0] * e2[0] + e2[1] * e2[1] );

  if ( s > 0.0 )
  {
    s = 1.0 / s;
  }
  else
  {
    s = 0.0;
  }

  t = s * (e1[0] * e2[1] - e1[1] * e2[0]);
  u = s * (e1[0] * e2[0] + e1[1] * e2[1]);

  a = DEGREES_PER_RAD * ::asin ( t );

  if      ( u < 0.0 )
  {
    a = 180.0 - a;
  }
  else if ( t < 0.0 )
  {
    a = 360.0 + a;
  }

  return a;
}


JEM_END_PACKAGE( gl )
