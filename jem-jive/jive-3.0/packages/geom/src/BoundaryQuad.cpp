
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */


#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/geom/Names.h>
#include <jive/geom/StdSquare.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/ParametricSurface.h>
#include <jive/geom/BShapeFactory.h>
#include <jive/geom/BoundaryQuad.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class BoundaryQuad
//=======================================================================


class BoundaryQuad
{
 public:

  typedef Ref<BShape>     (*BShapeCtor)

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<SShape>&      sfuncs );


  static Ref<BShape>        getShape

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      BShapeCtor              ctor,
      String&                 ischeme,
      const String&           sfType );

};


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    BShapeCtor         ctor,
    String&            ischeme,
    const String&      sfType )

{
  Ref<SShape>  sfuncs;
  Ref<BShape>  shape;


  sfuncs = StdShapeFactory::findShapeFuncs (
    StdSquare::GEOMETRY,
    conf, props, sfType,
    StdSquare::RANK
  );

  props.find ( ischeme, PropNames::ISCHEME );

  try
  {
    shape = ctor (
      name,
      StdSquare::getIntegrationScheme ( ischeme ),
      sfuncs
    );
  }
  catch ( jem::IllegalInputException& ex )
  {
    ex.setContext ( props.getContext( PropNames::ISCHEME ) );
    throw;
  }

  conf.set ( PropNames::ISCHEME, ischeme );

  return shape;
}


//=======================================================================
//   class BoundaryQuad4
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryQuad4::TYPE_NAME = "BQuad4";
const char*  BoundaryQuad4::ISCHEME   = "2*2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad4::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryQuad4::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdSquare::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryQuad4::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme, newInstance<StdSquare4>(), sfuncs
  );
}


Ref<BShape> BoundaryQuad4::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryQuad::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Linear" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryQuad4::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryQuad4", & getShape );
}


//=======================================================================
//   class BoundaryQuad6
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryQuad6::TYPE_NAME = "BQuad6";
const char*  BoundaryQuad6::ISCHEME   = "3*2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad6::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryQuad6::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdSquare::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryQuad6::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme, newInstance<StdSquare6>(), sfuncs
  );
}


Ref<BShape> BoundaryQuad6::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryQuad::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Quadratic*Linear" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryQuad6::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryQuad6", & getShape );
}


//=======================================================================
//   class BoundaryQuad8
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryQuad8::TYPE_NAME = "BQuad8";
const char*  BoundaryQuad8::ISCHEME   = "3*3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad8::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryQuad8::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdSquare::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryQuad8::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme, newInstance<StdSquare8>(), sfuncs
  );
}


Ref<BShape> BoundaryQuad8::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryQuad::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Quadratic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryQuad8::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryQuad8", & getShape );
}


//=======================================================================
//   class BoundaryQuad9
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryQuad9::TYPE_NAME = "BQuad9";
const char*  BoundaryQuad9::ISCHEME   = "3*3";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad9::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryQuad9::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdSquare::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryQuad9::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme, newInstance<StdSquare9>(), sfuncs
  );
}


Ref<BShape> BoundaryQuad9::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryQuad::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "BiQuadratic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryQuad9::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                   & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryQuad9", & getShape );
}


//=======================================================================
//   class BoundaryQuad12
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryQuad12::TYPE_NAME = "BQuad12";
const char*  BoundaryQuad12::ISCHEME   = "4*4";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryQuad12::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryQuad12::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdSquare::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryQuad12::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<ParametricSurface> (
    name,
    ischeme, newInstance<StdSquare12>(), sfuncs
  );
}


Ref<BShape> BoundaryQuad12::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryQuad::getShape ( name,
                                  conf .makeProps ( name ),
                                  props.findProps ( name ),
                                  & getShape,
                                  ischeme,
                                  "Cubic" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryQuad12::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                    & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryQuad12", & getShape );
}


JIVE_END_PACKAGE( geom )
