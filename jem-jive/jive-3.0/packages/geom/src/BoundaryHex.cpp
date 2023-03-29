
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
#include <jive/geom/StdCube.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/Parametric4DSurf.h>
#include <jive/geom/BShapeFactory.h>
#include <jive/geom/BoundaryHex.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::newInstance;


//=======================================================================
//   class BoundaryHex
//=======================================================================


class BoundaryHex
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


Ref<BShape> BoundaryHex::getShape

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
    StdCube::GEOMETRY,
    conf, props, sfType,
    StdCube::RANK
  );

  props.find ( ischeme, PropNames::ISCHEME );

  try
  {
    shape = ctor (
      name,
      StdCube::getIntegrationScheme ( ischeme ),
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
//   class BoundaryHex8
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BoundaryHex8::TYPE_NAME = "BHex8";
const char*  BoundaryHex8::ISCHEME   = "2*2*2";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<BShape> BoundaryHex8::getShape

  ( const String&       name,
    const Ref<SShape>&  sfuncs )

{
  String  ischeme = ISCHEME;

  return getShape ( name, ischeme, sfuncs );
}


Ref<BShape> BoundaryHex8::getShape

  ( const String&       name,
    String&             ischeme,
    const Ref<SShape>&  sfuncs )

{
  return getShape (
    name,
    StdCube::getIntegrationScheme ( ischeme ),
    sfuncs
  );
}


Ref<BShape> BoundaryHex8::getShape

  ( const String&       name,
    const Matrix&       ischeme,
    const Ref<SShape>&  sfuncs )

{
  return newInstance<Parametric4DSurf> (
    name,
    ischeme, newInstance<StdCube8>(), sfuncs
  );
}


Ref<BShape> BoundaryHex8::getShape

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  String  ischeme = ISCHEME;

  return BoundaryHex::getShape ( name,
                                 conf .makeProps ( name ),
                                 props.findProps ( name ),
                                 & getShape,
                                 ischeme,
                                 "Linear" );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BoundaryHex8::declare ()
{
  BShapeFactory::declare ( TYPE_NAME,                  & getShape );
  BShapeFactory::declare ( "jive::geom::BoundaryHex8", & getShape );
}


JIVE_END_PACKAGE( geom )
