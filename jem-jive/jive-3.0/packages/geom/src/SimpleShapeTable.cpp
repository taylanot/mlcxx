
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


#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jive/util/ItemSet.h>
#include <jive/geom/Shape.h>
#include <jive/geom/SimpleShapeTable.h>


JEM_DEFINE_SERIABLE_CLASS( jive::geom::SimpleShapeTable );


JIVE_BEGIN_PACKAGE( geom )


//=======================================================================
//   class SimpleShapeTable
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SimpleShapeTable::SimpleShapeTable ()
{}


SimpleShapeTable::SimpleShapeTable

  ( const String&        name,
    const Ref<ItemSet>&  items,
    const IdxVector&     map,
    const ShapeMatrix&   shapes ) :

    Super ( name, items )

{
  JEM_PRECHECK2 ( map.size() <= items->size(),
                  "shape map array has wrong size" );

  shapes_.ref ( shapes );
  map_   .ref ( map    );

}


SimpleShapeTable::~SimpleShapeTable ()
{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void SimpleShapeTable::readFrom ( ObjectInput& in )
{
  decode   ( in, myName_, items_, shapes_, map_ );
  connect_ ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void SimpleShapeTable::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, items_, shapes_, map_ );
}


JIVE_END_PACKAGE( geom )
