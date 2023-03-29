
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Dictionary.h>
#include <jive/util/utilities.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/mesh/Geometry.h>


JEM_DEFINE_CLASS( jive::mesh::Geometry );


JIVE_BEGIN_PACKAGE( mesh )


using jem::newInstance;
using jive::util::joinNames;
using jive::util::ItemGroup;
using jive::util::ArrayItemGroup;


//=======================================================================
//   class Geometry
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  Geometry::PI = 3.14159265358979323846;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Geometry::Geometry ( const String& name ) :

  Super ( name )

{}


Geometry::~Geometry ()
{}


//-----------------------------------------------------------------------
//   configure & getConfig
//-----------------------------------------------------------------------


void Geometry::configure ( const Properties& props )
{}


void Geometry::getConfig ( const Properties& conf ) const

{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Geometry::getContext () const
{
  return NamedObject::makeContext ( "geometry", myName_ );
}


//-----------------------------------------------------------------------
//   addGroup
//-----------------------------------------------------------------------


void Geometry::addGroup

  ( const Params&     params,
    const String&     name,
    const IdxVector&  inodes,
    const NodeSet&    nodes )

{
  if ( nodes && params.nodeGroups )
  {
    Ref<ItemGroup>  group =

      newInstance<ArrayItemGroup> ( inodes.clone(),
                                    nodes .getData() );

    params.nodeGroups->insert (
      joinNames ( params.groupPrefix, name ),
      group
    );
  }
}


void Geometry::addGroup

  ( const Params&      params,
    const String&      name,
    const IdxVector&   ielems,
    const ElementSet&  elems )

{
  if ( elems && params.elemGroups )
  {
    Ref<ItemGroup>  group =

      newInstance<ArrayItemGroup> ( ielems.clone(),
                                    elems .getData() );
    params.elemGroups->insert (
      joinNames ( params.groupPrefix, name ),
      group
    );
  }
}


//=======================================================================
//   class Geometry::Params
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


Geometry::Params::Params

  ( double  elemSize,
    double  scale )

{
  JEM_PRECHECK ( elemSize > 0.0 && scale > 0.0 );

  this->elemSize = elemSize;
  this->scale    = scale;
}


JIVE_END_PACKAGE( mesh )
