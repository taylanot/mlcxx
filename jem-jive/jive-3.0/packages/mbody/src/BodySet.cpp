
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
#include <jem/base/Class.h>
#include <jem/base/IllegalInputException.h>
#include <jive/mbody/Globdat.h>
#include <jive/mbody/BodySet.h>


JIVE_BEGIN_PACKAGE( mbody )


//=======================================================================
//   class BodySet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BodySet::CLASS_NAME_ = "jive::mbody::BodySet";


//-----------------------------------------------------------------------
//   getNodesOf
//-----------------------------------------------------------------------


IdxVector BodySet::getNodesOf

  ( const IdxVector&  ibodies ) const

{
  JEM_PRECHECK ( *this );

  return getElements ()

    .getNodesOf ( bodies_().getMembersOf( ibodies ) );
}


//-----------------------------------------------------------------------
//   getUniqueNodesOf
//-----------------------------------------------------------------------


IdxVector BodySet::getUniqueNodesOf

  ( const IdxVector&  ibodies ) const

{
  JEM_PRECHECK ( *this );

  return getElements ()

    .getUniqueNodesOf ( bodies_().getMembersOf( ibodies ) );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void BodySet::store

  ( BodyKind           kind,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( Globdat::BODIES[kind], globdat );
}


void BodySet::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  JEM_PRECHECK ( *this );

  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


BodySet BodySet::find

  ( BodyKind           kind,
    const Properties&  globdat )

{
  return find ( Globdat::BODIES[kind], globdat );
}


BodySet BodySet::find

  ( const String&      name,
    const Properties&  globdat )

{
  using jem::dynamicCast;
  using jive::util::ItemSet;

  Ref<Data>  data =

    dynamicCast<Data> ( ItemSet::find( name, globdat ) );

  if ( data && isValidData( data ) )
  {
    return BodySet ( data );
  }
  else
  {
    return BodySet ();
  }
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


BodySet BodySet::get

  ( BodyKind           kind,
    const Properties&  globdat,
    const String&      context )

{
  BodySet  bodies = find ( Globdat::BODIES[kind], globdat );

  if ( ! bodies )
  {
    String  msg;

    if ( kind == REGULAR_BODIES )
    {
      msg = "no bodies have been defined";
    }
    else
    {
      msg = "no surface bodies have been defined";
    }

    throw jem::IllegalInputException ( context, msg );
  }

  return bodies;
}


BodySet BodySet::get

  ( const String&      name,
    const Properties&  globdat,
    const String&      context )

{
  BodySet  bodies = find ( name, globdat );

  if ( ! bodies )
  {
    throw jem::IllegalInputException (
      context,
      "undefined body set: " + name
    );
  }

  return bodies;
}


//-----------------------------------------------------------------------
//   isValidData
//-----------------------------------------------------------------------


bool BodySet::isValidData ( const Ref<Data>& data )
{
  if ( data )
  {
    return ElementSet::Data::getType()->isBaseOf (
      data->getGroupedItems()->getClass ()
    );
  }
  else
  {
    return true;
  }
}


JIVE_END_PACKAGE( mbody )
