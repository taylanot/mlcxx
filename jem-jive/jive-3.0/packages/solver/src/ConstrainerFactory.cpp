
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
#include <jem/util/Properties.h>
#include <jive/util/CtorMap.h>
#include <jive/solver/ConstrainerFactory.h>


JIVE_BEGIN_PACKAGE( solver )


//=======================================================================
//   class ConstrainerFactory::CtorMap_
//=======================================================================


class ConstrainerFactory::CtorMap_ :

  public util::CtorMap<Constructor>

{};


//=======================================================================
//   class ConstrainerFactory
//=======================================================================

//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ConstrainerFactory::declare

  ( const String&  type,
    Constructor    ctor )

{
  JEM_PRECHECK ( ctor );

  CtorMap_::insert ( type, ctor );
}


//-----------------------------------------------------------------------
//   exists
//-----------------------------------------------------------------------


bool ConstrainerFactory::exists ( const String& type )
{
  return CtorMap_::contains ( type );
}


//-----------------------------------------------------------------------
//   listKnownTypes
//-----------------------------------------------------------------------


StringVector ConstrainerFactory::listKnownTypes ()
{
  return CtorMap_::listKnownTypes ();
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<Constrainer> ConstrainerFactory::newInstance

  ( const String&        type,
    const String&        name,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  Constructor ctor = CtorMap_::find ( type );

  if ( ctor )
  {
    return ctor ( name, conf, props, cons, matrix );
  }
  else
  {
    return nullptr;
  }
}


Ref<Constrainer> ConstrainerFactory::newInstance

  ( const String&        name,
    const Properties&    conf,
    const Properties&    props,
    Ref<Constraints>     cons,
    Ref<AbstractMatrix>  matrix )

{
  Properties        subConf  = conf .makeProps ( name );
  Properties        subProps = props.getProps  ( name );

  Ref<Constrainer>  conman;

  String            type;


  subProps.get ( type, TYPE_PROP );
  subConf .set ( TYPE_PROP, type );

  conman = newInstance ( type, name,
                         conf, props,
                         cons, matrix );

  if ( ! conman )
  {
    if ( CtorMap_::contains( type ) )
    {
      subProps.propertyError (
        TYPE_PROP,
        String::format (
          "invalid constrainer: `%s\' (wrong matrix type)", type
        )
      );
    }
    else
    {
      noSuchTypeError (
        subProps.getContext ( TYPE_PROP ),
        type, "constrainer",
        CtorMap_::listKnownTypes ()
      );
    }
  }

  return conman;
}


JIVE_END_PACKAGE( solver )
