
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


#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/GroupConverter.h>
#include <jive/app/GroupconvModule.h>


JEM_DEFINE_CLASS( jive::app::GroupconvModule );


JIVE_BEGIN_PACKAGE( app )


//=======================================================================
//   class GroupconvModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GroupconvModule::TYPE_NAME = "Groupconv";
const int    GroupconvModule::MAX_ITER  = 100;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GroupconvModule::GroupconvModule

  ( const String&          name,
    const Ref<Converter>&  conv ) :

    Super ( name ),
    conv_ ( conv )

{}


GroupconvModule::~GroupconvModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GroupconvModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::System;


  Properties      myConf  = conf .makeProps ( myName_ );
  Properties      myProps = props.findProps ( myName_ );

  Ref<Converter>  conv    = conv_;

  int             iter;
  idx_t           k, n;


  if ( ! conv )
  {
    conv = newConverter_ ( conf, props, globdat );
  }

  if ( ! conv )
  {
    return DONE;
  }

  print ( System::out(), "Converting item groups ...\n" );

  conv->init ( myConf, myProps );

  n = 0;

  for ( iter = 0; iter < 100; iter++ )
  {
    if ( conv->isDone() )
    {
      break;
    }

    k = conv->convert ();

    if ( k == 0 )
    {
      break;
    }

    n += k;

    if ( iter > 0 )
    {
      print ( System::out(),
              "  iteration ", iter, " (", k, " new groups)\n" );
    }
  }

  conv->finish ();

  if      ( n == 0 )
  {
    print ( System::out(), "Created zero new item groups\n\n" );
  }
  else if ( n == 1 )
  {
    print ( System::out(), "Created one new item group\n\n" );
  }
  else
  {
    print ( System::out(), "Created ", n, " new item groups\n\n" );
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> GroupconvModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return jem::newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void GroupconvModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newConverter
//-----------------------------------------------------------------------


Ref<GroupConverter> GroupconvModule::newConverter_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return nullptr;
}


JIVE_END_PACKAGE( app )
