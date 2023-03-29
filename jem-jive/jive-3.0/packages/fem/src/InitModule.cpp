
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
#include <jive/util/XDofSpace.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/utilities.h>
#include <jive/fem/Names.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/InitModule.h>


JEM_DEFINE_CLASS( jive::fem::InitModule );


JIVE_BEGIN_PACKAGE( fem )


//=======================================================================
//   class InitModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  InitModule::TYPE_NAME = "FEInit";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


InitModule::InitModule ( const String& name ) :

  Super ( name )

{}


InitModule::~InitModule ()
{}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> InitModule::makeNew

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


void InitModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   reorderDofs_
//-----------------------------------------------------------------------


void InitModule::reorderDofs_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::System;


  Properties  myConf  = conf. makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );

  bool        reorder = false;


  myProps.find ( reorder, PropNames::REORDER );
  myConf .set  ( PropNames::REORDER, reorder );

  if ( reorder )
  {
    ElementSet elems = ElementSet::find ( globdat );

    if ( elems )
    {
      NodeSet         nodes = elems.getNodes ();

      Ref<XDofSpace>  dofs  =

        jem::dynamicCast<XDofSpace> (
          DofSpace::find ( nodes.getData(), globdat )
        );

      if ( dofs )
      {
        print ( System::out(), "Reordering dofs ...\n\n" );

        reorderDofs ( *dofs, elems );
      }
    }
  }
}


JIVE_END_PACKAGE( fem )
