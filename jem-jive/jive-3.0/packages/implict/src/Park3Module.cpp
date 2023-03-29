
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
#include <jive/util/utilities.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/Park3Model.h>
#include <jive/implict/Park3Module.h>


JEM_DEFINE_CLASS( jive::implict::Park3Module );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;


//=======================================================================
//   class Park3Module
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Park3Module::TYPE_NAME = "Park3";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Park3Module::Park3Module

  ( const String&      name,
    Ref<SolverModule>  solver ) :

    Super ( name, solver )

{}


Park3Module::~Park3Module ()
{}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> Park3Module::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<SolverModule>  solver =

    newSolverModule ( joinNames( name, PropNames::SOLVER ),
                      conf, props, globdat );

  return newInstance<Self> ( name, solver );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void Park3Module::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newTransModel_
//-----------------------------------------------------------------------


Ref<Model> Park3Module::newTransModel_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  const String     context = getContext ();

  Ref<Park3Model>  p3model =

    newInstance<Park3Model> (
      myName_,
      Model::get ( globdat, context ),
      Park3Model::MANAGED
    );

  p3model->setContext ( context );
  p3model->init       ( globdat );

  return p3model;
}


JIVE_END_PACKAGE( implict )
