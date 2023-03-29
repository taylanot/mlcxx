
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
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/TransientModel.h>
#include <jive/implict/NewmarkStepper.h>
#include <jive/implict/NewmarkModule.h>


JEM_DEFINE_CLASS( jive::implict::NewmarkModule );


JIVE_BEGIN_PACKAGE( implict )


using jem::newInstance;


//=======================================================================
//   class NewmarkModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NewmarkModule::TYPE_NAME = "Newmark";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NewmarkModule::NewmarkModule

  ( const String&      name,
    int                order,
    Ref<SolverModule>  solver ) :

    Super ( name, solver )

{
  JEM_PRECHECK ( order >= 1 && order <= 2 );

  order_ = order;
}


NewmarkModule::~NewmarkModule ()
{}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> NewmarkModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<SolverModule>  solver =

    newSolverModule ( joinNames( name, PropNames::SOLVER ),
                      conf, props, globdat );

  Properties  myConf  = conf .makeProps ( name );
  Properties  myProps = props.findProps ( name );

  int         order   = 2;

  myProps.find ( order, PropNames::ORDER, 1, 2 );
  myConf .set  ( PropNames::ORDER, order );

  return newInstance<Self> ( name, order, solver );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void NewmarkModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newTransModel_
//-----------------------------------------------------------------------


Ref<Model> NewmarkModule::newTransModel_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  const String         context = getContext ();

  Ref<DofSpace>        dofs    =

    DofSpace::get ( globdat, context );

  Ref<NewmarkStepper>  stepper =

    newInstance<NewmarkStepper> ( myName_, order_, dofs );

  Ref<TransientModel>  tmodel  =

    newInstance<TransientModel> (
      myName_,
      Model::get ( globdat, context ),
      stepper,
      TransientModel::MANAGED
    );

  stepper->setContext ( context );
  tmodel ->setContext ( context );
  tmodel ->init       ( globdat );

  return tmodel;
}


JIVE_END_PACKAGE( implict )
