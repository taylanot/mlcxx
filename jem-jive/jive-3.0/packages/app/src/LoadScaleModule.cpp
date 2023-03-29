
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
#include <jem/base/IllegalInputException.h>
#include <jem/util/Pattern.h>
#include <jem/util/Properties.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/LoadScaleModule.h>


JEM_DEFINE_CLASS( jive::app::LoadScaleModule );


JIVE_BEGIN_PACKAGE( app )


using jive::util::FuncUtils;


//=======================================================================
//   class LoadScaleModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LoadScaleModule::TYPE_NAME = "LoadScale";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LoadScaleModule::LoadScaleModule

  ( const String&  name,
    const String&  loadCase ) :

    Super ( name )

{
  loadCase_ = loadCase;
}


LoadScaleModule::~LoadScaleModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status LoadScaleModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status LoadScaleModule::run ( const Properties& globdat )
{
  using jive::util::Globdat;
  using jive::model::Model;
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( scaleFunc_ )
  {
    Ref<Model>  model = Model::get ( globdat, getContext() );

    Properties  params;
    double      scale = 0.0;

    try
    {
      scale = FuncUtils::evalFunc ( *scaleFunc_, globdat );
    }
    catch ( jem::Exception& ex )
    {
      ex.setContext ( getContext() );
      throw;
    }

    if ( Globdat::hasVariable( myName_, globdat ) )
    {
      Properties  vars = Globdat::getVariables ( myName_, globdat );

      vars.set ( VarNames::LOAD_SCALE, scale );
    }

    params.set ( ActionParams::SCALE_FACTOR, scale     );
    params.set ( ActionParams::LOAD_CASE,    loadCase_ );

    model->takeAction ( Actions::SET_LOAD_SCALE, params, globdat );
  }

  return OK;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LoadScaleModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::util::Pattern;
  using jem::IllegalInputException;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties  myProps = props.findProps ( myName_ );


  myProps.find ( loadCase_, PropNames::LOAD_CASE  );

  try
  {
    Pattern::check ( loadCase_ );
  }
  catch ( const IllegalInputException& ex )
  {
    myProps.propertyError (
      PropNames::LOAD_CASE,
      "invalid pattern: " + ex.what ()
    );
  }

  FuncUtils::configFunc ( scaleFunc_, PropNames::SCALE_FUNC,
                          myProps,    globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LoadScaleModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( PropNames::LOAD_CASE, loadCase_ );

  FuncUtils::getConfig ( myConf, scaleFunc_,
                         PropNames::SCALE_FUNC );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> LoadScaleModule::makeNew

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


void LoadScaleModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
