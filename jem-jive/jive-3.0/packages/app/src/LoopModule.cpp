
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
#include <jive/util/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Names.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/LoopModule.h>


JEM_DEFINE_CLASS( jive::app::LoopModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;


//=======================================================================
//   class LoopModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LoopModule::TYPE_NAME = "Loop";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LoopModule::LoopModule

  ( const Ref<Module>&  child,
    idx_t               iter ) :

    child_ ( child )

{
  JEM_PRECHECK ( child );

  myName_    = child->getName ();
  iterCount_ = iter;
}


LoopModule::~LoopModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status LoopModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return child_->init ( conf, props, globdat );
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status LoopModule::run ( const Properties& globdat )
{
  Status  status = OK;


  if ( iterCount_ < 0 )
  {
    do
    {
      status = child_->run ( globdat );
    }
    while ( status == OK );
  }
  else
  {
    for ( idx_t i = 0; i < iterCount_ && status == OK; i++ )
    {
      status = child_->run ( globdat );
    }
  }

  return status;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void LoopModule::shutdown ( const Properties& globdat )
{
  child_->shutdown ( globdat );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LoopModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LoopModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LoopModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> LoopModule::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Properties  myConf  = conf.makeProps ( myName );
  Properties  myProps = props.getProps ( myName );

  idx_t       iter    = -1;


  myProps.find ( iter, PropNames::ITER );
  myConf .set  ( PropNames::ITER, iter );

  Ref<Self>  self =

    newInstance<Self> (
      newModule ( joinNames( myName, PropNames::MODULE ),
                  conf, props, globdat )
    );

  self->myName_ = myName;

  return self;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void LoopModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
