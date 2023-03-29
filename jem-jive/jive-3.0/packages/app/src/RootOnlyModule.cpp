
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
#include <jem/mp/UniContext.h>
#include <jive/util/utilities.h>
#include <jive/mp/Globdat.h>
#include <jive/app/Names.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/RootOnlyModule.h>


JEM_DEFINE_CLASS( jive::app::RootOnlyModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;


//=======================================================================
//   class RootOnlyModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  RootOnlyModule::TYPE_NAME     = "RootOnly";
const char*  RootOnlyModule::ROOT_CONTEXT_ = "temp.rootMPContext";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


RootOnlyModule::RootOnlyModule ( const Ref<Module>& child ) :

  child_ ( child )

{
  JEM_PRECHECK ( child );

  myName_ = child->getName ();
  myRank_ = -1;
}


RootOnlyModule::~RootOnlyModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status RootOnlyModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::mp::UniContext;

  Status  status = DONE;


  myRank_      = -1;
  rootContext_ = nullptr;
  origContext_ = mp::Globdat::getMPContext ( globdat );

  if ( origContext_->size() == 1 )
  {
    rootContext_ = origContext_;
  }
  else if ( ! globdat.find( rootContext_, ROOT_CONTEXT_ ) )
  {
    rootContext_ = newInstance<UniContext> ();

    globdat.set ( ROOT_CONTEXT_, rootContext_ );
  }

  myRank_ = origContext_->myRank ();

  if ( myRank_ == 0 )
  {
    if ( rootContext_ == origContext_ )
    {
      status = child_->init ( conf, props, globdat );
    }
    else
    {
      try
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, rootContext_ );

        status = child_->init ( conf, props, globdat );

        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );
      }
      catch ( ... )
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );

        throw;
      }
    }
  }

  return status;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status RootOnlyModule::run ( const Properties& globdat )
{
  Status status = DONE;


  if ( myRank_ == 0 )
  {
    if ( rootContext_ == origContext_ )
    {
      status = child_->run ( globdat );
    }
    else
    {
      try
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, rootContext_ );

        status = child_->run ( globdat );

        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );
      }
      catch ( ... )
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );

        throw;
      }
    }
  }

  return status;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void RootOnlyModule::shutdown ( const Properties& globdat )
{
  if ( myRank_ == 0 )
  {
    if ( rootContext_ == origContext_ )
    {
      child_->shutdown ( globdat );
    }
    else
    {
      try
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, rootContext_ );

        child_->shutdown ( globdat );

        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );
      }
      catch ( ... )
      {
        globdat.set ( mp::Globdat::MP_CONTEXT, origContext_ );

        throw;
      }
    }
  }

  myRank_      = -1;
  origContext_ = nullptr;
  rootContext_ = nullptr;

  globdat.set ( ROOT_CONTEXT_, rootContext_ );
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void RootOnlyModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  child_->configure ( props, globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void RootOnlyModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  child_->getConfig ( conf, globdat );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool RootOnlyModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  return child_->takeAction ( action, params, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> RootOnlyModule::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Ref<Self>  self = newInstance<Self> (
    newModule ( joinNames( myName, PropNames::MODULE ),
                conf, props, globdat )
  );

  self->myName_ = myName;

  return self;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void RootOnlyModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
