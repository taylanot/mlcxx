
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
#include <jem/base/Thread.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/Exception.h>
#include <jem/base/CancelledException.h>
#include <jem/util/Properties.h>
#include <jive/Array.h>
#include <jive/util/utilities.h>
#include <jive/app/Names.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/ChainModule.h>


JEM_DEFINE_CLASS( jive::app::ChainModule );


JIVE_BEGIN_PACKAGE( app )


using jem::staticCast;
using jem::newInstance;
using jem::util::ObjFlex;


//=======================================================================
//   class ChainModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ChainModule::TYPE_NAME = "Chain";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ChainModule::ChainModule ( const String& name ) :

  Super ( name )

{}


ChainModule::~ChainModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status ChainModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<Module>  mod;
  Status       stat;


  runList_  .clear   ();
  runList_  .reserve ( modules_.size() );
  aliveList_.clear   ();
  aliveList_.reserve ( modules_.size() );

  try
  {
    for ( idx_t i = 0; i < modules_.size(); i++ )
    {
      mod  = modules_.getAs<Module> ( i );
      stat = mod->init ( conf, props, globdat );

      aliveList_.pushBack ( mod );

      if      ( stat == EXIT )
      {
        return EXIT;
      }
      else if ( stat != DONE )
      {
        runList_.pushBack ( mod );
      }
    }
  }
  catch ( const jem::Exception& )
  {
    for ( idx_t i = 0; i < aliveList_.size(); i++ )
    {
      mod = aliveList_.getAs<Module> ( i );
      mod->shutdown ( globdat );
    }

    runList_  .clear ();
    aliveList_.clear ();

    throw;
  }

  return runList_.size() ? OK : DONE;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status ChainModule::run ( const Properties&  globdat )
{
  Ref<Module>  mod;
  Status       stat;


  if ( jem::Thread::cancelled() )
  {
    throw jem::CancelledException (
      getContext (),
      "program execution cancelled"
    );
  }

  if ( ! runList_.size() )
  {
    return DONE;
  }

  for ( idx_t i = 0; i < runList_.size(); i++ )
  {
    mod  = runList_.getAs<Module> ( i );
    stat = mod->run ( globdat );

    if      ( stat == EXIT )
    {
      return EXIT;
    }
    else if ( stat == DONE )
    {
      runList_.erase ( i-- );
    }
  }

  return runList_.size() ? OK : DONE;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void  ChainModule::shutdown ( const Properties&  globdat )
{
  ObjFlex  list;

  list.swap ( aliveList_ );

  runList_.clear ();

  for ( idx_t i = 0; i < list.size(); i++ )
  {
    Ref<Module>  mod = list.getAs<Module> ( i );

    mod->shutdown ( globdat );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ChainModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  for ( idx_t i = 0; i < modules_.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    mod->configure ( props, globdat );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ChainModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  for ( idx_t i = 0; i < modules_.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    mod->getConfig ( conf, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ChainModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = false;

  for ( idx_t i = 0; i < modules_.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    if ( mod->takeAction( action, params, globdat ) )
    {
      result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ChainModule::clear ()
{
  modules_.clear ();
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ChainModule::pushBack ( const Ref<Module>& mod )
{
  JEM_PRECHECK ( mod );

  modules_.pushBack ( mod );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


bool ChainModule::erase ( const Ref<Module>& mod )
{
  return (modules_.erase( mod ) != 0);
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> ChainModule::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;

  Properties        myConf  = conf.makeProps ( myName );
  Properties        myProps = props.getProps ( myName );

  Ref<ChainModule>  chain;
  StringVector      names;
  Ref<Module>       mod;
  idx_t             i, n;


  myProps.get ( names, PropNames::MODULES );
  myConf .set ( PropNames::MODULES, names );

  chain = newInstance<ChainModule> ( myName );
  n     = names.size ();

  for ( i = 0; i < n; i++ )
  {
    if ( ! names[i].size() )
    {
      myProps.propertyError ( PropNames::MODULES,
                              "empty module name" );
    }

    mod = newModule ( joinNames( myName, names[i] ),
                      conf, props, globdat );

    chain->pushBack ( mod );
  }

  return chain;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ChainModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
