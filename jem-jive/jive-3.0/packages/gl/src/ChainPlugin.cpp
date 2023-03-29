
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
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/Array.h>
#include <jive/util/utilities.h>
#include <jive/gl/Names.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/ChainPlugin.h>


JEM_DEFINE_CLASS( jive::gl::ChainPlugin );


JIVE_BEGIN_PACKAGE( gl )


using jem::util::ObjFlex;


//=======================================================================
//   class ChainPlugin
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ChainPlugin::TYPE_NAME = "Chain";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ChainPlugin::ChainPlugin ( const String& name ) :

  Super ( name )

{}


ChainPlugin::ChainPlugin

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  using jive::util::joinNames;

  Properties    myConf  = conf .makeProps ( myName_ );
  Properties    myProps = props.findProps ( myName_ );

  StringVector  names;


  myProps.find ( names, PropNames::PLUGINS );
  myConf .set  ( PropNames::PLUGINS, names );

  plugins_.reserve ( names.size() );

  for ( idx_t i = 0; i < names.size(); i++ )
  {
    Ref<Plugin>  plugin =

      PluginFactory::newInstance (
        joinNames ( myName_, names[i] ),
        conf, props
      );

    plugins_.pushBack ( plugin );
  }
}


ChainPlugin::~ChainPlugin ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void ChainPlugin::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  Ref<Plugin>  plugin;


  aliveList_.clear   ();
  aliveList_.reserve ( plugins_.size() );

  try
  {
    for ( idx_t i = 0; i < plugins_.size(); i++ )
    {
      plugin = plugins_.getAs<Plugin> ( i );

      plugin->init ( conf, props, viewdat, apx );

      aliveList_.pushBack ( plugin );
    }
  }
  catch ( const jem::Exception& )
  {
    for ( idx_t i = 0; i < aliveList_.size(); i++ )
    {
      plugin = aliveList_.getAs<Plugin> ( i );
      plugin->shutdown ( viewdat, apx );
    }

    aliveList_.clear ();

    throw;
  }
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void ChainPlugin::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;

  for ( idx_t i = 0; i < aliveList_.size(); i++ )
  {
    Ref<Plugin>  plugin = aliveList_.getAs<Plugin> ( i );

    try
    {
      plugin->update ( viewdat, apx );
    }
    catch ( const jem::IllegalInputException& ex )
    {
      print ( System::warn(), ex, '\n' );
    }
  }
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void ChainPlugin::shutdown

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  ObjFlex  list;

  list.swap ( aliveList_ );

  for ( idx_t i = 0; i < list.size(); i++ )
  {
    Ref<Plugin>  plugin = list.getAs<Plugin> ( i );

    plugin->shutdown ( viewdat, apx );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void ChainPlugin::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  for ( idx_t i = 0; i < plugins_.size(); i++ )
  {
    Ref<Plugin>  plugin = plugins_.getAs<Plugin> ( i );

    plugin->configure ( props, viewdat, apx );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ChainPlugin::getConfig ( const Properties& props ) const
{
  for ( idx_t  i = 0; i < plugins_.size(); i++ )
  {
    Ref<Plugin>  plugin = plugins_.getAs<Plugin> ( i );

    plugin->getConfig ( props );
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void ChainPlugin::printHelp ( PrintWriter& pr ) const
{
  for ( idx_t i = 0; i < plugins_.size(); i++ )
  {
    Ref<Plugin>  plugin = plugins_.getAs<Plugin> ( i );

    plugin->printHelp ( pr );
  }
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ChainPlugin::clear ()
{
  plugins_.clear ();
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


void ChainPlugin::pushBack ( const Ref<Plugin>&  plugin )
{
  JEM_PRECHECK ( plugin );

  plugins_.pushBack ( plugin );
}


//-----------------------------------------------------------------------
//   erase
//-----------------------------------------------------------------------


bool ChainPlugin::erase ( const Ref<Plugin>& plugin )
{
  return (plugins_.erase( plugin ) != 0);
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void ChainPlugin::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> ChainPlugin::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return jem::newInstance<Self> ( name, conf, props );
}


JIVE_END_PACKAGE( gl )
