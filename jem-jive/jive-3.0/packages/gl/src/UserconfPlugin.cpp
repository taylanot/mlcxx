
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
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jive/Array.h>
#include <jive/util/utilities.h>
#include <jive/gl/Names.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/UserconfPlugin.h>


JEM_DEFINE_CLASS( jive::gl::UserconfPlugin );


JIVE_BEGIN_PACKAGE( gl )


using jem::util::ObjFlex;


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  UserconfPlugin::TYPE_NAME = "Userconf";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


UserconfPlugin::UserconfPlugin ( const String& name ) :

  Super ( name )

{
  running_ = false;
}


UserconfPlugin::~UserconfPlugin ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void UserconfPlugin::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  Ref<Plugin>  plugin;


  running_ = false;

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

  running_ = true;
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void UserconfPlugin::update

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


void UserconfPlugin::shutdown

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


void UserconfPlugin::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::util::isNone;

  Properties    myProps = props.findProps ( myName_ );

  StringVector  names;
  StringVector  names2;

  Ref<Plugin>   plugin;

  idx_t         n;


  // Erase all or selected plugins.

  if ( myProps.find( names, PropNames::PLUGINS ) )
  {
    for ( idx_t i = 0; i < aliveList_.size(); i++ )
    {
      plugin = aliveList_.getAs<Plugin> ( i );
      plugin->shutdown ( viewdat, apx );
    }

    plugins_  .clear ();
    aliveList_.clear ();
  }
  else
  {
    Ref<Object>  obj;

    for ( idx_t i = 0; i < plugins_.size(); i++ )
    {
      plugin = plugins_.getAs<Plugin> ( i );

      if ( ! props.find( obj, plugin->getName() ) )
      {
        continue;
      }

      if ( isNone( obj ) )
      {
        plugins_.erase ( i );

        if ( aliveList_.erase( plugin.get() ) )
        {
          plugin->shutdown ( viewdat, apx );
        }

        i--;
      }
    }
  }

  // Configure the current plugins.

  for ( idx_t i = 0; i < plugins_.size(); i++ )
  {
    plugin = plugins_.getAs<Plugin> ( i );
    plugin->configure ( props, viewdat, apx );
  }

  if ( myProps.find( names2, PropNames::NEW_PLUGINS ) )
  {
    n = names.size ();

    if ( n )
    {
      names.reshape ( n + names2.size() );
      names[slice(n,END)] = names2;
    }
    else
    {
      names.ref ( names2 );
    }
  }

  // Add new plugins.

  n = names.size ();

  if ( n )
  {
    Properties  conf;

    plugins_.reserve ( plugins_.size() + n );

    for ( idx_t i = 0; i < n; i++ )
    {
      plugin  = newPlugin_ ( names[i], conf, props, viewdat, apx );

      plugin->configure ( props, viewdat, apx );

      plugins_.pushBack ( plugin );

      if ( running_ )
      {
        plugin->init ( conf, props, viewdat, apx );

        aliveList_.pushBack ( plugin );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void UserconfPlugin::getConfig ( const Properties& props ) const
{
  for ( idx_t i = 0; i < plugins_.size(); i++ )
  {
    Ref<Plugin>  plugin = plugins_.getAs<Plugin> ( i );

    plugin->getConfig ( props );
  }
}


//-----------------------------------------------------------------------
//   printHelp
//-----------------------------------------------------------------------


void UserconfPlugin::printHelp ( PrintWriter& pr ) const
{
  for ( idx_t i = 0; i < plugins_.size(); i++ )
  {
    Ref<Plugin>  plugin = plugins_.getAs<Plugin> ( i );

    plugin->printHelp ( pr );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void UserconfPlugin::declare ()
{
  PluginFactory::declare ( TYPE_NAME,  & makeNew );
  PluginFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> UserconfPlugin::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return jem::newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   newPlugin_
//-----------------------------------------------------------------------


Ref<Plugin> UserconfPlugin::newPlugin_

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx ) const

{
  using jive::util::joinNames;

  Ref<Plugin>  p =

    PluginFactory::newInstance ( joinNames ( myName_, name ),
                                 conf, props );

  return p;
}


JIVE_END_PACKAGE( gl )
