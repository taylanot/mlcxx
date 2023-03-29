
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
#include <jive/gl/Names.h>
#include <jive/gl/Viewdat.h>
#include <jive/gl/PluginFactory.h>
#include <jive/gl/MarkerStyleTable.h>
#include <jive/gl/MarkerManager.h>


JEM_DEFINE_CLASS( jive::gl::MarkerManager );


JIVE_BEGIN_PACKAGE( gl )


//=======================================================================
//   class MarkerManager
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  MarkerManager::SELF_COUNT_ = "markerManagerCount";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


MarkerManager::MarkerManager ( const String& name ) :

  Super ( name )

{}


MarkerManager::~MarkerManager ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void MarkerManager::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;

  Ref<MarkerStyleTable>  styles;

  idx_t                  count = 0;


  styles_ = nullptr;

  viewdat.find ( count, SELF_COUNT_ );

  count++;

  if ( count > 1 )
  {
    String  context = getContext ();

    print ( System::warn(), context,
            " : another marker style manager is already active;\n" );

    print ( System::warn(), context,
            " : disabling this marker style manager\n\n" );
  }
  else
  {
    Properties    myConf  = conf .makeProps ( myName_ );
    Properties    myProps = props.findProps ( myName_ );

    StringVector  names;


    styles = MarkerStyleTable::get ( viewdat );

    styles->configure ( myProps );
    styles->getConfig ( myConf  );

    if ( myProps.find( names, PropNames::STYLES ) )
    {
      styles->addStyles ( names, myConf, myProps );
    }

    myConf.set ( PropNames::STYLES, names );

    styles_ = styles;
  }

  viewdat.set ( SELF_COUNT_, count );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void MarkerManager::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void MarkerManager::shutdown

  ( const Properties&  viewdat,
    const AppContext&  apx )

{
  idx_t  count = 1;

  if ( viewdat.find( count, SELF_COUNT_ ) )
  {
    count--;

    if ( count <= 0 )
    {
      viewdat.erase ( SELF_COUNT_ );
    }
    else
    {
      viewdat.set   ( SELF_COUNT_, count );
    }
  }

  styles_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void MarkerManager::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( props.contains( myName_ ) && styles_ )
  {
    Properties    myProps = props.findProps ( myName_ );

    StringVector  names;

    styles_->configure ( myProps );

    if ( myProps.find( names, PropNames::STYLES ) )
    {
      Properties  myConf;

      styles_->clear     ();
      styles_->addStyles ( names, myConf, myProps );
    }
    else
    {
      styles_->configure ( myProps );
    }

    if ( myProps.find( names, PropNames::NEW_STYLES ) )
    {
      Properties  myConf;

      styles_->addStyles ( names, myConf, myProps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void MarkerManager::getConfig ( const Properties& props ) const
{
  if ( styles_ )
  {
    styles_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void MarkerManager::declare ()
{
  PluginFactory::declare ( "MarkerManager", & makeNew );
  PluginFactory::declare ( CLASS_NAME,      & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> MarkerManager::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return jem::newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
