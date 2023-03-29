
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
#include <jive/gl/PluginFactory.h>
#include <jive/gl/PaletteTable.h>
#include <jive/gl/PaletteManager.h>


JEM_DEFINE_CLASS( jive::gl::PaletteManager );


JIVE_BEGIN_PACKAGE( gl )


//=======================================================================
//   class PaletteManager
//=======================================================================


const char*  PaletteManager::SELF_COUNT_ = "paletteManagerCount";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PaletteManager::PaletteManager ( const String& name ) :

  Super ( name )

{}


PaletteManager::~PaletteManager ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void PaletteManager::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  using jem::System;

  Ref<PaletteTable>  palettes;

  idx_t              count = 0;


  palettes_ = nullptr;

  viewdat.find ( count, SELF_COUNT_ );

  count++;

  if ( count > 1 )
  {
    String  context = getContext ();

    print ( System::warn(), context,
            " : another palette manager is already active;\n" );

    print ( System::warn(), context,
            " : disabling this palette manager\n\n" );
  }
  else
  {
    Properties    myConf  = conf .makeProps ( myName_ );
    Properties    myProps = props.findProps ( myName_ );

    StringVector  names;


    palettes = PaletteTable::get ( viewdat );

    palettes->configure ( myProps );
    palettes->getConfig ( myConf  );

    if ( myProps.find( names, PropNames::PALETTES ) )
    {
      palettes->addPalettes ( names, myConf, myProps );
    }

    myConf.set ( PropNames::PALETTES, names );

    palettes_ = palettes;
  }

  viewdat.set ( SELF_COUNT_, count );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


void PaletteManager::update

  ( const Properties&  viewdat,
    const AppContext&  apx )

{}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void PaletteManager::shutdown

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

  palettes_ = nullptr;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void PaletteManager::configure

  ( const Properties&  props,
    const Properties&  viewdat,
    const AppContext&  apx )

{
  if ( props.contains( myName_ ) && palettes_ )
  {
    Properties    myProps = props.findProps ( myName_ );

    StringVector  names;

    if ( myProps.find( names, PropNames::PALETTES ) )
    {
      Properties  myConf;

      palettes_->clear       ();
      palettes_->addPalettes ( names, myConf, myProps );
    }
    else
    {
      palettes_->configure   ( myProps );
    }

    if ( myProps.find( names, PropNames::NEW_PALETTES ) )
    {
      Properties  myConf;

      palettes_->addPalettes ( names, myConf, myProps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PaletteManager::getConfig ( const Properties& props ) const
{
  if ( palettes_ )
  {
    palettes_->getConfig ( props.makeProps( myName_ ) );
  }
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void PaletteManager::declare ()
{
  PluginFactory::declare ( "PaletteManager", & makeNew );
  PluginFactory::declare ( CLASS_NAME,       & makeNew );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Plugin> PaletteManager::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props )

{
  return jem::newInstance<Self> ( name );
}


JIVE_END_PACKAGE( gl )
