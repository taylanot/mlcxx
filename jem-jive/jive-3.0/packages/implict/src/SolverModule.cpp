
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
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/implict/Names.h>
#include <jive/implict/SolverInfo.h>
#include <jive/implict/SolverModule.h>


JEM_DEFINE_CLASS( jive::implict::SolverModule );


JIVE_BEGIN_PACKAGE( implict )


//=======================================================================
//   class SolverModule
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SolverModule::SolverModule ( const String& name ) :

  Super ( name )

{}


SolverModule::~SolverModule ()
{}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status SolverModule::run ( const Properties& globdat )
{
  using jem::System;

  Properties  info = SolverInfo::get ( globdat );

  while ( true )
  {
    info.clear ();

    advance ( globdat );

    try
    {
      solve ( info, globdat );
    }
    catch ( const jem::Exception& )
    {
      cancel ( globdat );
      throw;
    }

    if ( commit( globdat ) )
    {
      break;
    }

    print  ( System::info( myName_ ),
             "Solution rejected; re-trying\n" );

    cancel ( globdat );
  }

  return OK;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SolverModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    double      eps;

    if ( myProps.find( eps, PropNames::PRECISION, 0.0, 1.0e20 ) )
    {
      setPrecision ( eps );
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SolverModule::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  Properties  myProps = props.makeProps ( myName_ );

  myProps.set ( PropNames::PRECISION, getPrecision() );
}


JIVE_END_PACKAGE( implict )
