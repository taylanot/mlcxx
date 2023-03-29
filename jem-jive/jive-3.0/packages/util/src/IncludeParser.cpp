
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


#include <jem/io/GzipInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/mp/UniContext.h>
#include <jem/mp/BcastStream.h>
#include <jive/util/utilities.h>
#include <jive/util/IncludeParser.h>


using jem::newInstance;
using jem::io::InputStream;
using jem::io::GzipInputStream;
using jem::io::InputStreamReader;
using jem::mp::BcastStream;
using jem::mp::UniContext;


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class IncludeParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


IncludeParser::IncludeParser ( const Ref<MPContext>& mpx ) :

  mpx_ ( mpx )

{
  if ( ! mpx_ )
  {
    mpx_ = newInstance<UniContext> ();
  }
}


IncludeParser::~IncludeParser ()
{}


//-----------------------------------------------------------------------
//   openFile
//-----------------------------------------------------------------------


Ref<jem::io::Reader> IncludeParser::openFile

  ( State&         state,
    const String&  fname )

{
  Ref<InputStream>  in = BcastStream::open ( 0, mpx_, fname );


  if ( isGzipFile( fname ) )
  {
    in = newInstance<GzipInputStream> ( in );
  }

  return newInstance<InputStreamReader> ( in );
}


JIVE_END_PACKAGE( util )
