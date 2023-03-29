
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <cstdlib>
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/PrintWriter.h>
#include <jem/gl/PrintErrHandler.h>


JEM_DEFINE_CLASS( jem::gl::PrintErrHandler );


JEM_BEGIN_PACKAGE( gl )


using jem::io::Writer;
using jem::io::PrintWriter;


//=======================================================================
//   class PrintErrHandler
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


PrintErrHandler::PrintErrHandler ( const String& context ) :

  context_ ( context )

{}


PrintErrHandler::~PrintErrHandler ()
{}


//-----------------------------------------------------------------------
//   error
//-----------------------------------------------------------------------


void PrintErrHandler::error ()
{
  Writer&  warn = System::warn ();

  if ( context_.size() )
  {
    print ( warn, context_, " : caught an unknown exception;\n" );
    print ( warn, context_, " : trying to continue\n\n" );
  }
  else
  {
    print ( warn, "Caught an unknown exception; "
            "trying to continue\n\n" );
  }
}


void PrintErrHandler::error ( const Throwable& t )
{
  using jem::io::indent;
  using jem::io::outdent;
  using jem::io::blankLine;

  const Exception*  ex = dynamic_cast<const Exception*> ( &t );

  Ref<PrintWriter>  prn;
  Ref<Writer>       out;

  if ( ex )
  {
    out = & System::warn ();
  }
  else
  {
    out = & System::err  ();
  }

  prn = newInstance<PrintWriter> ( out );

  prn->setPageWidth ( 80 );

  if ( context_.size() )
  {
    print ( *prn, context_, ", " );
  }

  print ( *prn, t.where(), " : ", indent,
          t.what(), outdent, blankLine );

  if ( ! ex )
  {
    String  trace = t.getStackTrace ();

    if ( trace.size() )
    {
      print ( *prn, "Stack trace:\n\n",
              indent, trace, outdent, blankLine );
    }

    if ( context_.size() )
    {
      print ( *prn, context_, " : terminating program.\n\n" );
    }
    else
    {
      print ( *prn, "Terminating program.\n\n" );
    }

    prn->flush ();

    ::exit ( 1 );
  }
  else
  {
    prn->flush ();
  }
}


JEM_END_PACKAGE( gl )
