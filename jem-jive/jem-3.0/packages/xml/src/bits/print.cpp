
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


#include <jem/xml/print.h>


JEM_BEGIN_PACKAGE( xml )


using jem::io::endl;
using jem::io::space;
using jem::io::indent;
using jem::io::outdent;
using jem::io::blankLine;


//=======================================================================
//   static data
//=======================================================================


const EndEmptyTag    endEmptyTag  = {};
const EndStartTag    endStartTag  = {};
const BeginComment   beginComment = {};
const EndComment     endComment   = {};


//=======================================================================
//   output operators
//=======================================================================

//-----------------------------------------------------------------------
//   BeginEmptyTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const BeginEmptyTag&    tag )

{
  if ( ! pr.atStartOfLine() )
  {
    print ( pr, endl );
  }

  print ( pr, '<', tag.name, indent );
}


//-----------------------------------------------------------------------
//   EndEmptyTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const EndEmptyTag&      tag )

{
  print ( pr, "/>", endl, indent );
}


//-----------------------------------------------------------------------
//   PrintEmptyTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const PrintEmptyTag&    tag )

{
  print ( pr, '<', tag.name, "/>", endl );
}


//-----------------------------------------------------------------------
//   BeginStartTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const BeginStartTag&    tag )

{
  print ( pr, blankLine, '<', tag.name, indent );
}


//-----------------------------------------------------------------------
//   EndStartTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const EndStartTag&      tag )

{
  print ( pr, '>', endl );
}


//-----------------------------------------------------------------------
//   PrintStartTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const PrintStartTag&    tag )

{
  print ( pr, blankLine, '<', tag.name, indent, '>', endl );
}


//-----------------------------------------------------------------------
//   PrintEndTag
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const PrintEndTag&      tag )

{
  if ( ! pr.atStartOfLine() )
  {
    print ( pr, endl );
  }

  print ( pr, outdent, "</", tag.name, '>', blankLine );
}


//-----------------------------------------------------------------------
//   PrintAttrib
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const PrintAttrib&      attr )

{
  print ( pr, space, attr.name );

  if ( attr.value.size() > 0 )
  {
    print ( pr, "=\"", attr.value, '\"' );
  }
}


//-----------------------------------------------------------------------
//   BeginComment
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const BeginComment&     com )

{
  print ( pr, "<!-- ", indent );
}


//-----------------------------------------------------------------------
//   EndComment
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const EndComment&       com )

{
  print ( pr, outdent, "-->" );
}


//-----------------------------------------------------------------------
//   Comment
//-----------------------------------------------------------------------


void                      print

  ( PrintWriter&            pr,
    const PrintComment&     com )

{
  print ( pr, "<!-- ", indent, com.text, outdent, " -->" );
}


JEM_END_PACKAGE( xml )
