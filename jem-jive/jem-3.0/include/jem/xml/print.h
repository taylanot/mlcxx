
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

#ifndef JEM_XML_PRINT_H
#define JEM_XML_PRINT_H

#include <jem/io/PrintWriter.h>
#include <jem/xml/import.h>


JEM_BEGIN_PACKAGE( xml )


//-----------------------------------------------------------------------
//   class BeginEmptyTag
//-----------------------------------------------------------------------


class BeginEmptyTag
{
 public:

  explicit inline           BeginEmptyTag

    ( const String&           name );


 public:

  String                    name;

};


typedef BeginEmptyTag       beginEmptyTag;


//-----------------------------------------------------------------------
//   class EndEmptyTag
//-----------------------------------------------------------------------


struct                      EndEmptyTag {};
extern const EndEmptyTag    endEmptyTag;


//-----------------------------------------------------------------------
//   class PrintEmptyTag
//-----------------------------------------------------------------------


class PrintEmptyTag
{
 public:

  explicit inline           PrintEmptyTag

    ( const String&           name );


 public:

  String                    name;

};


typedef PrintEmptyTag       emptyTag;


//-----------------------------------------------------------------------
//   class BeginStartTag
//-----------------------------------------------------------------------


class BeginStartTag
{
 public:

  explicit inline           BeginStartTag

    ( const String&           name );


 public:

  String                    name;

};


typedef BeginStartTag       beginStartTag;


//-----------------------------------------------------------------------
//   class EndStartTag
//-----------------------------------------------------------------------


struct                      EndStartTag {};
extern const EndStartTag    endStartTag;


//-----------------------------------------------------------------------
//   class PrintStartTag
//-----------------------------------------------------------------------


class PrintStartTag
{
 public:

  explicit inline           PrintStartTag

    ( const String&           name );


 public:

  String                    name;

};


typedef PrintStartTag       startTag;


//-----------------------------------------------------------------------
//   class PrintEndTag
//-----------------------------------------------------------------------


class PrintEndTag
{
 public:

  explicit inline           PrintEndTag

    ( const String&           name );


 public:

  String                    name;

};


typedef PrintEndTag         endTag;


//-----------------------------------------------------------------------
//   class PrintAttrib
//-----------------------------------------------------------------------


class PrintAttrib
{
 public:

  explicit inline           PrintAttrib

    ( const String&           name );

  inline                    PrintAttrib

    ( const String&           name,
      const String&           value );


 public:

  String                    name;
  String                    value;

};


typedef PrintAttrib         attribute;


//-----------------------------------------------------------------------
//   class BeginComment
//-----------------------------------------------------------------------


struct                      BeginComment {};
extern const BeginComment   beginComment;


//-----------------------------------------------------------------------
//   class EndComment
//-----------------------------------------------------------------------


struct                      EndComment {};
extern const EndComment     endComment;


//-----------------------------------------------------------------------
//   class PrintComment
//-----------------------------------------------------------------------


class PrintComment
{
 public:

  explicit inline           PrintComment

    ( const String&           text );


 public:

  String                    text;

};


typedef PrintComment        comment;


//-----------------------------------------------------------------------
//   output operators
//-----------------------------------------------------------------------


void                        print

  ( PrintWriter&              pr,
    const BeginEmptyTag&      tag );

void                        print

  ( PrintWriter&              pr,
    const EndEmptyTag&        tag );

void                        print

  ( PrintWriter&              pr,
    const PrintEmptyTag&      tag );

void                        print

  ( PrintWriter&              pr,
    const BeginStartTag&      tag );

void                        print

  ( PrintWriter&              pr,
    const EndStartTag&        tag );

void                        print

  ( PrintWriter&              pr,
    const PrintStartTag&      tag );

void                        print

  ( PrintWriter&              pr,
    const PrintEndTag&        tag );

void                        print

  ( PrintWriter&              pr,
    const PrintAttrib&        attr );

void                        print

  ( PrintWriter&              pr,
    const BeginComment&       com );

void                        print

  ( PrintWriter&              pr,
    const EndComment&         com );

void                        print

  ( PrintWriter&              pr,
    const PrintComment&       com );






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   BeginEmptyTag
//-----------------------------------------------------------------------


inline BeginEmptyTag::BeginEmptyTag ( const String& nam ) :

  name ( nam )

{}


//-----------------------------------------------------------------------
//   PrintEmptyTag
//-----------------------------------------------------------------------


inline PrintEmptyTag::PrintEmptyTag ( const String& nam ) :

  name ( nam )

{}


//-----------------------------------------------------------------------
//   BeginStartTag
//-----------------------------------------------------------------------


inline BeginStartTag::BeginStartTag ( const String& nam ) :

  name ( nam )

{}


//-----------------------------------------------------------------------
//   PrintStartTag
//-----------------------------------------------------------------------


inline PrintStartTag::PrintStartTag ( const String& nam ) :

  name ( nam )

{}


//-----------------------------------------------------------------------
//   PrintEndTag
//-----------------------------------------------------------------------


inline PrintEndTag::PrintEndTag ( const String& nam ) :

  name ( nam )

{}


//-----------------------------------------------------------------------
//   PrintAttrib
//-----------------------------------------------------------------------


inline PrintAttrib::PrintAttrib ( const String& nam ) :

  name ( nam )

{}


inline PrintAttrib::PrintAttrib

  ( const String&  nam,
    const String&  val ) :

    name  ( nam ),
    value ( val )

{}


//-----------------------------------------------------------------------
//   PrintCommend
//-----------------------------------------------------------------------


inline PrintComment::PrintComment ( const String& txt ) :

  text ( txt )

{}


JEM_END_PACKAGE( xml )

#endif
