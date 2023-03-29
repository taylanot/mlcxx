
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

#ifndef JEM_UTIL_PROPS_PARSER_H
#define JEM_UTIL_PROPS_PARSER_H

#include <jem/base/StringBuffer.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>


JEM_BEGIN_PACKAGE( util )

class Tokenizer;

JEM_BEGIN_NAMESPACE( props )


//-----------------------------------------------------------------------
//   class Parser
//-----------------------------------------------------------------------


class Parser
{
 public:

  typedef Parser          Self;
  typedef Ref<Object>     ObjRef;
  typedef Properties
    ::ParseOptions        Options;

  static const int        NONE_TOKEN;
  static const int        INCLUDE_TOKEN;
  static const int        SINCLUDE_TOKEN;

  static const int        ASSIGN_PROPERTY;
  static const int        EXTEND_PROPERTY;

  static const int        MAX_INCLUDE_DEPTH;


                          Parser            ();

                         ~Parser            ();

  void                    parse

    ( const Properties&     props,
      Tokenizer&            input,
      const String&         source,
      Options               options );

  inline String           getSource         () const;
  inline lint             getLineNumber     () const;


 private:

                          Parser

    ( const Self&           rhs );

  Self&                   operator =

    ( const Self&           rhs );


 private:

  void                    parse_            ();

  String                  parseName_

    ( int                   token );

  void                    parseProperty_

    ( const String&         name,
      int                   mode );

  ObjRef                  parseArray_       ();

  ObjRef                  parseStringArray_ ();

  void                    parseInclude_

    ( int                   token );

  ObjRef                  getProperty_

    ( const String&         name );

  bool                    findProperty_

    ( ObjRef&               obj,
      const String&         name );

  Ref<Dict>               findProperties_

    ( const String&         name );

  void                    setProperty_

    ( const String&         name,
      const ObjRef&         value );

  void                    setProperty_

    ( Properties&           props,
      const String&         name,
      const ObjRef&         value );

  Ref<Dict>               newProperties_    ();

  static bool             checkName_

    ( const String&         name );

  void                    expandString_

    ( String&               str );

  String                  expandBuiltin_

    ( const String&         str )              const;

  void                    expansionError_

    ( const String&         str )              const;

  ObjRef                  extendProperty_

    ( const String&         name,
      const ObjRef&         rhs );

  template <class T>

    ObjRef                extendArray_

    ( const String&         name,
      const Array<T>&       lhs,
      const ObjRef&         rhs );

  inline int              nextToken_        ();
  int                     symbolToken_      ();


 private:

  struct                  InputFrame_
  {
    Ref<Tokenizer>          input;
    String                  source;
  };


  Flex<Properties>        propsStack_;
  Flex<InputFrame_>       inputStack_;
  Ref<Tokenizer>          input_;
  String                  source_;
  StringBuffer            strbuf_;
  Options                 options_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getSource
//-----------------------------------------------------------------------


inline String Parser::getSource () const
{
  return source_;
}


//-----------------------------------------------------------------------
//   getLineNumber
//-----------------------------------------------------------------------


inline lint Parser::getLineNumber () const
{
  if ( input_ )
  {
    return input_->getLineNumber ();
  }
  else
  {
    return -1_lint;
  }
}


JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )

#endif
