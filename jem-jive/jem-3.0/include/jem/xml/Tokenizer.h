
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

#ifndef JEM_XML_TOKENIZER_H
#define JEM_XML_TOKENIZER_H

#include <jem/base/Flags.h>
#include <jem/base/Object.h>
#include <jem/xml/import.h>
#include <jem/xml/Scanner.h>


JEM_BEGIN_PACKAGE( xml )


//-----------------------------------------------------------------------
//   class Tokenizer
//-----------------------------------------------------------------------


class Tokenizer : public Object
{
 public:

  typedef Tokenizer       Self;
  typedef Object          Super;


  static const int        NULL_TOKEN;
  static const int        EOF_TOKEN;
  static const int        COMMENT_TOKEN;
  static const int        START_TAG_TOKEN;
  static const int        TAG_END_TOKEN;
  static const int        EMPTY_TAG_END_TOKEN;
  static const int        END_TAG_TOKEN;
  static const int        SYMBOL_TOKEN;
  static const int        STRING_TOKEN;
  static const int        BOOLEAN_TOKEN;
  static const int        INTEGER_TOKEN;
  static const int        FLOAT_TOKEN;

  enum                    Option
  {
                            PARSE_COMMENTS       = 1 << 0,
                            PARSE_SYMBOLS        = 1 << 1,
                            PARSE_STRINGS        = 1 << 2,
                            PARSE_BOOLEANS       = 1 << 3,
                            PARSE_NUMBERS        = 1 << 4,
                            PARSE_SIGNED_NUMBERS = 1 << 5
  };

  typedef Flags<Option>   Options;


  explicit                Tokenizer

    ( const String&         in = String() );

  explicit                Tokenizer

    ( const Ref<Reader>&    in,
      idx_t                 bufsize = -1 );

  void                    setInput

    ( const String&         str );

  int                     nextToken       ();
  void                    pushBackToken   ();
  int                     getToken        () const;
  lint                    getLineNumber   () const noexcept;

  void                    setLineNumber

    ( lint                  lineno );

  String                  getTokenString  () const;
  String                  getComment      () const;
  String                  getTagName      () const;
  String                  getSymbol       () const;
  String                  getString       () const;
  bool                    getBoolean      () const;
  lint                    getInteger      () const;
  double                  getFloat        () const;

  void                    setOption

    ( Option                option,
      bool                  yesno = true );

  void                    setOptions

    ( Options               options );

  Options                 getOptions      () const noexcept;

  bool                    readAttribute

    ( String&               name,
      String&               value );

  int                     skipAttributes  ();

  int                     readAttributes

    ( TextOutput&           out );

  void                    skipToEndTag

    ( const String&         tagName );

  String                  readToEndTag

    ( const String&         tagName );

  void                    readToEndTag

    ( TextOutput&           out,
      const String&         tagName );

  String                  readText        ();

  int                     readText

    ( TextOutput&           out );


 protected:

  virtual                ~Tokenizer       ();


 private:

  void                    init_           ();

  int                     readTag_        ();

  void                    endTagError_

    ( const String&         tagName )        const;


 private:

  Scanner                 scanner_;

  int                     token_;
  int                     pushedToken_;
  Options                 options_;

};


JEM_DEFINE_FLAG_OPS( Tokenizer::Options )


JEM_END_PACKAGE( xml )

#endif
