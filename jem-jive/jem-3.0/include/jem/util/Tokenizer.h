
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

#ifndef JEM_UTIL_TOKENIZER_H
#define JEM_UTIL_TOKENIZER_H

#include <jem/base/Object.h>
#include <jem/util/CppScanner.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Tokenizer
//-----------------------------------------------------------------------


class Tokenizer : public Object
{
 public:

  typedef Tokenizer       Self;
  typedef Object          Super;
  typedef io::Reader      Reader;


  static const int        NULL_TOKEN;
  static const int        EOF_TOKEN;
  static const int        EOL_TOKEN;
  static const int        COMMENT_TOKEN;
  static const int        SYMBOL_TOKEN;
  static const int        STRING_TOKEN;
  static const int        BOOLEAN_TOKEN;
  static const int        INTEGER_TOKEN;
  static const int        FLOAT_TOKEN;

  static const int        PLUS_TOKEN;
  static const int        MINUS_TOKEN;
  static const int        MULT_TOKEN;
  static const int        DIV_TOKEN;
  static const int        MOD_TOKEN;
  static const int        BOR_TOKEN;
  static const int        BAND_TOKEN;
  static const int        BNOT_TOKEN;
  static const int        BXOR_TOKEN;
  static const int        LSHIFT_TOKEN;
  static const int        RSHIFT_TOKEN;
  static const int        PLUS_IS_TOKEN;
  static const int        MINUS_IS_TOKEN;
  static const int        MULT_IS_TOKEN;
  static const int        DIV_IS_TOKEN;
  static const int        MOD_IS_TOKEN;
  static const int        BOR_IS_TOKEN;
  static const int        BAND_IS_TOKEN;
  static const int        BNOT_IS_TOKEN;
  static const int        BXOR_IS_TOKEN;
  static const int        LSHIFT_IS_TOKEN;
  static const int        RSHIFT_IS_TOKEN;
  static const int        LOR_TOKEN;
  static const int        LAND_TOKEN;
  static const int        LNOT_TOKEN;
  static const int        ASSIGN_TOKEN;
  static const int        EQUAL_TOKEN;
  static const int        UNEQUAL_TOKEN;
  static const int        LESS_THAN_TOKEN;
  static const int        LESS_EQUAL_TOKEN;
  static const int        GREATER_THAN_TOKEN;
  static const int        GREATER_EQUAL_TOKEN;
  static const int        COMMA_TOKEN;
  static const int        DOT_TOKEN;
  static const int        COLON_TOKEN;
  static const int        DOUBLE_COLON_TOKEN;
  static const int        SEMI_COLON_TOKEN;
  static const int        OPEN_PAREN_TOKEN;
  static const int        CLOSE_PAREN_TOKEN;
  static const int        OPEN_BRACE_TOKEN;
  static const int        CLOSE_BRACE_TOKEN;
  static const int        OPEN_BRACKET_TOKEN;
  static const int        CLOSE_BRACKET_TOKEN;

  enum                    Option
  {
                            PARSE_COMMENTS       = 1 << 0,
                            PARSE_SYMBOLS        = 1 << 1,
                            PARSE_STRINGS        = 1 << 2,
                            PARSE_BOOLEANS       = 1 << 3,
                            PARSE_NUMBERS        = 1 << 4,
                            PARSE_SIGNED_NUMBERS = 1 << 5,
                            PARSE_OPERATORS      = 1 << 6,
                            PARSE_END_OF_LINE    = 1 << 7,
                            CATENATE_STRINGS     = 1 << 8
  };

  typedef Flags<Option>   Options;


  explicit                Tokenizer

    ( const String&         in      = "" );

  explicit                Tokenizer

    ( const Ref<Reader>&    in,
      idx_t                 bufsize = -1 );

  void                    setInput

    ( const String&         str );

  int                     nextToken       ();
  void                    pushBackToken   ();
  int                     getToken        () const;
  void                    skipLine        ();
  lint                    getLineNumber   () const;

  void                    setLineNumber

    ( lint                  lineno );

  String                  getTokenString  () const;

  const char*             getTokenChars

    ( idx_t&                len )            const;

  String                  getComment      () const;
  String                  getSymbol       () const;
  String                  getString       () const;
  bool                    getBoolean      () const;
  lint                    getInteger      () const;
  double                  getFloat        () const;
  int                     getStringType   () const;

  void                    setOption

    ( Option                option,
      bool                  yesno = true );

  void                    setOptions

    ( Options               options );

  Options                 getOptions      () const noexcept;

  void                    parseError      () const;

  void                    parseError

    ( const String&         msg )            const;

  void                    rangeError

    ( lint                  value,
      lint                  lbound,
      lint                  ubound )         const;

  void                    rangeError

    ( double                value,
      double                ubound )         const;


 protected:

  virtual                ~Tokenizer       ();


 private:

  void                    init_           ();

  int                     parseOperator_

    ( int                   c );

  int                     parseOperator2_

    ( int                   tok,
      int                   alt );


 private:

  CppScanner              scanner_;

  int                     token_;
  int                     pushedToken_;
  Options                 options_;

};


JEM_DEFINE_FLAG_OPS( Tokenizer::Options )


JEM_END_PACKAGE( util )

#endif
