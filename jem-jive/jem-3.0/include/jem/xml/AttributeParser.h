
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

#ifndef JEM_XML_ATTRIBUTEPARSER_H
#define JEM_XML_ATTRIBUTEPARSER_H

#include <jem/base/String.h>


JEM_BEGIN_PACKAGE( xml )


class ParserState;


//-----------------------------------------------------------------------
//   class AttributeParser
//-----------------------------------------------------------------------


class AttributeParser
{
 public:

  enum                  Kind
  {
                          OPTIONAL,
                          REQUIRED
  };


                        AttributeParser ();

  explicit              AttributeParser

    ( const String&       attribs );

  virtual              ~AttributeParser ();

  void                  clear           ();

  idx_t                 declareAttribute

    ( const String&       name,
      Kind                kind );

  idx_t                 declareAttributes

    ( const String&       attribs );

  int                   parseAttributes

    ( ParserState&        state,
      bool                lenient = false );

  virtual void          handleAttribute

    ( ParserState&        state,
      const String&       name,
      const String&       value,
      idx_t               index ) = 0;

  static bool           parseBoolean

    ( ParserState&        state,
      const String&       name,
      const String&       value );

  static lint           parseInteger

    ( ParserState&        state,
      const String&       name,
      const String&       value );

  static lint           parseInteger

    ( ParserState&        state,
      const String&       name,
      const String&       value,
      lint                min,
      lint                max );

  static double         parseFloat

    ( ParserState&        state,
      const String&       name,
      const String&       value );

  static double         parseFloat

    ( ParserState&        state,
      const String&       name,
      const String&       value,
      double              min,
      double              max );


 private:

                        AttributeParser ( const AttributeParser& );
  AttributeParser&      operator =      ( const AttributeParser& );


 private:

  class                 Attrib_;

  Attrib_*              list_;

};


JEM_END_PACKAGE( xml )

#endif
