
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

#ifndef JEM_XML_PARSER_H
#define JEM_XML_PARSER_H

#include <jem/base/Object.h>
#include <jem/xml/import.h>


JEM_BEGIN_PACKAGE( xml )


class Tokenizer;
class ParseLog;
class ParserState;


//-----------------------------------------------------------------------
//   class Parser
//-----------------------------------------------------------------------


class Parser : public Object
{
 public:

  JEM_DECLARE_CLASS       ( Parser, Object );

  typedef ParserState       State;


  virtual void              reset   ();

  virtual void              parse

    ( State&                  state  ) = 0;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params );

  void                      parseInput

    ( const Ref<Tokenizer>&   input,
      const String&           source,
      const Ref<ParseLog>&    log = nullptr );

  void                      parseFile

    ( const String&           fname,
      const Ref<ParseLog>&    log = nullptr );

  static void               pushInput

    ( State&                  state,
      const Ref<Tokenizer>&   input,
      const String&           source,
      const Ref<Parser>&      parser = nullptr );

  static void               parseEndTag

    ( State&                  state,
      const String&           tagName );

  static void               parseError

    ( State&                  state   );

  static void               parseError

    ( State&                  state,
      const String&           msg     );

  static void               inputError

    ( State&                  state,
      const String&           msg     );

  static void               illegalTagError

    ( State&                  state,
      const String&           tagName );

  static void               illegalAttributeError

    ( State&                  state,
      const String&           attribName );

  static void               missingAttributeError

    ( State&                  state,
      const String&           attribName );


 protected:

  virtual                  ~Parser ();

};


JEM_END_PACKAGE( xml )

#endif
