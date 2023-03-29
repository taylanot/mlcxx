
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

#ifndef JIVE_UTIL_FUNCSPARSER_H
#define JIVE_UTIL_FUNCSPARSER_H

#include <jem/xml/TagParser.h>
#include <jive/util/import.h>
#include <jive/util/typedefs.h>


JIVE_BEGIN_PACKAGE( util )


class FuncParser;
class ParserScope;


//-----------------------------------------------------------------------
//   class FuncsParser
//-----------------------------------------------------------------------


class FuncsParser : public jem::xml::TagParser
{
 public:

  JEM_DECLARE_CLASS       ( FuncsParser, jem::xml::TagParser );


  explicit                  FuncsParser

    ( const Ref<Dictionary>&  funcs  = nullptr,
      const String&           filter =      "*" );

  virtual void              reset      () override;

  virtual void              parseContents

    ( State&                  state     ) override;

  virtual void              handleEmptyContents

    ( State&                  state     ) override;

  virtual void              finish

    ( State&                  state     ) override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params    ) override;

  void                      setFuncs

    ( const Ref<Dictionary>&  funcs    );

  void                      setNameFilter

    ( const String&           pattern   );


 protected:

  virtual                  ~FuncsParser ();

  virtual void              handleAttribute

    ( State&                  state,
      const String&           name,
      const String&           value,
      idx_t                   index )     override;


 private:

  static const char*        ATTRIBS_;
  static const idx_t        NAME_ATTRIB_;

  Ref<ParserScope>          scope_;
  Ref<Dictionary>           funcs_;
  Ref<FuncParser>           funcParser_;

  String                    funcName_;
  String                    nameFilter_;

};


JIVE_END_PACKAGE( util )

#endif
