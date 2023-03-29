
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

#ifndef JEM_XML_SIMPLETAGPARSER_H
#define JEM_XML_SIMPLETAGPARSER_H

#include <jem/xml/TagParser.h>


JEM_BEGIN_PACKAGE( xml )


//-----------------------------------------------------------------------
//   class SimpleTagParser
//-----------------------------------------------------------------------


class SimpleTagParser : public TagParser
{
 public:

  typedef SimpleTagParser   Self;
  typedef TagParser         Super;


  explicit                  SimpleTagParser

    ( const Ref<Parser>&      parser );

  virtual void              reset           () override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )         override;

  virtual void              parseContents

    ( State&                  state )          override;


 protected:

  virtual                  ~SimpleTagParser ();

  virtual void              handleAttribute

    ( ParserState&            state,
      const String&           name,
      const String&           value,
      idx_t                   index )          override;


 private:

  Ref<Parser>               parser_;

};


JEM_END_PACKAGE( xml )

#endif
