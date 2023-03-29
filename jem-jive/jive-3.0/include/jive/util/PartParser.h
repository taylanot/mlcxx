
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

#ifndef JIVE_UTIL_PARTPARSER_H
#define JIVE_UTIL_PARTPARSER_H

#include <jem/util/Properties.h>
#include <jem/xml/Parser.h>
#include <jem/xml/AttributeParser.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


class ParserScope;
class PartEndParser;


//-----------------------------------------------------------------------
//   class PartParser
//-----------------------------------------------------------------------


class PartParser : public    jem::xml::Parser,
                   protected jem::xml::AttributeParser
{
 public:

  JEM_DECLARE_CLASS       ( PartParser, jem::xml::Parser );

  friend class              PartEndParser;

  static const char*        TAG_NAME;


  explicit                  PartParser

    ( idx_t                   maxParts  );

  virtual void              reset       () override;

  virtual void              parse

    ( State&                  state  )     override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )     override;


 protected:

  virtual                  ~PartParser  ();

  virtual void              handleAttribute

    ( State&                  state,
      const String&           name,
      const String&           value,
      idx_t                   index  )     override;


 private:

  struct                    Attribs_
  {
    String                    partName;
    idx_t                     maxParts;
  };

  static const char*        ATTRIB_NAMES_;
  static const idx_t        NAME_ATTRIB_;
  static const idx_t        MAX_PARTS_ATTRIB_;

  Ref<ParserScope>          scope_;
  Properties                parts_;
  idx_t                     maxParts_;

  Attribs_                  attribs_;

};


JIVE_END_PACKAGE( util )

#endif
