
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

#ifndef JIVE_UTIL_ITEMGROUPPARSER_H
#define JIVE_UTIL_ITEMGROUPPARSER_H

#include <jem/xml/Parser.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;
class XItemGroup;


//-----------------------------------------------------------------------
//   class ItemGroupParser
//-----------------------------------------------------------------------


class ItemGroupParser : public jem::xml::Parser
{
 public:

  JEM_DECLARE_CLASS       ( ItemGroupParser, jem::xml::Parser );


  explicit                  ItemGroupParser

    ( const Ref<XItemGroup>&  group  = nullptr,
      const Ref<Dictionary>&  groups = nullptr );

  virtual void              reset   ()          override;

  virtual void              parse

    ( State&                  state  )          override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )          override;

  void                      setGroup

    ( const Ref<XItemGroup>&  group  );

  void                      setGroups

    ( const Ref<Dictionary>&  groups );


 protected:

  virtual                  ~ItemGroupParser  ();


 private:

  class                     Operator_;
  class                     Operand_;
  class                     Data_;
  class                     Utils_;

  friend class              Data_;
  friend class              Utils_;

  Ref<Data_>                data_;

};



JIVE_END_PACKAGE( util )

#endif
