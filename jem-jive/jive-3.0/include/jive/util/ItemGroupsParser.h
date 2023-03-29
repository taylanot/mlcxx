
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

#ifndef JIVE_UTIL_ITEMGROUPSPARSER_H
#define JIVE_UTIL_ITEMGROUPSPARSER_H

#include <jem/xml/TagParser.h>
#include <jive/util/import.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;
class XItemGroup;
class ItemGroupParser;
class ParserScope;


//-----------------------------------------------------------------------
//   class ItemGroupsParser
//-----------------------------------------------------------------------


class ItemGroupsParser : public jem::xml::TagParser
{
 public:

  JEM_DECLARE_CLASS       ( ItemGroupsParser,
                            jem::xml::TagParser );


  explicit                  ItemGroupsParser

    ( const Ref<ItemSet>&     items   = nullptr,
      const Ref<Dictionary>&  groups  = nullptr,
      const String&           filter  = "*" );

  virtual void              reset      ()        override;

  virtual void              parseContents

    ( State&                  state     )        override;

  virtual void              handleEmptyContents

    ( State&                  state     )        override;

  virtual void              finish

    ( State&                  state     )        override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params    )        override;

  void                      setGroups

    ( const Ref<ItemSet>&     items,
      const Ref<Dictionary>&  groups  );

  void                      setNameFilter

    ( const String&           pattern );


 protected:

  virtual                  ~ItemGroupsParser  ();

  virtual void              handleAttribute

    ( State&                  state,
      const String&           name,
      const String&           value,
      idx_t                   index )            override;


 private:

  void                      newGroup_

    ( State&                  state );


 private:

  Ref<ParserScope>          scope_;
  Ref<ItemSet>              items_;
  Ref<XItemGroup>           group_;
  Ref<Dictionary>           groups_;
  Ref<ItemGroupParser>      groupParser_;

  String                    groupName_;
  String                    nameFilter_;

};


JIVE_END_PACKAGE( util )

#endif
