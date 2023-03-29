
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

#ifndef JIVE_UTIL_DATAPARSER_H
#define JIVE_UTIL_DATAPARSER_H

#include <jem/xml/CompoundParser.h>
#include <jive/util/import.h>
#include <jive/util/typedefs.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;
class XPointSet;
class XGroupSet;
class XMemberSet;
class XTable;
class Database;


//-----------------------------------------------------------------------
//   class DataParser
//-----------------------------------------------------------------------


class DataParser : public jem::xml::CompoundParser
{
 public:

  JEM_DECLARE_CLASS       ( DataParser, jem::xml::CompoundParser );


  explicit                  DataParser

    ( const Ref<MPContext>&   mpx = nullptr );

  void                      parseFile

    ( const String&           fname,
      const Ref<Writer>&      logger = nullptr );

  void                      setMaxParts

    ( idx_t                   maxParts );

  void                      skipUnknownTags

    ( bool                    choice    = true );

  void                      addPointSetParser

    ( const Ref<XPointSet>&   points,
      idx_t                   rank      = -1,
      const String&           tag       = "" );

  void                      addGroupSetParser

    ( const Ref<XGroupSet>&   groups,
      idx_t                   groupSize = -1,
      const String&           tag       = "" );

  void                      addMemberSetParser

    ( const Ref<XMemberSet>&  members,
      idx_t                   maxLocal  = -1,
      const String&           tag       = "" );

  void                      addItemGroupsParser

    ( const Ref<ItemSet>&     items,
      const Ref<Dictionary>&  groups,
      const String&           filter    = "*",
      const String&           tag       =  "" );

  void                      addItemGroupsParsers

    ( const Properties&       globdat,
      const String&           filter    = "*" );

  void                      addTableParser

    ( const Ref<XTable>&      table,
      const Ref<Dictionary>&  rowGroups = nullptr,
      const String&           tag       = "" );

  void                      addTablesParser

    ( const Ref<ItemSet>&     items,
      const Ref<Dictionary>&  tables,
      const Ref<Dictionary>&  rowGroups = nullptr,
      const String&           filter    = "*",
      const String&           tag       = "" );

  void                      addTablesParsers

    ( const Properties&       globdat,
      const String&           filter     = "*" );

  void                      addDbaseParser

    ( const Ref<Database>&    dbase,
      const Ref<Dictionary>&  rowGroups = nullptr,
      const String&           tag       = "" );

  void                      addDbasesParser

    ( const Ref<ItemSet>&     items,
      const Ref<Dictionary>&  dbases,
      const Ref<Dictionary>&  rowGroups = nullptr,
      const String&           filter    = "*",
      const String&           tag       = "" );

  void                      addDbasesParsers

    ( const Properties&       globdat,
      const String&           filter    = "*" );

  void                      addConstraintsParser

    ( const Ref<ConParser>&   conParser,
      const String&           tag       = "" );

  void                      addConstraintsParsers

    ( const Properties&       globdat );

  void                      addFuncsParser

    ( const Ref<Dictionary>&  funcs,
      const String&           filter    = "*",
      const String&           tag       = "" );

  void                      addFuncsParser

    ( const Properties&       globdat,
      const String&           filter    = "*",
      const String&           tag       = "" );


 protected:

  virtual                  ~DataParser ();


 private:

  class                     Utils_;

  Ref<MPContext>            mpx_;

};


JIVE_END_PACKAGE( util )

#endif
