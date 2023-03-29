
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

#ifndef JIVE_UTIL_GROUPCONVERTER_H
#define JIVE_UTIL_GROUPCONVERTER_H

#include <jem/base/Object.h>
#include <jem/util/Properties.h>
#include <jive/util/import.h>


namespace jem
{
  namespace numeric
  {
    class SparseStructure;
  }
}


JIVE_BEGIN_PACKAGE( util )


class ItemSet;

typedef jem::numeric
  ::SparseStructure         Topology;


//-----------------------------------------------------------------------
//   class GroupConverter
//-----------------------------------------------------------------------


class GroupConverter : public Object
{
 public:

  JEM_DECLARE_CLASS       ( GroupConverter, Object );


  static const char*        NAME_PROP;
  static const char*        TEMP_PROP;
  static const char*        SOURCE_PROP;
  static const char*        MINDEG_PROP;
  static const char*        MAXDEG_PROP;


                            GroupConverter

    ( const Ref<ItemSet>&     aItems,
      const Ref<ItemSet>&     bItems,
      const Properties&       globdat,
      const String&           context = "" );

                            GroupConverter

    ( const Ref<ItemSet>&     aItems,
      const Ref<Dictionary>&  aGroups,
      const Ref<ItemSet>&     bItems,
      const Ref<Dictionary>&  bGroups,
      const String&           context = "" );

  void                      init

    ( const Properties&       conf,
      const Properties&       props );

  void                      init

    ( const Properties&       aConf,
      const Properties&       aProps,
      const Properties&       bConf,
      const Properties&       bProps );

  bool                      isDone         () const;

  idx_t                     convert

    ( const Topology&         abMap,
      const Topology&         baMap );

  void                      finish         ();


 protected:

  virtual                  ~GroupConverter ();


 private:

  void                      init_

    ( const Ref<ItemSet>&     aItems,
      const Ref<Dictionary>&  aGroups,
      const Ref<ItemSet>&     bItems,
      const Ref<Dictionary>&  bGroups );


 private:

  static const char*        DONE_PROP_;


  class                     Utils_;
  friend class              Utils_;


  struct                    Groups_
  {
    inline                    Groups_      ();

    Ref<ItemSet>              items;
    Ref<Dictionary>           groups;

    Properties                conf;
    Properties                props;

    String                    prefix;
    bool                      done;
  };

  String                    context_;
  Groups_                   groups_[2];

};


JIVE_END_PACKAGE( util )

#endif
