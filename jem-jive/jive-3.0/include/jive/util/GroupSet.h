
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

#ifndef JIVE_UTIL_GROUPSET_H
#define JIVE_UTIL_GROUPSET_H

#include <jive/util/Topology.h>
#include <jive/util/ItemSet.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class GroupSet
//-----------------------------------------------------------------------


class GroupSet : public ItemSet
{
 public:

  JEM_DECLARE_CLASS     ( GroupSet, ItemSet );

  Event < Self& >         newTopoEvent;


  virtual String          getItemName     () const override;

  virtual idx_t           getGroupSize

    ( idx_t                 igroup )         const = 0;

  virtual idx_t           getGroupMembers

    ( const IdxVector&      iitems,
      idx_t                 igroup  )        const = 0;

  virtual void            getSomeGroupMembers

    ( const IdxVector&      iitems,
      const IdxVector&      index,
      idx_t                 igroup  )        const;

  virtual idx_t           getOneGroupMember

    ( idx_t                 index,
      idx_t                 igroup  )        const;

  virtual IdxVector       getMembersOf

    ( const IdxVector&      igroups )        const;

  virtual IdxVector       getUniqueMembersOf

    ( const IdxVector&      igroups )        const;

  virtual idx_t           maxGroupSize    () const;

  virtual idx_t           maxGroupSizeOf

    ( const IdxVector&      igroups )        const;

  virtual IdxVector       toVector        () const;
  virtual Topology        toMatrix        () const;
  virtual ItemSet*        getGroupedItems () const = 0;

  virtual void            printSubset

    ( PrintWriter&          out,
      const IdxVector&      igroups )        const override;


 protected:

  explicit                GroupSet

    ( const String&         name = "" );

  virtual                ~GroupSet        ();

};


JIVE_END_PACKAGE( util )

#endif
