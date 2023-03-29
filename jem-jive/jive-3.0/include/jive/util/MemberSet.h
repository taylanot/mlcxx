
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

#ifndef JIVE_UTIL_MEMBERSET_H
#define JIVE_UTIL_MEMBERSET_H

#include <jive/util/Topology.h>
#include <jive/util/ItemSet.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class MemberSet
//-----------------------------------------------------------------------


class MemberSet : public ItemSet
{
 public:

  JEM_DECLARE_CLASS     ( MemberSet, ItemSet );


  virtual String          getItemName       () const override;

  virtual void            getMember

    ( idx_t&                iitem,
      idx_t&                ilocal,
      idx_t                 imbr )             const = 0;

  virtual ItemSet*        getCompoundItems  () const = 0;

  virtual void            printSubset

    ( PrintWriter&          out,
      const IdxVector&      imbrs )            const override;

  IdxVector               getMembersOf

    ( const IdxVector&      iitems )           const;

  static IdxVector        getMembersOf

    ( const IdxVector&      iitems,
      const Topology&       topo   );

  Topology                getMembersPerItem () const;


 protected:

  explicit                MemberSet

    ( const String&         name = "" );

  virtual                ~MemberSet         ();

};


JIVE_END_PACKAGE( util )

#endif
