
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

#ifndef JIVE_UTIL_XGROUPSET_H
#define JIVE_UTIL_XGROUPSET_H

#include <jive/util/GroupSet.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class XGroupSet
//-----------------------------------------------------------------------


class XGroupSet : public GroupSet
{
 public:

  JEM_DECLARE_CLASS     ( XGroupSet, GroupSet );


  virtual void            clear         ()       = 0;

  virtual void            reserve

    ( idx_t                 count );

  virtual void            trimToSize    ();

  virtual idx_t           addGroup

    ( const IdxVector&      iitems )             = 0;

  virtual idx_t           addGroup

    ( idx_t                 id,
      const IdxVector&      iitems )             = 0;

  virtual idx_t           addGroups

    ( const IdxMatrix&      groups );

  void                    eraseGroups

    ( const IdxVector&      igroups );

  virtual void            reorderGroups

    ( const Reordering&     reord )              = 0;

  virtual void            setGroupMembers

    ( idx_t                 igroup,
      const IdxVector&      iitems )             = 0;


 protected:

                          XGroupSet

    ( const String&         name = "" );

  virtual                ~XGroupSet     ();

};


JIVE_END_PACKAGE( util )

#endif
