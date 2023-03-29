
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

#ifndef JIVE_UTIL_XDOFSPACE_H
#define JIVE_UTIL_XDOFSPACE_H

#include <jive/util/DofSpace.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class XDofSpace
//-----------------------------------------------------------------------


class XDofSpace : public DofSpace
{
 public:

  JEM_DECLARE_CLASS         ( XDofSpace, DofSpace );


  virtual void                clearDofs     ()       = 0;
  virtual void                clearAll      ()       = 0;

  virtual void                reserve

    ( idx_t                     dofCount );

  virtual void                trimToSize    ();

  virtual idx_t               addType

    ( const String&             name )               = 0;

  virtual IdxVector           addTypes

    ( const StringVector&       names );

  virtual idx_t               addDof

    ( idx_t                     iitem,
      idx_t                     itype )              = 0;

  virtual void                addDofs

    ( const IdxVector&          iitems,
      const IdxVector&          itypes );

  void                        addDofs

    ( idx_t                     iitem,
      const IdxVector&          itypes );

  void                        addDofs

    ( const IdxVector&          iitems,
      idx_t                     itype );

  void                        eraseDofs

    ( const IdxVector&          idofs );

  virtual void                reorderDofs

    ( const Reordering&         reord )              = 0;

  static Ref<XDofSpace>       find

    ( const Properties&         globdat );

  static Ref<XDofSpace>       find

    ( const Ref<ItemSet>&       items,
      const Properties&         globdat );

  static Ref<XDofSpace>       get

    ( const Properties&         globdat,
      const String&             context );

  static Ref<XDofSpace>       get

    ( const Ref<ItemSet>&       items,
      const Properties&         globdat );


 protected:

  virtual                    ~XDofSpace     ();


 private:

  static Ref<XDofSpace>       newDofSpace_

    ( const Ref<ItemSet>&       items );

};


JIVE_END_PACKAGE( util )

#endif
