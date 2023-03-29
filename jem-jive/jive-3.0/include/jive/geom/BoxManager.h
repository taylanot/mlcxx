
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

#ifndef JIVE_GEOM_BOXMANAGER_H
#define JIVE_GEOM_BOXMANAGER_H

#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/geom/import.h>


JIVE_BEGIN_PACKAGE( geom )


class MaskMatrix;


//-----------------------------------------------------------------------
//   class BoxManager
//-----------------------------------------------------------------------


class BoxManager : public Object
{
 public:

  JEM_DECLARE_CLASS     ( BoxManager, Object );


  virtual idx_t           size          () const = 0;
  virtual idx_t           rank          () const = 0;
  virtual void            clear         ()       = 0;

  virtual void            reserve

    ( idx_t                 boxCount );

  virtual void            trimToSize    ();
  virtual void            flushChanges  ();

  virtual void            setMaskMatrix

    ( Ref<MaskMatrix>       mask )               = 0;

  virtual void            getEnclosingBox

    ( const Vector&         box )          const = 0;

  virtual idx_t           addBox

    ( const Vector&         box,
      idx_t                 color = 0 )          = 0;

  virtual idx_t           addBoxes

    ( const Matrix&         boxes,
      idx_t                 color = 0 );

  void                    eraseBoxes

    ( const IdxVector&      iboxes );

  virtual void            reorderBoxes

    ( const Reordering&     reord )              = 0;

  virtual void            getBox

    ( const Vector&         box,
      idx_t                 ibox )         const = 0;

  virtual void            getBoxes

    ( const Matrix&         boxes )        const;

  virtual void            getSomeBoxes

    ( const Matrix&         boxes,
      const IdxVector&      iboxes )       const;

  virtual void            setColors

    ( const IdxVector&      colors );

  virtual void            setSomeColors

    ( const IdxVector&      iboxes,
      const IdxVector&      colors )             = 0;

  virtual void            updateBoxes

    ( const Matrix&         boxes );

  virtual void            updateSomeBoxes

    ( const IdxVector&      iboxes,
      const Matrix&         boxes )              = 0;

  virtual idx_t           findChangedBoxes

    ( const IdxVector&      iboxes,
      const Matrix&         boxes )        const = 0;

  virtual idx_t           findBoxNeighbors

    ( const IdxVector&      iboxes,
      idx_t                 jbox )         const = 0;

  virtual idx_t           findBoxNeighbors

    ( const IdxVector&      iboxes,
      const Vector&         box,
      idx_t                 color = 0 )    const = 0;


 protected:

  virtual                ~BoxManager    ();

};


JIVE_END_PACKAGE( geom )

#endif
