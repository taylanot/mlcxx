
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

#ifndef JIVE_GEOM_GRIDBOXMANAGER_H
#define JIVE_GEOM_GRIDBOXMANAGER_H

#include <jem/io/Serializable.h>
#include <jive/geom/BoxManager.h>
#include <jive/geom/import.h>


JIVE_BEGIN_PACKAGE( geom )


//-----------------------------------------------------------------------
//   class GridBoxManager
//-----------------------------------------------------------------------


class GridBoxManager : public BoxManager,
                       public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( GridBoxManager, BoxManager );


                          GridBoxManager ();

  explicit                GridBoxManager

    ( idx_t                 rank,
      idx_t                 boxesPerCell = 10 );

  virtual void            readFrom

    ( ObjectInput&          in )                  override;

  virtual void            writeTo

    ( ObjectOutput&         out )           const override;


  virtual idx_t           size           () const override;
  virtual idx_t           rank           () const override;

  virtual void            clear          ()       override;
  virtual void            reserve

    ( idx_t                 boxCount )            override;

  virtual void            trimToSize     ()       override;
  virtual void            flushChanges   ()       override;

  virtual void            setMaskMatrix

    ( Ref<MaskMatrix>       mask )                override;

  virtual void            getEnclosingBox

    ( const Vector&         box )           const override;

  virtual idx_t           addBox

    ( const Vector&         box,
      idx_t                 color )               override;

  virtual idx_t           addBoxes

    ( const Matrix&         boxes,
      idx_t                 color )               override;

  virtual void            reorderBoxes

    ( const Reordering&     reord  )              override;

  virtual void            getBox

    ( const Vector&         box,
      idx_t                 ibox )          const override;

  virtual void            getBoxes

    ( const Matrix&         boxes )         const override;

  virtual void            getSomeBoxes

    ( const Matrix&         boxes,
      const IdxVector&      iboxes )        const override;

  virtual void            setColors

    ( const IdxVector&      colors )              override;

  virtual void            setSomeColors

    ( const IdxVector&      iboxes,
      const IdxVector&      colors )              override;

  virtual void            updateBoxes

    ( const Matrix&         boxes )               override;

  virtual void            updateSomeBoxes

    ( const IdxVector&      iboxes,
      const Matrix&         boxes )               override;

  virtual idx_t           findChangedBoxes

    ( const IdxVector&      iboxes,
      const Matrix&         boxes )         const override;

  virtual idx_t           findBoxNeighbors

    ( const IdxVector&      iboxes,
      idx_t                 jbox )          const override;

  virtual idx_t           findBoxNeighbors

    ( const IdxVector&      iboxes,
      const Vector&         box,
      idx_t                 color )         const override;


 protected:

  virtual                ~GridBoxManager ();


 private:

  void                    init_

    ( idx_t                 rank,
      idx_t                 boxesPerCell );


 private:

  class                   Grid_;
  class                   Utils_;
  class                   Scratch_;
  class                   Data_;

  friend class            Utils_;
  friend class            Data_;

  Ref<Data_>              data_;

};


JIVE_END_PACKAGE( geom )

#endif
