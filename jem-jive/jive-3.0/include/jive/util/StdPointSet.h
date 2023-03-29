
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

#ifndef JIVE_UTIL_STDPOINTSET_H
#define JIVE_UTIL_STDPOINTSET_H

#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jive/util/XPointSet.h>


JIVE_BEGIN_PACKAGE( util )


class XItemMap;


//-----------------------------------------------------------------------
//   class StdPointSet
//-----------------------------------------------------------------------


class StdPointSet : public XPointSet,
                    public Clonable,
                    public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( StdPointSet, XPointSet );


                            StdPointSet ();

                            StdPointSet

    ( const String&           name,
      const String&           pointName,
      const Ref<XItemMap>&    map = nullptr );

                            StdPointSet

    ( const Self&             rhs );

  virtual void              resetEvents   ()       override;

  virtual void              readFrom

    ( ObjectInput&            in  )                override;

  virtual void              writeTo

    ( ObjectOutput&           out )          const override;

  virtual Ref<Object>       clone         () const override;
  virtual idx_t             size          () const override;
  virtual idx_t             rank          () const override;
  virtual String            getItemName   () const override;
  virtual ItemMap*          getItemMap    () const override;
  virtual void              clear         ()       override;

  virtual void              reserve

    ( idx_t                   count )              override;

  virtual void              trimToSize    ()       override;

  virtual idx_t             addPoint

    ( const Vector&           coords )             override;

  virtual idx_t             addPoint

    ( idx_t                   id,
      const Vector&           coords )             override;

  virtual idx_t             addPoints

    ( const Matrix&           coords )             override;

  virtual void              reorderPoints

    ( const Reordering&       reord )              override;

  virtual void              getPointCoords

    ( const Vector&           coords,
      idx_t                   ipoint )       const override;

  virtual double            getPointXCoord

    ( idx_t                   ipoint,
      idx_t                   idim )         const override;

  virtual void              setPointCoords

    ( idx_t                   ipoint,
      const Vector&           coords )             override;

  virtual void              setPointXCoord

    ( idx_t                   ipoint,
      double                  coord,
      idx_t                   idim )               override;

  virtual void              getCoords

    ( const Matrix&           coords )       const override;

  virtual void              getXCoords

    ( const Vector&           coords,
      idx_t                   idim )         const override;

  virtual void              getSomeCoords

    ( const Matrix&           coords,
      const IdxVector&        ipoints )      const override;

  virtual void              getSomeXCoords

    ( const Vector&           coords,
      const IdxVector&        ipoints,
      idx_t                   idim )         const override;

  virtual void              setCoords

    ( const Matrix&           coords )             override;

  virtual void              setXCoords

    ( const Vector&           coords,
      idx_t                   idim )               override;

  virtual void              setSomeCoords

    ( const IdxVector&        ipoints,
      const Matrix&           coords )             override;

  virtual void              setSomeXCoords

    ( const IdxVector&        ipoints,
      const Vector&           coords,
      idx_t                   idim )               override;


 protected:

  virtual                  ~StdPointSet   ();

  virtual void              emitEvents_

    ( int                     events )             override;


 private:

  class                     Data_;

  Ref<Data_>                data_;
  Ref<XItemMap>             map_;

};


JIVE_END_PACKAGE( util )

#endif
