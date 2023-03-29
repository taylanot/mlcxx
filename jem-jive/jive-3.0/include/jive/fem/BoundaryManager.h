
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

#ifndef JIVE_FEM_BOUNDARYMANAGER_H
#define JIVE_FEM_BOUNDARYMANAGER_H

#include <jem/io/Serializable.h>
#include <jive/util/EventMask.h>
#include <jive/fem/import.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/XBoundarySet.h>
#include <jive/fem/BoundaryGroup.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class BoundaryManager
//-----------------------------------------------------------------------


class BoundaryManager : public Object,
                        public EventSource,
                        public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( BoundaryManager, Object );

  Event < Self& >           newBuddiesEvent;


                            BoundaryManager ();

                            BoundaryManager

    ( const ElementSet&       elems,
      const Ref<ShapeTable>&  shapes );

                            BoundaryManager

    ( const ElementGroup&     group,
      const Ref<ShapeTable>&  shapes );

                            BoundaryManager

    ( const XBoundarySet&     bounds,
      const Ref<ShapeTable>&  shapes );

                            BoundaryManager

    ( const ElementGroup&     group,
      const XBoundarySet&     bounds,
      const Ref<ShapeTable>&  shapes );

  virtual void              resetEvents     ()       override;

  virtual void              readFrom

    ( ObjectInput&            in  )                  override;

  virtual void              writeTo

    ( ObjectOutput&           out )            const override;

  inline void               update          ();

  BoundarySet               getBoundaries   () const;
  BoundaryGroup             getXBoundaries  () const;
  BoundaryGroup             getIBoundaries  () const;
  IdxVector                 getBuddies      () const;


 protected:

  virtual                  ~BoundaryManager ();

  virtual void              emitEvents_

    ( int                     events )               override;


 private:

  void                      init_           ();
  void                      connect_        ();
  void                      update_         ();
  void                      invalidate_     ();


 private:

  static const int          NEW_BUDDIES_;

  ElementGroup              egroup_;
  XBoundarySet              bounds_;
  Ref<ShapeTable>           shapes_;
  IdxVector                 buddies_;
  EventMask                 events_;

  bool                      updated_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


inline void BoundaryManager::update ()
{
  if ( ! updated_ )
  {
    update_ ();
  }
}


JIVE_END_PACKAGE( fem )

#endif
