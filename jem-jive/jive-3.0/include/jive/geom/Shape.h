
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

#ifndef JIVE_GEOM_SHAPE_H
#define JIVE_GEOM_SHAPE_H

#include <jem/base/NamedObject.h>
#include <jem/util/event/Event.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( geom )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class Shape
//-----------------------------------------------------------------------


class Shape : public NamedObject
{
 public:

  JEM_DECLARE_CLASS     ( Shape, NamedObject );

  typedef const void*     ExtensionID;

  Event < Self& >         changeEvent;


  explicit                Shape

    ( const String&         name = "" );

  virtual String          getContext            () const override;

  bool                    isRegular             () const;
  virtual idx_t           globalRank            () const = 0;
  virtual idx_t           localRank             () const = 0;
  virtual idx_t           nodeCount             () const = 0;
  virtual idx_t           vertexCount           () const = 0;
  virtual idx_t           ipointCount           () const = 0;
  virtual idx_t           shapeFuncCount        () const = 0;
  virtual String          getGeometry           () const;
  inline  idx_t           integrationPointCount () const;
  virtual Matrix          getVertexCoords       () const = 0;
  virtual Matrix          getIntegrationScheme  () const = 0;

  virtual void            getGlobalIntegrationPoints

    ( const Matrix&         x,
      const Matrix&         c )                    const;

  virtual void            getIntegrationWeights

    ( const Vector&         w,
      const Matrix&         c )                    const = 0;

  virtual void            getGlobalPoint

    ( const Vector&         x,
      const Vector&         u,
      const Matrix&         c )                    const = 0;

  virtual bool            getLocalPoint

    ( const Vector&         u,
      const Vector&         x,
      double                eps,
      const Matrix&         c )                    const;

  virtual bool            containsLocalPoint

    ( const Vector&         u )                    const = 0;

  virtual Matrix          getShapeFunctions     () const = 0;
  virtual Matrix          getVertexFunctions    () const = 0;

  virtual void            evalShapeFunctions

    ( const Vector&         h,
      const Vector&         u )                    const = 0;

  virtual void*           getExtByID

    ( ExtensionID           extID )                const;

  template <class Ext>
    inline bool           hasExtension          () const;

  template <class Ext>
    inline Ext*           getExtension          () const;


 protected:

  virtual                ~Shape                 ();

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   integrationPointCount
//-----------------------------------------------------------------------


inline idx_t Shape::integrationPointCount () const
{
  return ipointCount ();
}


//-----------------------------------------------------------------------
//   hasExtension
//-----------------------------------------------------------------------


template <class Ext>

  inline bool Shape::hasExtension () const

{
  return (getExtByID( Ext::ID ) != 0);
}


//-----------------------------------------------------------------------
//   getExtension
//-----------------------------------------------------------------------


template <class Ext>

  inline Ext* Shape::getExtension () const

{
  return static_cast<Ext*> ( getExtByID( Ext::ID ) );
}


JIVE_END_PACKAGE( geom )

#endif
