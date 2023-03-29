
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

#ifndef JIVE_MESH_GEOMETRY_H
#define JIVE_MESH_GEOMETRY_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/mesh/import.h>


JIVE_BEGIN_PACKAGE( mesh )


class XMesh;


//-----------------------------------------------------------------------
//   class Geometry
//-----------------------------------------------------------------------


class Geometry : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( Geometry, NamedObject );

  static const double       PI;

  class                     Params;


  explicit                  Geometry

    ( const String&           name = "" );

  virtual idx_t             rank          () const = 0;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       conf )         const;

  virtual void              makeMesh

    ( const XMesh&            mesh,
      const Params&           params )             = 0;

  virtual String            getContext    () const override;

  static void               addGroup

    ( const Params&           params,
      const String&           name,
      const IdxVector&        inodes,
      const NodeSet&          nodes );

  static void               addGroup

    ( const Params&           params,
      const String&           name,
      const IdxVector&        ielems,
      const ElementSet&       elems );


 protected:

  virtual                  ~Geometry      ();

};


//-----------------------------------------------------------------------
//   class Geometry::Params
//-----------------------------------------------------------------------


class Geometry::Params
{
 public:

  typedef Params            Self;


  explicit                  Params

    ( double                  elemSize = 1.0,
      double                  scale    = 1.0 );

  inline bool               isValid () const;


 public:

  double                    elemSize;
  double                    scale;

  String                    groupPrefix;

  Ref<Dictionary>           nodeGroups;
  Ref<Dictionary>           elemGroups;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Geometry>               newGeometry

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         globdat );





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Geometry::Params
//=======================================================================

//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


inline bool Geometry::Params::isValid () const
{
  return (elemSize > 0.0 && scale > 0.0);
}


JIVE_END_PACKAGE( mesh )

#endif
