
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

#ifndef JIVE_MESH_ELLIPSE_H
#define JIVE_MESH_ELLIPSE_H

#include <jive/mesh/Geometry.h>


JIVE_BEGIN_PACKAGE( mesh )


class StripMesher;


//-----------------------------------------------------------------------
//   class Ellipse
//-----------------------------------------------------------------------


class Ellipse : public Geometry
{
 public:

  JEM_DECLARE_CLASS       ( Ellipse, Geometry );

  struct                    GroupNames
  {
    String                    edge;
    String                    center;
  };

  static const char*        TYPE_NAME;

  enum                      ElemType
  {
                              TRIANGLE3,
                              TRIANGLE6
  };


  explicit                  Ellipse

    ( const String&           name = "ellipse" );

  virtual idx_t             rank      () const override;

  virtual void              configure

    ( const Properties&       props )          override;

  virtual void              getConfig

    ( const Properties&       conf )     const override;

  virtual void              makeMesh

    ( const XMesh&            mesh,
      const Params&           param )          override;

  static void               declare   ();

  static Ref<Geometry>      makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


 public:

  GroupNames                groupNames;


 protected:

  virtual                  ~Ellipse   ();


 private:

  void                      makeCircularMesh_

    ( const XMesh&            mesh,
      const Params&           Params );

  void                      makeElongatedMesh_

    ( const XMesh&            mesh,
      const Params&           Params );


 private:

  Ref<StripMesher>          stripMesher_;

  double                    xradius_;
  double                    yradius_;
  int                       elemType_;

  IdxVector                 edgeNodes_;

  Vector                    xinner_;
  Vector                    xouter_;
  IdxVector                 iinner_;
  IdxVector                 iouter_;

  Vector                    xnode_;

};


JIVE_END_PACKAGE( mesh )

#endif
