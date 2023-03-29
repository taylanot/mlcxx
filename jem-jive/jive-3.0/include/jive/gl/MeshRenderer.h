
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

#ifndef JIVE_GL_MESHRENDERER_H
#define JIVE_GL_MESHRENDERER_H

#include <jem/gl/LeafNode.h>
#include <jem/util/event/Event.h>
#include <jive/Array.h>
#include <jive/gl/import.h>
#include <jive/gl/AppRef.h>
#include <jive/gl/GLArray.h>


JIVE_BEGIN_PACKAGE( gl )


class GfxMesher;
class MeshAttribs;

using jem::util::Event;


//-----------------------------------------------------------------------
//   class MeshRenderer
//-----------------------------------------------------------------------


class MeshRenderer : public jem::gl::LeafNode
{
 public:

  JEM_DECLARE_CLASS       ( MeshRenderer, jem::gl::LeafNode );

  typedef MeshAttribs       Attribs;

  Event< Self& >            newMeshEvent;


                            MeshRenderer

    ( const String&           name,
      const AppContext&       apx,
      const ElementGroup&     group,
      const Ref<ShapeTable>&  shapes,
      const Ref<GfxMesher>&   mesher );

  virtual void              getBBox

    ( jem::gl::Box&           box )            const override;

  virtual void              configure

    ( const Properties&       props )                override;

  virtual void              getConfig

    ( const Properties&       props )          const override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )               override;

  virtual void              redraw

    ( RenderContext&          rtx )                  override;

  virtual void              redrawFast

    ( RenderContext&          rtx,
      double                  lod )                  override;

  void                      drawCData

    ( RenderContext&          rtx );

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd,
      const Properties&       params )               override;

  virtual void              listOptions

    ( jem::gl::Menu&          menu )           const override;

  void                      update

    ( const AppContext&       apx );

  void                      clearColors     ();

  void                      setNodeColors

    ( const GLfloatVector&    colors,
      const Ref<ColorMap>&    colorMap,
      const AppContext&       apx );

  void                      setElemColors

    ( const GLubyteMatrix&    colors,
      const AppContext&       apx );

  void                      trackMeshGeom   ();

  void                      setNodeCoords

    ( const Matrix&           coords,
      const AppContext&       apx );

  void                      setNodeCoords

    ( idx_t                   idim,
      const Vector&           coords,
      const AppContext&       apx );

  void                      addToNodeCoords

    ( const Matrix&           delta,
      const AppContext&       apx );

  void                      addToNodeCoords

    ( idx_t                   idim,
      const Vector&           delta,
      const AppContext&       apx );

  idx_t                     findElement

    ( idx_t&                  elemID,
      double                  x,
      double                  y,
      double                  z )              const;

  VertexSet*                getVertices     () const;
  IdxVector                 getVxNodeMap    () const;
  jem::gl::Color            getColor        () const;

  IdxVector                 getNodeVxMap

    ( const AppContext&        apx )           const;

  IdxVector                 getElemIndices

    ( const AppContext&       apx )            const;

  ElementGroup              getElemGroup

    ( const AppContext&       apx )            const;

  ElementSet                getElements

    ( const AppContext&       apx )            const;

  NodeSet                   getNodes

    ( const AppContext&       apx )            const;

  static Ref<Self>          get

    ( const Properties&       viewdat,
      const String&           context );


 protected:

  virtual                  ~MeshRenderer    ();


 private:

  bool                      updateMesh_

    ( const AppContext&       apx );

  void                      checkShapes_

    ( const ShapeTable&       shapes,
      const ElementSet&       elems )          const;


 private:

  class                     AppData_;
  class                     OwnData_;
  class                     DrawScope_;
  friend class              OwnData_;
  friend class              DrawScope_;

  AppRef<AppData_>          appData_;
  Ref<OwnData_>             ownData_;

};


JIVE_END_PACKAGE( gl )

#endif
