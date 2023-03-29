
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#ifndef JEM_GL_TRIANGLESET_H
#define JEM_GL_TRIANGLESET_H

#include <jem/gl/Array.h>
#include <jem/gl/LeafNode.h>


JEM_BEGIN_PACKAGE( gl )


class ColorMap;
class VertexSet;
class ShaderProg;


//-----------------------------------------------------------------------
//   class TriangleSet
//-----------------------------------------------------------------------


class TriangleSet : public LeafNode
{
 public:

  JEM_DECLARE_CLASS       ( TriangleSet, LeafNode );


  enum                      ColorMode
  {
                              NO_COLORS,
                              TRIANG_COLORS,
                              VERTEX_COLORS
  };

  enum                      NormalMode
  {
                              AUTO_NORMALS,
                              TRIANG_NORMALS,
                              VERTEX_NORMALS
  };


                            TriangleSet

    ( const String&           name,
      const Ref<VertexSet>&   vx );

  virtual void              getBBox

    ( Box&                    box )              const override;

  virtual void              redraw

    ( RenderContext&          rtx )                    override;

  virtual void              redrawFast

    ( RenderContext&          rtx,
      double                  lod )                    override;

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd,
      const Properties&       params )                 override;

  virtual void              listOptions

    ( Menu&                   menu )             const override;

  inline idx_t              size              () const noexcept;
  IdxMatrix                 getTopology       () const;

  void                      setTopology

    ( const IdxMatrix&       topo );

  void                      setNoColors       ();

  void                      setTriangColors

    ( const GLubyteMatrix&    colors );

  void                      setVertexColors

    ( const GLfloatVector&    colors,
      const Ref<ColorMap>&    colorMap );

  inline ColorMode          getColorMode      () const noexcept;
  inline GLubyteMatrix      getTriangColors   () const noexcept;
  inline GLfloatVector      getVertexColors   () const noexcept;
  inline ColorMap*          getColorMap       () const noexcept;
  void                      setAutoNormals    ();

  void                      setTriangNormals

    ( const GLbyteMatrix&     normals );

  void                      setVertexNormals

    ( const GLbyteMatrix&     normals );

  inline NormalMode         getNormalMode     () const noexcept;
  inline GLbyteMatrix       getNormals        () const noexcept;

  void                      findTriangle

    ( idx_t&                  index,
      GLfloat&                dist,
      GLfloat                 x,
      GLfloat                 y,
      GLfloat                 z = 0.0F )         const;


 protected:

  virtual                  ~TriangleSet       ();


 private:

  void                      drawNoColors2D_

    ( GLContext&              gtx )              const;

  void                      drawNoColors3D_

    ( GLContext&              gtx )              const;

  void                      drawTrColors2D_

    ( GLContext&              gtx )              const;

  void                      drawTrColors3D_

    ( GLContext&              gtx )              const;

  void                      drawVxColors2D_

    ( GLContext&              gtx )              const;

  void                      drawVxColors3D_

    ( GLContext&              gtx )              const;

  void                      blendNoColors3D_

    ( GLContext&              gtx )              const;

  void                      blendTrColors3D_

    ( GLContext&              gtx )              const;

  void                      blendVxColors3D_

    ( GLContext&              gtx )              const;

  void                      plotNoColors2D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotNoColors3D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotTrColors2D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotTrColors3D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotVxColors2D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotVxColors3D_

    ( GLContext&              gtx,
      double                  lod )              const;

  inline void               beginBlend_

    ( GLContext&              gtx,
      GLboolean&              zmask )            const;

  void                      endBlend_

    ( GLContext&              gtx,
      GLboolean               zmask )            const;

  void                      findTriangle2D_

    ( idx_t&                  index,
      GLfloat&                dist,
      const GLfloat           pos[3] )           const;

  void                      findTriangle3D_

    ( idx_t&                  index,
      GLfloat&                dist,
      const GLfloat           pos[3] )           const;

  void                      checkData_

    ( const String&           context );

  void                      updateNormals_    ();
  void                      connectHandlers_  ();
  void                      newGeomHandler_   ();
  void                      newShapeHandler_  ();

  GLfloat                   getPointSize_

    ( GLContext&              gtx,
      double                  lod )              const;

  static void               copyNormal_

    ( GLfloat*                dest,
      const GLbyte*           src );


 private:

  class                     Work_;
  class                     Work2D_;
  class                     Work3D_;
  class                     Buffer_;

  friend class              Work_;
  friend class              Work2D_;
  friend class              Work3D_;

  Ref<VertexSet>            vertices_;
  IdxMatrix                 triangles_;

  ColorMode                 colorMode_;
  GLubyteMatrix             trColors_;
  GLfloatVector             vxColors_;
  Ref<ColorMap>             colorMap_;
  Ref<Buffer_>              buffer_;

  NormalMode                normalMode_;
  GLbyteMatrix              normals_;

  bool                      opaque_;
  bool                      checked_;
  bool                      updated_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t TriangleSet::size () const noexcept
{
  return triangles_.size (1);
}


//-----------------------------------------------------------------------
//   getColorMode
//-----------------------------------------------------------------------


inline TriangleSet::ColorMode
  TriangleSet::getColorMode () const noexcept
{
  return colorMode_;
}


//-----------------------------------------------------------------------
//   getTriangColors
//-----------------------------------------------------------------------


inline GLubyteMatrix TriangleSet::getTriangColors () const noexcept
{
  return trColors_;
}


//-----------------------------------------------------------------------
//   getVertexColors
//-----------------------------------------------------------------------


inline GLfloatVector TriangleSet::getVertexColors () const noexcept
{
  return vxColors_;
}


//-----------------------------------------------------------------------
//   getColorMap
//-----------------------------------------------------------------------


inline ColorMap* TriangleSet::getColorMap () const noexcept
{
  return colorMap_.get ();
}


//-----------------------------------------------------------------------
//   getNormalMode
//-----------------------------------------------------------------------


inline TriangleSet::NormalMode
  TriangleSet::getNormalMode () const noexcept
{
  return normalMode_;
}


//-----------------------------------------------------------------------
//   getNormals
//-----------------------------------------------------------------------


inline GLbyteMatrix TriangleSet::getNormals () const noexcept
{
  return normals_;
}


JEM_END_PACKAGE( gl )

#endif
