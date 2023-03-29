
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

#ifndef JEM_GL_LINESET_H
#define JEM_GL_LINESET_H

#include <jem/gl/Array.h>
#include <jem/gl/LeafNode.h>


JEM_BEGIN_PACKAGE( gl )


class ColorMap;
class GLContext;
class VertexSet;
class ShaderProg;


//-----------------------------------------------------------------------
//   class LineSet
//-----------------------------------------------------------------------


class LineSet : public LeafNode
{
 public:

  JEM_DECLARE_CLASS       ( LineSet, LeafNode );


  enum                      ColorMode
  {
                              NO_COLORS,
                              LINE_COLORS,
                              VERTEX_COLORS
  };

  static const int          SAMPLE_COUNT;


                            LineSet

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

  void                      setTopology

    ( const IdxVector&        offsets,
      const IdxVector&        indices );

  IdxVector                 getLineOffsets    () const noexcept;
  IdxVector                 getVertexIndices  () const noexcept;
  void                      setNoColors       ();

  void                      setLineColors

    ( const GLubyteMatrix&    colors );

  void                      setVertexColors

    ( const GLfloatVector&    colors,
      const Ref<ColorMap>&    colorMap );

  inline ColorMode          getColorMode      () const noexcept;
  inline GLubyteMatrix      getLineColors     () const noexcept;
  inline GLfloatVector      getVertexColors   () const noexcept;
  inline ColorMap*          getColorMap       () const noexcept;

  void                      findLine

    ( idx_t&                  index,
      GLfloat&                dist,
      GLfloat                 x,
      GLfloat                 y,
      GLfloat                 z = 0.0F )         const;


 protected:

  virtual                  ~LineSet           ();


 private:

  void                      drawNoColors2D_

    ( GLContext&             gtx )               const;

  void                      drawNoColors3D_

    ( GLContext&             gtx )               const;

  void                      drawLnColors2D_

    ( GLContext&             gtx )               const;

  void                      drawLnColors3D_

    ( GLContext&             gtx )               const;

  void                      drawVxColors2D_

    ( GLContext&             gtx )               const;

  void                      drawVxColors3D_

    ( GLContext&             gtx )               const;

  void                      blendNoColors3D_

    ( GLContext&             gtx )               const;

  void                      blendLnColors3D_

    ( GLContext&             gtx )               const;

  void                      blendVxColors3D_

    ( GLContext&             gtx )               const;

  void                      plotNoColors2D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotNoColors3D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotLnColors2D_

    ( GLContext&              gtx,
      double                  lod )              const;

  void                      plotLnColors3D_

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

  void                      checkData_

    ( const String&           context );

  void                      connectHandlers_  ();
  void                      newGeomHandler_   ();
  void                      newShapeHandler_  ();


 private:

  class                     Work_;
  class                     Buffer_;

  friend class              Work_;

  Ref<VertexSet>            vertices_;
  IdxMatrix                 segments_;
  IdxVector                 segLines_;
  IdxVector                 lnOffsets_;
  IdxVector                 vxIndices_;

  ColorMode                 colorMode_;
  GLubyteMatrix             lnColors_;
  GLfloatVector             vxColors_;
  Ref<ColorMap>             colorMap_;
  Ref<Buffer_>              buffer_;

  bool                      opaque_;
  bool                      checked_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t LineSet::size () const noexcept
{
  return (lnOffsets_.size() - 1);
}


//-----------------------------------------------------------------------
//   getColorMode
//-----------------------------------------------------------------------


inline LineSet::ColorMode LineSet::getColorMode () const noexcept
{
  return colorMode_;
}


//-----------------------------------------------------------------------
//   getLineColors
//-----------------------------------------------------------------------


inline GLubyteMatrix LineSet::getLineColors () const noexcept
{
  return lnColors_;
}


//-----------------------------------------------------------------------
//   getVertexColors
//-----------------------------------------------------------------------


inline GLfloatVector LineSet::getVertexColors () const noexcept
{
  return vxColors_;
}


//-----------------------------------------------------------------------
//   getColorMap
//-----------------------------------------------------------------------


inline ColorMap* LineSet::getColorMap () const noexcept
{
  return colorMap_.get ();
}


JEM_END_PACKAGE( gl )

#endif
