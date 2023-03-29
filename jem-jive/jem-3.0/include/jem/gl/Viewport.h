
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

#ifndef JEM_GL_VIEWPORT_H
#define JEM_GL_VIEWPORT_H

#include <jem/gl/Rect.h>
#include <jem/gl/Color.h>
#include <jem/gl/Vertex.h>
#include <jem/gl/Pivotable.h>
#include <jem/gl/RedrawRoot.h>
#include <jem/gl/WrapperNode.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class Viewport
//-----------------------------------------------------------------------


class Viewport : public WrapperNode,
                 public Pivotable,
                 public RedrawRoot
{
 public:

  JEM_DECLARE_CLASS       ( Viewport, WrapperNode );


                            Viewport

    ( const String&           name,
      const Ref<GfxNode>&     child );

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )             override;

  virtual void              configure

    ( const Properties&       props )              override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              listProps

    ( const Properties&       props )        const override;

  virtual void              setChild

    ( const Ref<GfxNode>&     child )              override;

  virtual Ref<GfxNode>      getChild      () const override;

  void                      setBgColor

    ( const Color&            bg );

  inline Color              getBgColor    () const noexcept;

  void                      setBounds

    ( const Vertex4d&         bounds );

  inline Vertex4d           getBounds     () const noexcept;
  inline Rect               getViewport   () const noexcept;


 protected:

  virtual                  ~Viewport      ();


 private:

  void                      initView_

    ( const Properties&       params );

  void                      resizeChild_

    ( RenderContext&          rtx );


 private:

  Vertex4d                  bounds_;
  Rect                      vport_;
  Rect                      vport0_;
  bool                      newVport_;
  Color                     bgColor_;
  GLbitfield                clearMask_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getBgColor
//-----------------------------------------------------------------------


inline Color Viewport::getBgColor () const noexcept
{
  return bgColor_;
}


//-----------------------------------------------------------------------
//   getBounds
//-----------------------------------------------------------------------


inline Vertex4d Viewport::getBounds () const noexcept
{
  return bounds_;
}


//-----------------------------------------------------------------------
//   getViewport
//-----------------------------------------------------------------------


inline Rect Viewport::getViewport () const noexcept
{
  return vport_;
}


JEM_END_PACKAGE( gl )

#endif
