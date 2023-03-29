
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

#ifndef JEM_GL_MENUNODE_H
#define JEM_GL_MENUNODE_H

#include <jem/base/array/Array.h>
#include <jem/util/event/Event.h>
#include <jem/gl/Font.h>
#include <jem/gl/Rect.h>
#include <jem/gl/Pivotable.h>
#include <jem/gl/WrapperNode.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class MenuNode
//-----------------------------------------------------------------------


class MenuNode : public WrapperNode,
                 public Pivotable
{
 public:

  JEM_DECLARE_CLASS       ( MenuNode, WrapperNode );

  util::Event
    < RenderContext&,
      const String& >       selectEvent;


                            MenuNode

    ( const String&           name,
      const Menu&             menu );

                            MenuNode

    ( const String&           name,
      const Array<String>&    list );

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params )             override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )             override;

  virtual void              setChild

    ( const Ref<GfxNode>&     child )              override;

  virtual Ref<GfxNode>      getChild      () const override;

  void                      show

    ( RenderContext&          rtx,
      const Ref<GfxNode>&     node );

  void                      show

    ( RenderContext&          rtx,
      const Ref<GfxNode>&     node,
      int                     xPos,
      int                     yPos );

  void                      hide          ();

  void                      hide

    ( RenderContext&          rtx );

  inline bool               isActive      () const noexcept;


 protected:

  virtual                  ~MenuNode      ();


 private:

  void                      show_

    ( RenderContext&          rtx,
      Pivotable&              pivot,
      GLint                   xPos,
      GLint                   yPos );

  void                      init_

    ( RenderContext&          rtx,
      GLint                   xPos,
      GLint                   yPos );

  void                      redraw_

    ( RenderContext&          rtx );

  void                      redrawV1_

    ( RenderContext&          rtx );

  void                      redrawV3_

    ( RenderContext&          rtx );

  void                      setFocus_

    ( RenderContext&          rtx,
      GLint                   xPos,
      GLint                   yPos );

  void                      moveFocus_

    ( RenderContext&          rtx,
      int                     dir );

  void                      selectFocus_

    ( RenderContext&          rtx );


 private:

  struct                    Item_
  {
    Rect                      rect;
    String                    label;
  };

  static const char*        HIDE_;
  static const idx_t        FOCUS_NONE_;

  Array<Item_>              items_;
  Font                      font_;
  Rect                      rect_;
  GLsizei                   space_;
  idx_t                     focus_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isActive
//-----------------------------------------------------------------------


inline bool MenuNode::isActive () const noexcept
{
  return (getParent() != 0);
}


JEM_END_PACKAGE( gl )

#endif
