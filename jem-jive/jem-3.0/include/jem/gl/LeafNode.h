
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

#ifndef JEM_GL_LEAFNODE_H
#define JEM_GL_LEAFNODE_H

#include <jem/gl/GfxNode.h>


JEM_BEGIN_PACKAGE( gl )


class Menu;


//-----------------------------------------------------------------------
//   class LeafNode
//-----------------------------------------------------------------------


class LeafNode : public GfxNode
{
 public:

  JEM_DECLARE_CLASS       ( LeafNode, GfxNode );


  explicit                  LeafNode

    ( const String&           name = "" );

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params )           override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )           override;

  virtual void              getVisibility

    ( bool&                   viz,
      Box&                    vbox,
      const Transform&        proj )       const override;

  virtual void              configure

    ( const Properties&       props )            override;

  virtual void              getConfig

    ( const Properties&       props )      const override;

  virtual void              listProps

    ( const Properties&       props )      const override;

  virtual void              redraw

    ( RenderContext&          rtx )              = 0;

  virtual void              redrawFast

    ( RenderContext&          rtx,
      double                  lod )              = 0;

  virtual bool              initGL

    ( RenderContext&          rtx );

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd,
      const Properties&       params );

  virtual void              listOptions

    ( Menu&                   menu )       const;

  virtual bool              setFont     ();


 public:

  bool                      hidden;


 protected:

  virtual                  ~LeafNode    ();

};


JEM_END_PACKAGE( gl )

#endif
