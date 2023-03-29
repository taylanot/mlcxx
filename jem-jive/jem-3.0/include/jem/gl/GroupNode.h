
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

#ifndef JEM_GL_GROUPNODE_H
#define JEM_GL_GROUPNODE_H

#include <jem/gl/MultiNode.h>
#include <jem/gl/Transformable.h>


JEM_BEGIN_PACKAGE( gl )


class RenderContext;


//-----------------------------------------------------------------------
//   class GroupNode
//-----------------------------------------------------------------------


class GroupNode : public MultiNode,
                  public Transformable
{
 public:

  JEM_DECLARE_CLASS       ( GroupNode, MultiNode );


  enum                      Rank
  {
                              RANK2D = 2,
                              RANK3D = 3
  };


  explicit                  GroupNode

    ( const String&           name    = "",
      Rank                    rank    = RANK3D );

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params )             override;

  virtual bool              takeGLAction

    ( RenderContext&          rtx,
      int                     action,
      const Properties&       params )             override;

  virtual void              getBBox

    ( Box&                    box )          const override;

  virtual void              getBareBBox

    ( Box&                    box )          const override;

  virtual void              getVisibility

    ( bool&                   viz,
      Box&                    vbox,
      const Transform&        proj )         const override;

  virtual void              configure

    ( const Properties&       props )              override;

  virtual void              getConfig

    ( const Properties&       props )        const override;

  virtual void              listProps

    ( const Properties&       props )        const override;

  virtual Transform         getTransform  () const override;

  virtual void              setTransform

    ( const Transform&        tr )                 override;


 public:

  bool                      hidden;


 protected:

  virtual                  ~GroupNode     ();


 private:

  const Rank                rank_;

  Transform                 transfm_;

};


JEM_END_PACKAGE( gl )

#endif
