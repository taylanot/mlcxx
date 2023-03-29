
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

#ifndef JEM_GL_MARKERSET_H
#define JEM_GL_MARKERSET_H

#include <jem/gl/Array.h>
#include <jem/gl/LeafNode.h>


JEM_BEGIN_PACKAGE( gl )


class VertexSet;
class MarkerStyle;


//-----------------------------------------------------------------------
//   class MarkerSet
//-----------------------------------------------------------------------


class MarkerSet : public LeafNode
{
 public:

  JEM_DECLARE_CLASS       ( MarkerSet, LeafNode );

  typedef MarkerStyle       Style;


  explicit                  MarkerSet

    ( const String&           name,
      const Ref<VertexSet>&   vx );

  virtual void              getBBox

    ( Box&                    box )            const override;

  virtual void              redraw

    ( RenderContext&          rtx )                  override;

  virtual void              redrawFast

    ( RenderContext&          rtx,
      double                  lod )                  override;

  virtual bool              exeCommand

    ( RenderContext&          rtx,
      const String&           cmd,
      const Properties&       params )               override;

  virtual void              listOptions

    ( Menu&                   menu )           const override;

  virtual bool              setFont         ()       override;

  idx_t                     styleCount      () const noexcept;

  idx_t                     addStyle

    ( const String&           name,
      const Ref<Style>&       style );

  idx_t                     findStyle

    ( const String&           name )           const noexcept;

  Style*                    getStyle

    ( idx_t                   istyle )         const;

  void                      setStyle

    ( idx_t                   istyle,
      const Ref<Style>&       style );

  String                    getStyleName

    ( idx_t                   istyle )         const;

  void                      eraseStyle

    ( const String&           name );

  void                      markVertex

    ( idx_t                   ivx,
      idx_t                   istyle,
      idx_t                   tag = 0 );

  void                      markVertices

    ( const IdxVector&        ivx,
      idx_t                   istyle,
      idx_t                   tag = 0 );

  void                      markVertices

    ( const IdxVector&        ivx,
      idx_t                   istyle,
      const IdxVector&        tags );

  void                      markVertices

    ( const IdxVector&        ivx,
      const IdxVector&        istyles,
      const IdxVector&        tags );

  void                      unmarkVertex

    ( idx_t                   ivx,
      idx_t                   istyle = -1 );

  void                      unmarkVertices

    ( const IdxVector&        ivx,
      idx_t                   istyle = -1 );

  void                      unmarkAll       ();


 protected:

  virtual                  ~MarkerSet       ();


 private:

  void                      shapeChanged_   ();

  void                      redrawFastV1_

    ( RenderContext&          rtx,
      double                  lod );

  void                      redrawFastV3_

    ( RenderContext&          rtx,
      double                  lod );


 private:

  class                     Marker_;
  class                     Style_;
  class                     Data_;

  friend class              Marker_;
  friend class              Style_;
  friend class              Data_;

  Ref<Data_>                data_;

};


JEM_END_PACKAGE( gl )

#endif
