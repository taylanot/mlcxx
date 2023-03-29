
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

#ifndef JIVE_FEM_MESHBLOCK_H
#define JIVE_FEM_MESHBLOCK_H

#include <jem/base/Object.h>
#include <jem/io/Serializable.h>
#include <jive/util/StorageMode.h>
#include <jive/mp/XBorderSet.h>
#include <jive/fem/import.h>
#include <jive/fem/typedefs.h>
#include <jive/fem/XNodeSet.h>
#include <jive/fem/XElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class MeshBlock
//-----------------------------------------------------------------------


class MeshBlock : public Object,
                  public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( MeshBlock, Object );

  enum NoOverlapTag       { NO_OVERLAP };


                            MeshBlock   ();

  explicit                  MeshBlock

    ( const String&           name,
      util::StorageMode       mode = util::DEFAULT_STORAGE );

                            MeshBlock

    ( const String&           name,
      NoOverlapTag            tag,
      util::StorageMode       mode = util::DEFAULT_STORAGE );

                            MeshBlock

    ( const XNodeSet&         nodes,
      const XElementSet&      elems,
      util::StorageMode       mode = util::DEFAULT_STORAGE );

                            MeshBlock

    ( const XNodeSet&         nodes,
      const XElementSet&      elems,
      NoOverlapTag            tag,
      util::StorageMode       mode = util::DEFAULT_STORAGE );

  virtual void              readFrom

    ( ObjectInput&            in )               override;

  virtual void              writeTo

    ( ObjectOutput&           out )        const override;

  void                      printTo

    ( PrintWriter&            out )        const;


 public:

  XNodeSet                  nodes;
  XElementSet               elems;

  Ref<ItemMask>             nodeMask;
  Ref<ItemMask>             elemMask;

  XBorderSet                recvNodeBorders;
  XBorderSet                sendNodeBorders;

  XBorderSet                recvElemBorders;
  XBorderSet                sendElemBorders;


 protected:

  virtual                  ~MeshBlock   ();


 private:

  static void               printMask_

    ( PrintWriter&            out,
      const ItemMask&         mask );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


void                        sendMeshBlock

  ( MPContext&                mpx,
    int                       rank,
    const MeshBlock&          block );

Ref<MeshBlock>              recvMeshBlock

  ( MPContext&                mpx,
    int                       rank );

void                        checkMeshBlocks

  ( const jem::Array
    < Ref<MeshBlock> >&       blocks );


JIVE_END_PACKAGE( fem )

#endif
