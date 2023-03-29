
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

#ifndef JIVE_FEM_MESHCUTTER_H
#define JIVE_FEM_MESHCUTTER_H

#include <jem/base/Flags.h>
#include <jem/base/Object.h>
#include <jem/base/Exception.h>
#include <jive/Array.h>
#include <jive/SparseMatrix.h>
#include <jive/util/StorageMode.h>
#include <jive/fem/import.h>
#include <jive/fem/ElementSet.h>


namespace jive
{
  namespace mp
  {
    class ItemMask;
    class XBorderSet;
  }
}


JIVE_BEGIN_PACKAGE( fem )


class XNodeSet;
class XElementSet;
class MeshBlock;


//-----------------------------------------------------------------------
//   class MeshCutter
//-----------------------------------------------------------------------


class MeshCutter : public Object
{
 public:

  JEM_DECLARE_CLASS       ( MeshCutter, Object );

  typedef mp::ItemMask      ItemMask;
  typedef mp::XBorderSet    XBorderSet;

  class                     Info;
  class                     Params;
  class                     Exception;

  enum                      Option
  {
                              EXPAND_FIRST = 1 << 0
  };

  typedef
    jem::Flags<Option>      Options;


  explicit                  MeshCutter

    ( const Params&           params );

                            MeshCutter

    ( const ElementSet&       elems,
      const IdxVector&        elemMap,
      idx_t                   overlap = 0 );

  void                      getInfo

    ( Info&                   info )       const;

  idx_t                     blockCount  () const;

  Ref<MeshBlock>            makeBlock

    ( idx_t                   iblock,
      util::StorageMode       mode = util::DEFAULT_STORAGE );

  void                      getBlockElems

    ( idx_t                   iblock,
      const XNodeSet&         nodes,
      const XElementSet&      elems,
      const Ref<ItemMask>&    nodeMask = nullptr,
      const Ref<ItemMask>&    elemMask = nullptr );

  void                      getBlockBorders

    ( idx_t                   iblock,
      const XBorderSet&       borders );

  void                      getBlockBorders

    ( idx_t                   iblock,
      const XBorderSet&       recvNodeBorders,
      const XBorderSet&       sendNodeBorders,
      const XBorderSet&       recvElemBorders,
      const XBorderSet&       sendElemBorders );


 protected:

  virtual                  ~MeshCutter  ();


 private:

  void                      init_

    ( const Params&           params );

  void                      getBlockElems_

    ( idx_t                   iblock,
      const XNodeSet&         nodes,
      const XElementSet&      elems,
      ItemMask*               nodeMask,
      ItemMask*               elemMask );

  static void               checkParams_

    ( const Params&           params );


 private:

  class                     Data_;
  class                     Mesh_;
  class                     Utils_;

  friend class              Utils_;

  Ref<Data_>                data_;

};


JEM_DEFINE_FLAG_OPS( MeshCutter::Options )


//-----------------------------------------------------------------------
//   class MeshCutter::Info
//-----------------------------------------------------------------------


class MeshCutter::Info
{
 public:

                            Info        ();

  void                      printTo

    ( PrintWriter&            out )        const;


 public:

  idx_t                     blockCount;
  idx_t                     overlap;

  idx_t                     minNodeCount;
  idx_t                     maxNodeCount;
  idx_t                     borderNodeCount;

  idx_t                     minElemCount;
  idx_t                     maxElemCount;
  idx_t                     borderElemCount;

};


//-----------------------------------------------------------------------
//   class MeshCutter::Params
//-----------------------------------------------------------------------


class MeshCutter::Params
{
 public:

  explicit                  Params

    ( const ElementSet&       elems );

                            Params

    ( const ElementSet&       elems,
      const IdxVector&        elemMap,
      idx_t                   overlap = 0,
      Options                 options = 0 );


 public:

  ElementSet                elems;
  IdxVector                 elemMap;
  idx_t                     overlap;
  Options                   options;

  SparseStruct              nodeCons;
  SparseStruct              elemCons;

};


//-----------------------------------------------------------------------
//   class MeshCutter::Exception
//-----------------------------------------------------------------------


class MeshCutter::Exception : public jem::Exception
{
 public:

  typedef Exception         Self;
  typedef jem::Exception    Super;


  explicit                  Exception

    ( const String&           what );

  virtual String            name  () const override;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Output>

  inline void               print

  ( Output&                   out,
    const MeshCutter::Info&   info );

void                        checkMeshBlocks

  ( MeshCutter&               mcutter );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void               print

  ( Output&                   out,
    const MeshCutter::Info&   info )

{
  info.printTo ( out );
}


JIVE_END_PACKAGE( fem )

#endif
