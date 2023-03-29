
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


#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/util/Flex.h>
#include <jem/util/Properties.h>
#include <jem/util/SparseArray.h>
#include <jem/util/Dictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jive/util/ItemSet.h>
#include <jive/util/Assignable.h>
#include <jive/util/AllItemGroup.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/EmptyItemGroup.h>
#include <jive/geom/ShapeTable.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/XBoundarySet.h>
#include <jive/fem/XElementSet.h>
#include <jive/mesh/Names.h>
#include <jive/mesh/BndgenModule.h>


JEM_DEFINE_CLASS( jive::mesh::BndgenModule );


JIVE_BEGIN_PACKAGE( mesh )


using jem::newInstance;
using jem::dynamicCast;
using jem::Collectable;
using jem::io::Writer;
using jem::util::Flex;
using jem::util::SparseArray;
using jive::util::Topology;
using jive::util::TopoVector;
using jive::util::ItemSet;
using jive::util::ItemGroup;
using jive::geom::ShapeTable;
using jive::fem::BoundarySet;
using jive::fem::XBoundarySet;


//=======================================================================
//   class BndGenerator
//=======================================================================


class BndGenerator : public jem::Collectable
{
 public:

  enum                    Task
  {
                            NONE,
                            MAKE_ELEMENTS,
                            MAKE_BOUNDARIES
  };


                          BndGenerator

    ( const ElementSet&     elems,
      const ShapeTable&     shapes );

  void                    makeBoundaries

    ( Writer&               out,
      const XElementSet&    elems,
      const Properties&     globdat );

  void                    makeBoundaries

    ( Writer&               out,
      const XBoundarySet&   bounds,
      const Properties&     globdat );

  void                    addToGroup

    ( const String&         name,
      const Properties&     globdat );


 private:

  void                    generate_

    ( Writer&               out,
      const NodeSet&        nodes,
      const Properties&     globdat );

  void                    initBndMask_

    ( const BoundarySet&    bounds );

  Ref<ItemGroup>          newBndGroup_

    ( const IdxVector&      inodes );

  void                    addElemBounds_

    ( idx_t                 ielem,
      const Topology&       elemTopo );


 private:

  Ref<ItemSet>            items_;
  Topology                meshTopo_;
  Topology                nodeTopo_;
  TopoVector              elemTopos_;
  IdxVector               topoMap_;
  IdxVector               elemMask_;
  IdxVector               nodeMask_;
  SparseArray<idx_t,2>    bndMask_;
  Flex<idx_t>             newItems_;

  int                     task_;
  idx_t                   counter_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


BndGenerator::BndGenerator

  ( const ElementSet&  elems,
    const ShapeTable&  shapes ) :

    bndMask_ ( -1 )

{
  NodeSet      nodes     = elems.getNodes ();

  const idx_t  elemCount = elems.size ();
  const idx_t  nodeCount = nodes.size ();


  topoMap_ .resize ( elemCount );
  elemMask_.resize ( elemCount );
  nodeMask_.resize ( nodeCount );

  elemTopos_.ref   ( shapes.getShapeTopos( topoMap_ ) );

  meshTopo_ = elems    .toMatrix  ();
  nodeTopo_ = meshTopo_.transpose ();
  task_     = NONE;
  counter_  = 0;
}


//-----------------------------------------------------------------------
//   makeBoundaries
//-----------------------------------------------------------------------



void BndGenerator::makeBoundaries

  ( Writer&             out,
    const XElementSet&  elems,
    const Properties&   globdat )

{
  task_  = MAKE_ELEMENTS;
  items_ = elems.getData ();

  generate_ ( out, elems.getNodes(), globdat );
}


void BndGenerator::makeBoundaries

  ( Writer&              out,
    const XBoundarySet&  bounds,
    const Properties&    globdat )

{
  task_  = MAKE_BOUNDARIES;
  items_ = bounds.getData ();

  initBndMask_ ( bounds );
  generate_    ( out, bounds.getNodes(), globdat );
}


//-----------------------------------------------------------------------
//   generate_
//-----------------------------------------------------------------------


void BndGenerator::generate_

  ( Writer&            out,
    const NodeSet&     nodes,
    const Properties&  globdat )

{
  using jem::io::endl;
  using jem::util::Dict;
  using jem::util::DictEnum;
  using jive::util::AllItemGroup;

  Ref<Dict>   itemGroups =

    ItemGroup::getFor ( items_, globdat );

  Ref<Dict>   nodeGroups =

    ItemGroup::getFor ( nodes.getData(), globdat );

  String      itemName   = items_->getItemName ();


  elemMask_ = -1_idx;
  nodeMask_ = -1_idx;
  counter_  =  0_idx;

  newItems_.clear ();

  for ( Ref<DictEnum>
        e = nodeGroups->enumerate(); ! e->atEnd(); e->toNext() )
  {
    String  groupName = e->getKey ();

    Ref<ItemGroup>  group =

      dynamicCast<ItemGroup> ( e->getValue() );

    if ( ! group )
    {
      continue;
    }

    // Skip special node groups.

    if ( dynamicCast<AllItemGroup*>( group ) )
    {
      continue;
    }

    // Print some progress info.

    if ( counter_ == 0 )
    {
      print ( out, endl );
    }

    print ( out, "  creating ", itemName, " group `",
            groupName, "\' ...\n" );

    group = newBndGroup_ ( group->getIndices() );

    itemGroups->insert ( groupName, group );

    counter_++;
  }

  if ( counter_ )
  {
    print ( out, endl );
  }
}


//-----------------------------------------------------------------------
//   addToGroup
//-----------------------------------------------------------------------


void BndGenerator::addToGroup

  ( const String&      name,
    const Properties&  globdat )

{
  using jive::util::XItemGroup;
  using jive::util::ArrayItemGroup;

  if ( ! items_ )
  {
    return;
  }

  Ref<XItemGroup>  group =

    XItemGroup::find ( name, items_, globdat );

  const idx_t  newCount = newItems_.size ();

  BoolVector   mask     ( items_->size() );
  IdxVector    iitems   ( newCount );

  idx_t        i, j;


  mask = true;

  if ( group )
  {
    mask[group->getIndices()] = false;
  }

  for ( i = j = 0; i < newCount; i++ )
  {
    idx_t  iitem = newItems_[i];

    if ( mask[iitem] )
    {
      mask[iitem] = false;
      iitems[j++] = iitem;
    }
  }

  if ( j < newCount )
  {
    iitems.reshape ( j );
  }

  if ( ! group )
  {
    group = newInstance<ArrayItemGroup> ( iitems, items_ );

    group->store  ( name, globdat );
  }
  else
  {
    group->append ( iitems );
  }
}


//-----------------------------------------------------------------------
//   initBndMask_
//-----------------------------------------------------------------------


void BndGenerator::initBndMask_ ( const BoundarySet& bounds )
{
  using jem::shape;

  const idx_t  bndCount  = bounds   .size ();
  const idx_t  elemCount = elemMask_.size ();

  bndMask_.clear   ();
  bndMask_.reshape ( shape( elemCount, 8 ) );
  bndMask_.reserve ( bndCount + elemCount / 32 );

  for ( idx_t ibnd = 0; ibnd < bndCount; ibnd++ )
  {
    idx_t  ielem, ilocal;

    bounds.getBoundary ( ielem, ilocal, ibnd );

    bndMask_(ielem,ilocal) = ibnd;
  }
}


//-----------------------------------------------------------------------
//   newBndGroup_
//-----------------------------------------------------------------------


Ref<ItemGroup> BndGenerator::newBndGroup_

  ( const IdxVector&  inodes )

{
  using jive::util::ArrayItemGroup;
  using jive::util::EmptyItemGroup;

  IdxVector    nodeOffsets = nodeTopo_.getRowOffsets    ();
  IdxVector    elemIndices = nodeTopo_.getColumnIndices ();

  const idx_t  nodeCount   = inodes.size    ();
  const idx_t  ibegin      = newItems_.size ();


  nodeMask_[inodes] = counter_;

  for ( idx_t i = 0; i < nodeCount; i++ )
  {
    idx_t  inode = inodes[i];
    idx_t  n     = nodeOffsets[inode + 1];

    // Iterate over all elements attached to this node.

    for ( idx_t j = nodeOffsets[inode]; j < n; j++ )
    {
      idx_t  ielem = elemIndices[j];

      // Skip elements that have already been visited.

      if ( elemMask_[ielem] == counter_ )
      {
        continue;
      }

      idx_t  itopo = topoMap_[ielem];

      if ( itopo >= 0 )
      {
        addElemBounds_ ( ielem, elemTopos_[itopo] );
      }

      elemMask_[ielem] = counter_;
    }
  }

  const idx_t  iend = newItems_.size ();

  // Return the new boundary group.

  if ( iend == ibegin )
  {
    return newInstance<EmptyItemGroup> ( items_ );
  }
  else
  {
    IdxVector  iitems ( iend - ibegin );

    for ( idx_t i = ibegin; i < iend; i++ )
    {
      iitems[i - ibegin] = newItems_[i];
    }

    return newInstance<ArrayItemGroup> ( iitems, items_ );
  }
}


//-----------------------------------------------------------------------
//   addElemBounds_
//-----------------------------------------------------------------------


void BndGenerator::addElemBounds_

  ( idx_t              ielem,
    const Topology&  elemTopo )

{
  using jem::checkedCast;
  using jive::util::XGroupSet;
  using jive::util::XMemberSet;

  IdxVector    elemNodes = meshTopo_.getColumnIndices ( ielem );
  IdxVector    offsets   = elemTopo .getRowOffsets    ();
  IdxVector    bndNodes  = elemTopo .getColumnIndices ();

  const idx_t  bndCount  = elemTopo.size (0);
  const idx_t  counter   = counter_;


  // Iterate over all boundaries of the current element.

  for ( idx_t ilocal = 0; ilocal < bndCount; ilocal++ )
  {
    idx_t  ibegin = offsets[ilocal];
    idx_t  iend   = offsets[ilocal + 1];
    idx_t  k      = iend - ibegin;

    // Iterate over all nodes on this element boundary.

    for ( idx_t j = ibegin; j < iend; j++ )
    {
      idx_t  inode = elemNodes[bndNodes[j]];

      if ( nodeMask_[inode] == counter )
      {
        k--;
      }
    }

    // If all nodes on the boundary are part of the current node
    // group, then add a new boundary to the current boundary or
    // element group.

    if ( k != 0 )
    {
      continue;
    }

    if ( task_ == MAKE_ELEMENTS )
    {
      XGroupSet*  elems  = checkedCast<XGroupSet*> ( items_ );

      IdxVector   inodes ( iend - ibegin );

      for ( idx_t j = ibegin; j < iend; j++ )
      {
        inodes[j - ibegin] = elemNodes[bndNodes[j]];
      }

      newItems_.pushBack ( elems->addGroup( inodes ) );

      continue;
    }

    if ( task_ == MAKE_BOUNDARIES )
    {
      idx_t*  iptr = bndMask_.find ( ielem, ilocal );
      idx_t   ibnd;

      if ( iptr )
      {
        ibnd = *iptr;
      }
      else
      {
        XMemberSet*  bounds = checkedCast<XMemberSet*> ( items_ );

        bndMask_(ielem,ilocal) = ibnd =

          bounds->addMember ( ielem, ilocal );
      }

      newItems_.pushBack ( ibnd );
    }
  }
}


//=======================================================================
//   class BndgenModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BndgenModule::TYPE_NAME = "Bndgen";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


BndgenModule::BndgenModule ( const String& name ) :

  Super ( name )

{
  options_ = MAKE_BOUNDARIES;
}


BndgenModule::~BndgenModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status BndgenModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jem::System;
  using jive::util::getStorageMode;
  using jive::util::StorageMode;
  using jive::util::Assignable;
  using jive::fem::toXElementSet;

  if ( ! options_ )
  {
    return DONE;
  }

  Assignable<XBoundarySet>  xbounds;
  Assignable<XElementSet>   xelems;
  Assignable<ElementSet>    elems;


  if ( options_ & MAKE_BOUNDARIES )
  {
    xbounds = XBoundarySet::find ( globdat );

    if ( ! xbounds )
    {
      elems = ElementSet::find ( globdat );

      if ( elems )
      {
        Properties   myConf  = conf .makeProps ( myName_ );
        Properties   myProps = props.findProps ( myName_ );

        StorageMode  smode   = util::COMPACT_STORAGE;

        getStorageMode ( smode,   conf,   props );
        getStorageMode ( smode, myConf, myProps );

        xbounds = newXBoundarySet ( elems, smode );

        xbounds.store ( globdat );
      }
    }
    else
    {
      elems = xbounds.getElements ();
    }
  }

  if ( options_ & MAKE_ELEMENTS )
  {
    if ( ! elems )
    {
      elems = ElementSet::find ( globdat );
    }

    if ( elems )
    {
      xelems = toXElementSet ( elems );

      if ( ! xelems )
      {
        print ( System::warn(), myName_,
                " : ", elems.getContext(),
                " can not be modified\n" );
      }
    }
  }

  if ( ! elems )
  {
    print ( System::warn(), myName_,
            " : no elements have been defined\n" );

    return DONE;
  }

  Ref<ShapeTable>  st =

    ShapeTable::find ( elems.getData(), globdat );

  if ( ! st )
  {
    print ( System::warn(), myName_,
            " : no shape table has been defined\n" );

    return DONE;
  }

  print ( System::out(), "Creating boundaries ...\n" );

  Ref<BndGenerator>  bndgen =

    newInstance<BndGenerator> ( elems, *st );

  if ( xbounds )
  {
    bndgen->makeBoundaries ( System::info ( myName_ ),
                             xbounds, globdat );

    if ( addToGroup_.size() )
    {
      bndgen->addToGroup ( addToGroup_, globdat );
    }
  }

  if ( xelems )
  {
    bndgen->makeBoundaries ( System::info ( myName_ ),
                             xelems, globdat );

    if ( addToGroup_.size() )
    {
      bndgen->addToGroup ( addToGroup_, globdat );
    }
  }

  return DONE;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void BndgenModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::util::findBool;

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    findBool ( options_, MAKE_ELEMENTS,
               myProps,  PropNames::MAKE_ELEMENTS );

    findBool ( options_, MAKE_BOUNDARIES,
               myProps,  PropNames::MAKE_BOUNDARIES );

    myProps.find ( addToGroup_, PropNames::ADD_TO_GROUP );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void BndgenModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  setBool ( myConf,   PropNames::MAKE_ELEMENTS,
            options_, MAKE_ELEMENTS );

  setBool ( myConf,   PropNames::MAKE_BOUNDARIES,
            options_, MAKE_BOUNDARIES );

  myConf.set ( PropNames::ADD_TO_GROUP, addToGroup_ );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> BndgenModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void BndgenModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( mesh )
