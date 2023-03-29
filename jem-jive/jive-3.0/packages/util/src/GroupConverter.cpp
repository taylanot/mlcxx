
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


#include <cctype>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/base/array/select.h>
#include <jem/numeric/sparse/SparseStructure.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/Dictionary.h>
#include <jem/util/DictionaryEnumerator.h>
#include <jem/xml/utilities.h>
#include <jive/util/typedefs.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ArrayItemGroup.h>
#include <jive/util/GroupConverter.h>


JEM_DEFINE_CLASS( jive::util::GroupConverter );


JIVE_BEGIN_PACKAGE( util )


using jem::dynamicCast;
using jem::newInstance;
using jem::IllegalInputException;
using jem::IllegalArgumentException;
using jem::xml::makeTag;

typedef jem::util::ArrayBuffer<idx_t>  IdxBuffer;


//=======================================================================
//   class GroupConverter::Utils_
//=======================================================================


class GroupConverter::Utils_
{
 public:

  static void             split

    ( String&               prefix,
      String&               name,
      const String&         source );

  static bool             checkDone

    ( const Properties&     props,
      const String&         prefix );

  static IdxVector        makeGroup

    ( const IdxVector&      aMask,
      const IdxVector&      bMask,
      const IdxVector&      ibItems,
      const Topology&       abMap,
      const Topology&       baMap,
      idx_t                 minDegree,
      idx_t                 maxDegree );

  static idx_t            makeGroups

    ( Groups_&              a,
      const Groups_&        b,
      const IdxVector&      aMask,
      const IdxVector&      bMask,
      const Topology&       abMap,
      const Topology&       baMap,
      const String&         context );

  static void             finish

    ( Groups_&              g,
      const String&         context );

};


//-----------------------------------------------------------------------
//   split
//-----------------------------------------------------------------------


void GroupConverter::Utils_::split

  ( String&        prefix,
    String&        name,
    const String&  source )

{
  const char*  s = source.addr ();

  idx_t        i, j, k, n;


  n = source.size ();
  i = n - 1;

  while ( i >= 0 && std::isspace( s[i] ) )
  {
    i--;
  }

  n = i + 1;
  i = 0;

  while ( i < n && std::isspace( s[i] ) )
  {
    i++;
  }

  j = i;

  while ( j < n && s[j] != '/' )
  {
    j++;
  }

  if ( j == n )
  {
    prefix = "";
    name   = source[slice(i,j)];

    return;
  }

  k = j + 1;
  j = j - 1;

  while ( j >= i && std::isspace( s[j] ) )
  {
    j--;
  }

  prefix = source[slice(i,j + 1)];

  while ( k < n && std::isspace( s[k] ) )
  {
    k++;
  }

  name = source[slice(k,n)];
}


//-----------------------------------------------------------------------
//   checkDone
//-----------------------------------------------------------------------


bool GroupConverter::Utils_::checkDone

  ( const Properties&  props,
    const String&      prefix )

{
  Properties     groupProps;
  String         source;
  String         name;
  String         pfx;

  bool           done;


  for ( Ref<DictEnum>
        e = props.enumerate(); ! e->atEnd(); e->toNext() )
  {
    props.get ( groupProps, e->getKey() );

    done = false;

    groupProps.find ( done, DONE_PROP_ );

    if ( done )
    {
      continue;
    }

    groupProps.get ( source, SOURCE_PROP );

    split ( pfx, name, source );

    if ( pfx == prefix )
    {
      return false;
    }
  }

  return true;
}


//-----------------------------------------------------------------------
//   makeGroup
//-----------------------------------------------------------------------


IdxVector GroupConverter::Utils_::makeGroup

  ( const IdxVector&  aMask,
    const IdxVector&  bMask,
    const IdxVector&  bItems,
    const Topology&   abMap,
    const Topology&   baMap,
    idx_t             minDegree,
    idx_t             maxDegree )

{
  const idx_t  bItemCount = bItems.size ();

  IdxVector    aOffsets   = abMap.getRowOffsets    ();
  IdxVector    bIndices   = abMap.getColumnIndices ();
  IdxVector    bOffsets   = baMap.getRowOffsets    ();
  IdxVector    aIndices   = baMap.getColumnIndices ();

  IdxVector    aItems;
  IdxBuffer    ibuf;

  idx_t        deg, mindeg, maxdeg;

  idx_t        ia, ib, ii;
  idx_t        i,  j,  m, n;


  ibuf.reserve ( bItems.size() );

  bMask[bItems] = 1;

  for ( ii = 0; ii < bItemCount; ii++ )
  {
    ib = bItems[ii];
    i  = bOffsets[ib];
    m  = bOffsets[ib + 1];

    for ( ; i < m; i++ )
    {
      ia = aIndices[i];

      if ( aMask[ia] )
      {
        continue;
      }

      deg = 0;
      j   = aOffsets[ia];
      n   = aOffsets[ia + 1];

      if ( minDegree <= 0 )
      {
        mindeg = n - j + minDegree;
      }
      else
      {
        mindeg = minDegree;
      }

      if ( maxDegree <= 0 )
      {
        maxdeg = n - j + maxDegree;
      }
      else
      {
        maxdeg = maxDegree;
      }

      for ( ; j < n; j++ )
      {
        deg += bMask[bIndices[j]];
      }

      if ( deg >= mindeg && deg <= maxdeg )
      {
        ibuf.pushBack ( ia );

        aMask[ia] = 1;
      }
    }
  }

  ibuf.trimToSize ();

  aItems.ref ( ibuf.toArray() );

  aMask[aItems] = 0;
  bMask[bItems] = 0;

  return aItems;
}


//-----------------------------------------------------------------------
//   makeGroups
//-----------------------------------------------------------------------


idx_t GroupConverter::Utils_::makeGroups

  ( Groups_&          a,
    const Groups_&    b,
    const IdxVector&  aMask,
    const IdxVector&  bMask,
    const Topology&   abMap,
    const Topology&   baMap,
    const String&     context )

{
  Properties      groupConf  = a.conf;
  Properties      groupProps = a.props;

  Ref<ItemGroup>  bGroup;

  String          propName;
  String          aName, bName;
  String          prefix;
  String          source;

  bool            allDone;
  bool            done;
  bool            temp;

  idx_t           minDegree;
  idx_t           maxDegree;

  idx_t           n;


  n       = 0;
  allDone = true;

  for ( Ref<DictEnum>
        e = a.props.enumerate(); ! e->atEnd(); e->toNext() )
  {
    propName = e->getKey ();
    aName    = propName;
    done     = false;

    a.props.get ( groupProps, propName );

    groupProps.find ( done, DONE_PROP_ );

    if ( done )
    {
      continue;
    }

    groupProps.find ( aName, NAME_PROP );

    if ( a.groups->contains( aName ) )
    {
      String  itemName = a.items->getItemName ();

      throw IllegalInputException (
        context,
        String::format ( "%s group `%s\' already exists",
                         itemName,
                         aName )
      );
    }

    groupProps.get ( source, SOURCE_PROP );

    split ( prefix, bName, source );

    if ( prefix != b.prefix )
    {
      continue;
    }

    bGroup = dynamicCast<ItemGroup> ( b.groups->get( bName ) );

    if ( bGroup == nullptr || bGroup->getItems() != b.items.get() )
    {
      allDone = false;
      continue;
    }

    temp      = false;
    minDegree = 0;
    maxDegree = jem::maxOf ( maxDegree );

    groupProps.find ( temp,      TEMP_PROP   );
    groupProps.find ( minDegree, MINDEG_PROP );
    groupProps.find ( maxDegree, MAXDEG_PROP );

    a.groups->insert (
      aName,
      newInstance<ArrayItemGroup> (
        Utils_::makeGroup (
          aMask,     bMask,
          bGroup->getIndices (),
          abMap,     baMap,
          minDegree, maxDegree
        ),
        a.items
      )
    );

    n++;

    groupConf = a.conf.makeProps ( propName );
    done      = true;

    groupProps.set ( DONE_PROP_,  done      );
    groupConf .set ( NAME_PROP,   aName     );
    groupConf .set ( SOURCE_PROP, source    );
    groupConf .set ( TEMP_PROP,   temp      );
    groupConf .set ( MINDEG_PROP, minDegree );
    groupConf .set ( MAXDEG_PROP, maxDegree );
  }

  a.done = allDone;

  return n;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void GroupConverter::Utils_::finish

  ( Groups_&       g,
    const String&  context )

{
  Properties     groupProps = g.props;

  Ref<DictEnum>  e;

  String         name;
  bool           temp;
  bool           done;


  for ( e = g.props.enumerate(); ! e->atEnd(); e->toNext() )
  {
    name = e->getKey ();
    done = false;

    g.props.get ( groupProps, name );

    groupProps.find ( done, DONE_PROP_ );

    if ( ! done )
    {
      String  itemName = g.items->getItemName ();

      groupProps.find ( name, NAME_PROP );

      throw IllegalInputException (
        context,
        String::format (
          "failed to create %s group `%s\' "
          "(invalid source group or circular dependency)",
          itemName,
          name
        )
      );
    }

    temp = false;

    groupProps.find ( temp, TEMP_PROP );

    if ( temp )
    {
      groupProps.find ( name, NAME_PROP );
      g.groups->erase ( name );
    }
  }
}


//=======================================================================
//   class GroupConverter::Groups_
//=======================================================================


inline GroupConverter::Groups_::Groups_ () :

  props ( conf ),
  done  ( true )

{}


//=======================================================================
//   class GroupConverter
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  GroupConverter::NAME_PROP   = "name";
const char*  GroupConverter::TEMP_PROP   = "temp";
const char*  GroupConverter::SOURCE_PROP = "source";
const char*  GroupConverter::MINDEG_PROP = "minDegree";
const char*  GroupConverter::MAXDEG_PROP = "maxDegree";
const char*  GroupConverter::DONE_PROP_  = "$done";


//-----------------------------------------------------------------------
//   constructors & detsructor
//-----------------------------------------------------------------------


GroupConverter::GroupConverter

  ( const Ref<ItemSet>&  aItems,
    const Ref<ItemSet>&  bItems,
    const Properties&    globdat,
    const String&        context ) :

    context_ ( context )

{
  JEM_PRECHECK2 ( aItems && bItems, "NULL ItemSet" );

  init_ ( aItems, ItemGroup::getFor( aItems, globdat ),
          bItems, ItemGroup::getFor( bItems, globdat ) );
}


GroupConverter::GroupConverter

  ( const Ref<ItemSet>&  aItems,
    const Ref<Dict>&     aGroups,
    const Ref<ItemSet>&  bItems,
    const Ref<Dict>&     bGroups,
    const String&        context ) :

    context_ ( context )

{
  JEM_PRECHECK2 ( aItems && aGroups &&
                  bItems && bGroups,
                  "NULL reference" );

  init_ ( aItems, aGroups, bItems, bGroups );
}


GroupConverter::~GroupConverter ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


void GroupConverter::init

  ( const Properties&  conf,
    const Properties&  props )

{
  String  propNames[2];

  for ( int i = 0; i < 2; i++ )
  {
    String  tag  = makeTag ( groups_[i].items->getItemName(), false );

    propNames[i] = tag + "Groups";
  }

  init ( conf .makeProps ( propNames[0] ),
         props.findProps ( propNames[0] ),
         conf .makeProps ( propNames[1] ),
         props.findProps ( propNames[1] ) );
}


void GroupConverter::init

  ( const Properties&  aConf,
    const Properties&  aProps,
    const Properties&  bConf,
    const Properties&  bProps )

{
  Groups_&  a = groups_[0];
  Groups_&  b = groups_[1];

  a.conf      = aConf;
  a.props     = aProps;
  a.done      = Utils_::checkDone ( a.props, b.prefix );

  b.conf      = bConf;
  b.props     = bProps;
  b.done      = Utils_::checkDone ( b.props, a.prefix );
}


//-----------------------------------------------------------------------
//   isDone
//-----------------------------------------------------------------------


bool GroupConverter::isDone () const
{
  return (groups_[0].done && groups_[1].done);
}


//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


idx_t GroupConverter::convert

  ( const Topology&  abMap,
    const Topology&  baMap )

{
  JEM_PRECHECK2 ( abMap.size(0) == baMap.size(1) &&
                  abMap.size(1) == baMap.size(0),
                  "Topology shape mismatch" );

  Groups_&     a          = groups_[0];
  Groups_&     b          = groups_[1];

  const idx_t  aItemCount = a.items->size ();
  const idx_t  bItemCount = b.items->size ();

  IdxVector    aMask      ( aItemCount );
  IdxVector    bMask      ( bItemCount );

  int          iter;
  idx_t        k, n;


  if ( abMap.size(0) != aItemCount ||
       baMap.size(0) != bItemCount )
  {
    throw IllegalArgumentException (
      JEM_FUNC,
      "group mapping matrices have an invalid size"
    );
  }

  aMask = 0;
  bMask = 0;
  n     = 0;

  for ( iter = 0; iter < 8; iter++ )
  {
    k  = Utils_::makeGroups ( a,     b,
                              aMask, bMask,
                              abMap, baMap,
                              context_ );

    k += Utils_::makeGroups ( b,     a,
                              bMask, aMask,
                              baMap, abMap,
                              context_ );

    n += k;

    if ( k == 0 )
    {
      break;
    }
  }

  return n;
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void GroupConverter::finish ()
{
  Utils_::finish ( groups_[0], context_ );
  Utils_::finish ( groups_[1], context_ );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void GroupConverter::init_

  ( const Ref<ItemSet>&  aItems,
    const Ref<Dict>&     aGroups,
    const Ref<ItemSet>&  bItems,
    const Ref<Dict>&     bGroups )

{
  Groups_*  g = groups_;


  g[0].items  = aItems;
  g[0].groups = aGroups;
  g[1].items  = bItems;
  g[1].groups = bGroups;

  for ( int i = 0; i < 2; i++ )
  {
    g[i].prefix = makeTag ( g[i].items->getItemsName(), false );
    g[i].done   = false;
  }

  if ( context_.size() == 0 )
  {
    String  name0 = g[0].items->getItemName ();
    String  name1 = g[1].items->getItemName ();

    context_ = String::format ( "%s <-> %s group converter",
                                name0,
                                name1 );
  }
}


JIVE_END_PACKAGE( util )
