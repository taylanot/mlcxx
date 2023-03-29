
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

#ifndef JIVE_FEM_MESHPARTMODULE_H
#define JIVE_FEM_MESHPARTMODULE_H

#include <jive/Array.h>
#include <jive/app/Module.h>
#include <jive/fem/import.h>


namespace jive
{
  namespace graph
  {
    class Partitioner;
  }
}


JIVE_BEGIN_PACKAGE( fem )


class ElementSet;


//-----------------------------------------------------------------------
//   class MeshpartModule
//-----------------------------------------------------------------------


class MeshpartModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( MeshpartModule, Module );

  typedef jive::
    graph::Partitioner      Partitioner;

  static const char*        TYPE_NAME;


  explicit                  MeshpartModule

    ( const String&           name = "mpart" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat )        override;

  static void               partition

    ( const IdxVector&        elemMap,
      const ElementSet&       elems,
      Partitioner&            parter,
      idx_t                   blockCount,
      idx_t                   minDegree = 1 );

  static void               partition

    ( const IdxVector&        elemMap,
      const IdxVector&        elemWeights,
      const ElementSet&       elems,
      Partitioner&            parter,
      idx_t                   blockCount,
      idx_t                   minDegree = 1 );

  static void               multiPartition

    ( const IdxVector&        elemMap,
      const IdxVector&        elemWeights,
      const ElementSet&       elems,
      Partitioner&            parter,
      idx_t                   blockCount,
      idx_t                   minDegree = 1 );

  static void               readElemMap

    ( const String&           fname,
      const IdxVector&        elemMap,
      const ElementSet&       elems,
      const Ref<Writer>&      logger = nullptr );

  static void               writeElemMap

    ( const String&           fname,
      const IdxVector&        elemMap,
      const ElementSet&       elems );

  static void               writeElemMap

    ( PrintWriter&            out,
      const IdxVector&        elemMap,
      const ElementSet&       elems );

  static void               getElemWeights

    ( const IdxVector&        weights,
      const ElementSet&       elems );

  static double             getUnbalance

    ( const IdxVector&        elemMap,
      const IdxVector&        elemWeights );

  static idx_t              countBorderNodes

    ( const IdxVector&        elemMap,
      const ElementSet&       elems );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare         ();


 protected:

  virtual                  ~MeshpartModule  ();


 private:

  class                     Utils_;


 private:

  static String             getElemMapTag_

    ( const ElementSet&       elems );

};


JIVE_END_PACKAGE( fem )

#endif
