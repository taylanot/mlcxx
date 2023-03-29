
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

#ifndef JEM_UTIL_WEAKHASHMAP_H
#define JEM_UTIL_WEAKHASHMAP_H

#include <atomic>
#include <jem/base/Clonable.h>
#include <jem/io/Serializable.h>
#include <jem/util/ObjectMap.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class WeakHashMap
//-----------------------------------------------------------------------


class WeakHashMap : public ObjectMap,
                    public Clonable,
                    public io::Serializable
{
 public:

  JEM_DECLARE_CLASS       ( WeakHashMap, ObjectMap );


                            WeakHashMap     ();

  virtual void              readFrom

    ( io::ObjectInput&        in  )                  override;

  virtual void              writeTo

    ( io::ObjectOutput&       out )            const override;

  virtual Ref<Object>       clone           () const override;
  virtual Ref<MapEnum>      getMapEnum      () const override;

  virtual Ref<Object>       get

    ( const Ref<Object>&      key )            const override;

  virtual bool              find

    ( Ref<Object>&            value,
      const Ref<Object>&      key )            const override;

  virtual void              insert

    ( const Ref<Object>&      key,
      const Ref<Object>&      value )                override;

  virtual void              insert

    ( Ref<Object>&&           key,
      Ref<Object>&&           value )                override;

  virtual void              erase

    ( const Ref<Object>&      key )                  override;

  virtual void              clear           ()       override;

  virtual void              reserve

    ( idx_t                   count )                override;

  virtual void              trimToSize      ()       override;
  virtual idx_t             capacity        () const override;
  virtual idx_t             size            () const override;


 protected:

  virtual                  ~WeakHashMap     ();


 private:

  class                     Node_;
  class                     Enumerator_;
  friend class              Enumerator_;


 private:

  Node_*                    find_

    ( const Ref<Object>&      key )            const;

  void                      expand_         ();

  void                      realloc_

    ( idx_t                   cap );

  void                      cleanup_        ();

  inline void               checkWritable_  ();

  inline idx_t              getTableIndex_

    ( void*                   addr )           const;


 private:

  static const int          TINY_SIZE_ = 4;

  idx_t*                    table_;
  Node_*                    nodes_;

  idx_t                     capacity_;
  idx_t                     lastNode_;
  idx_t                     itemCount_;
  int                       tableBits_;

  std::atomic_int           enumCount_;
  int                       writeCount_;

};


JEM_END_PACKAGE( util )

#endif
