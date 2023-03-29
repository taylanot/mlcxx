
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

#ifndef JEM_XUTIL_SNOOPDICTIONARY_H
#define JEM_XUTIL_SNOOPDICTIONARY_H

#include <jem/base/Clonable.h>
#include <jem/base/Interface.h>
#include <jem/base/array/Array.h>
#include <jem/io/Serializable.h>
#include <jem/util/Dictionary.h>
#include <jem/xutil/import.h>


JEM_BEGIN_PACKAGE( xutil )


//-----------------------------------------------------------------------
//   class SnoopDictionary
//-----------------------------------------------------------------------


class SnoopDictionary : public Dictionary,
                        public Clonable,
                        public io::Serializable
{
 public:

  JEM_DECLARE_CLASS       ( SnoopDictionary, Dictionary );

  class                     Filter;
  class                     StdFilter;

  static const int          MAX_DEPTH;


                            SnoopDictionary   ();

  explicit                  SnoopDictionary

    ( const Ref<Dict>&        dict );

  virtual void              readFrom

    ( io::ObjectInput&        in )                     override;

  virtual void              writeTo

    ( io::ObjectOutput&       out )              const override;

  virtual Ref<Object>       clone             () const override;
  virtual Ref<DictEnum>     getDictEnum       () const override;

  virtual bool              contains

    ( const String&           key )              const override;

  virtual bool              find

    ( Ref<Object>&            value,
      const String&           key )              const override;

  virtual void              insert

    ( const String&           key,
      const Ref<Object>&      value )                  override;

  virtual void              insert

    ( String&&                key,
      Ref<Object>&&           value )                  override;

  virtual void              erase

    ( const String&           key )                    override;

  virtual void              clear             ()       override;

  virtual void              reserve

    ( idx_t                   count )                  override;

  virtual void              trimToSize        ()       override;
  virtual idx_t             capacity          () const override;
  virtual idx_t             size              () const override;
  void                      sync              ();
  void                      resetCounts       ();

  void                      resetCounts

    ( const Filter&           filter );

  void                      resetCounts

    ( const String&           key );

  void                      resetReadCount

    ( const String&           key );

  void                      resetWriteCount

    ( const String&           key );

  int                       getReadCount

    ( const String&           key )              const;

  int                       getWriteCount

    ( const String&           key )              const;

  Array<String>             listObjects

    ( const Filter&           filter,
      const String&           sep = "." )        const;

  Array<String>             listTouchedObjects

    ( const String&           sep = "." )        const;

  Array<String>             listUntouchedObjects

    ( const String&           sep = "." )        const;


 protected:

  virtual                  ~SnoopDictionary   ();


 private:

  void                      initDict_         () const;

  static String             join_

    ( const String&           path,
      const String&           sep,
      const String&           name );

  void                      depthError_

    ( const char*             where )            const;


 private:

  class                     Entry_;
  class                     Map_;
  friend class              Map_;

  Ref<Dictionary>           dict_;
  Map_*                     map_;

};


//-----------------------------------------------------------------------
//   related types
//-----------------------------------------------------------------------


typedef SnoopDictionary     SnoopDict;


//-----------------------------------------------------------------------
//   class SnoopDictionary::Filter
//-----------------------------------------------------------------------


class SnoopDictionary::Filter : public Interface
{
 public:

  virtual bool              accept

    ( int                     rcount,
      int                     wcount )     const = 0;


 protected:

  virtual                  ~Filter      ();

};


//-----------------------------------------------------------------------
//   class SnoopDictionary::StdFilter
//-----------------------------------------------------------------------


class SnoopDictionary::StdFilter : public Filter
{
 public:

  enum                      Mode
  {
                              OR_MODE,
                              XOR_MODE,
                              AND_MODE
  };


  explicit                  StdFilter

    ( Mode                    mode = OR_MODE );

  virtual bool              accept

    ( int                     rcount,
      int                     wcount )     const override;


 public:

  Mode                      mode;

  int                       minReadCount;
  int                       maxReadCount;
  int                       minWriteCount;
  int                       maxWriteCount;

};


JEM_END_PACKAGE( xutil )

#endif
