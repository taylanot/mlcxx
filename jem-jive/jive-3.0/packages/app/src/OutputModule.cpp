
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


#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/Array.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Integer.h>
#include <jem/base/StringBuffer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/utilities.h>
#include <jem/io/IOException.h>
#include <jem/io/FileOutputStream.h>
#include <jem/io/GzipOutputStream.h>
#include <jem/mp/utilities.h>
#include <jem/mp/GatherPrinter.h>
#include <jem/util/Properties.h>
#include <jem/util/StringUtils.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/SparseTable.h>
#include <jive/util/DofSpace.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/VectorManager.h>
#include <jive/model/Actions.h>
#include <jive/model/Model.h>
#include <jive/mp/Globdat.h>
#include <jive/app/import.h>
#include <jive/app/typedefs.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/StdDataPrinter.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/OutputModule.h>


JEM_DEFINE_CLASS( jive::app::OutputModule );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;
using jem::ParseException;
using jem::io::IOException;
using jem::io::PrintWriter;
using jem::mp::GatherPrinter;
using jem::util::StringUtils;
using jive::util::FuncUtils;
using jive::util::ItemSet;
using jive::util::ItemGroup;
using jive::util::XTable;


//=======================================================================
//   class OutputModule::Specs_
//=======================================================================


class OutputModule::Specs_ : public jem::Collectable
{
 public:

  explicit                  Specs_

    ( const StringVector&     specs );

  StringVector              getSpecs  () const;


 public:

  StringVector              labels;
  StringVector              names;
  StringVector              paths;
  StringVector              groups;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


OutputModule::Specs_::Specs_

  ( const StringVector&  specs )

{
  const idx_t  n = specs.size ();

  String       s;
  idx_t        i, j;


  labels.resize ( n );
  names .resize ( n );
  paths .resize ( n );
  groups.resize ( n );

  for ( i = 0; i < n; i++ )
  {
    s = specs[i].stripWhite ();
    j = s.find ( '=' );

    if ( j >= 0 )
    {
      labels[i] = s[slice(j + 1,END)].stripWhite ();
      s         = s[slice(BEGIN,j)]  .stripWhite ();
    }

    if ( s.endsWith( ']' ) )
    {
      j = s.rfind ( '[' );

      if ( j < 0 )
      {
        throw ParseException (
          JEM_FUNC,
          String::format (
            "invalid name: `%s\' (missing `]\')", s
          )
        );
      }

      groups[i] = s[slice(j + 1,s.size() - 1)].stripWhite ();
      s         = s[slice(BEGIN,j)]           .stripWhite ();
    }
    else
    {
      groups[i] = "";
    }

    j = s.find ( '/' );

    if ( j < 0 )
    {
      paths[i] = "";
    }
    else
    {
      paths[i] = s[slice(BEGIN,j)]  .stripWhite ();
      s        = s[slice(j + 1,END)].stripWhite ();
    }

    if ( ! s.size() )
    {
      throw ParseException ( JEM_FUNC, "empty name" );
    }

    names[i] = s;

    if ( ! labels[i].size() )
    {
      if ( groups[i].size() )
      {
        labels[i] = String::format ( "%s[%s]",
                                     names[i], & groups[i] );
      }
      else
      {
        labels[i] = names[i];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getSpecs
//-----------------------------------------------------------------------


StringVector OutputModule::Specs_::getSpecs () const
{
  using jem::StringBuffer;

  const idx_t   n     = names.size ();

  StringVector  specs ( n );
  StringBuffer  buf;


  for ( idx_t i = 0; i < n; i++ )
  {
    buf.clear ();

    if ( paths[i].size() )
    {
      print ( buf, paths[i], '/' );
    }

    buf += names[i];

    if ( groups[i].size() )
    {
      print ( buf, '[', groups[i], ']' );
    }

    print ( buf, " = ", labels[i] );

    specs[i] = buf.toString ();
  }

  return specs;
}


//=======================================================================
//   class OutputModule::RunData_
//=======================================================================


class OutputModule::RunData_ : public jem::Collectable
{
 public:

                            RunData_

    ( const String&           context,
      const Properties&       globdat );

  void                      openFile

    ( const String&           name,
      OpenFlags               flags,
      int                     prec,
      int                     width,
      idx_t                   bufsize );

  inline bool               isOpen      () const;
  inline void               flushFile   ();
  inline void               closeFile   ();

  inline void               setPrecision

    ( int                     prec );

  inline void               setPageWidth

    ( int                     width );

  void                      printVectors

    ( DataPrinter&            printer,
      const Specs_&           specs,
      const Properties&       globdat );

  void                      printTables

    ( DataPrinter&            printer,
      const Specs_&           specs,
      const Properties&       globdat );


 public:

  const String              context;

  Ref<MPContext>            mpContext;

  idx_t                     istep;
  int                       warnings;


 private:

  void                      warning_

    ( const String&           msg );

  inline void               setIntWidth_

    ( idx_t                   imax );

  Ref<XTable>               getTableFor_

    ( ItemSet*                items );

  void                      getSubVector_

    ( const Vector&           vec,
      const IdxVector&        iitems,
      const DofSpace&         dofs );


 private:

  static const int          TABLE_CACHE_SIZE_ = 4;

  Ref<GatherPrinter>        output_;
  String                    fileName_;
  Ref<XTable>               tableCache_[TABLE_CACHE_SIZE_];
  Properties                params_;
  Vector                    subvec_;
  IdxVector                 idofs_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


OutputModule::RunData_::RunData_

  ( const String&      ctx,
    const Properties&  globdat ) :

    context ( ctx ),
    params_ ( "actionParams" )

{
  mpContext = mp::Globdat::getMPContext ( globdat );
  istep     = 0;
  warnings  = 0;
}


//-----------------------------------------------------------------------
//   openFile
//-----------------------------------------------------------------------


void OutputModule::RunData_::openFile

  ( const String&  fname,
    OpenFlags      flags,
    int            prec,
    int            width,
    idx_t          bufsize )

{
  using jem::io::NumberFormat;
  using jem::io::OutputStream;
  using jem::io::FileOutputStream;
  using jem::io::GzipOutputStream;

  if ( fname == fileName_ )
  {
    return;
  }

  try
  {
    Ref<OutputStream>  out;
    NumberFormat       nf;

    if ( output_ )
    {
      output_->close ();

      output_   = nullptr;
      fileName_ = "";
    }

    if ( ! fname.size() )
    {
      return;
    }

    if ( mpContext->myRank() == 0 )
    {
      out = newInstance<FileOutputStream> ( fname, flags );

      if ( util::isGzipFile( fname ) )
      {
        out = newInstance<GzipOutputStream> ( out );
      }
    }

    output_ = GatherPrinter::open ( 0, mpContext, out, bufsize );

    nf.setScientific     ();
    nf.setFractionDigits ( prec );
    nf.setFloatWidth     ( prec + 8 );

    output_->nformat = nf;

    output_->setPageWidth ( width );

    fileName_ = fname;
  }
  catch ( IOException& ex )
  {
    ex.setContext ( context );
    throw;
  }
}


//-----------------------------------------------------------------------
//   isOpen
//-----------------------------------------------------------------------


inline bool OutputModule::RunData_::isOpen () const
{
  return (output_ != nullptr);
}


//-----------------------------------------------------------------------
//   flushFile
//-----------------------------------------------------------------------


inline void OutputModule::RunData_::flushFile ()
{
  if ( output_ )
  {
    try
    {
      output_->flush ();
    }
    catch ( IOException& ex )
    {
      ex.setContext ( context );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   closeFile
//-----------------------------------------------------------------------


inline void OutputModule::RunData_::closeFile ()
{
  if ( output_ )
  {
    try
    {
      output_->close ();
    }
    catch ( const IOException& ex )
    {
      throw IOException ( context, ex.what() );
    }

    output_   = nullptr;
    fileName_ = "";
  }
}


//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------


inline void OutputModule::RunData_::setPrecision ( int prec )
{
  if ( output_ )
  {
    output_->nformat.setFloatWidth     ( prec + 8 );
    output_->nformat.setFractionDigits ( prec );
  }
}


//-----------------------------------------------------------------------
//   setPageWidth
//-----------------------------------------------------------------------


inline void OutputModule::RunData_::setPageWidth ( int width )
{
  if ( output_ )
  {
    output_->setPageWidth ( width );
  }
}


//-----------------------------------------------------------------------
//   printVectors
//-----------------------------------------------------------------------


void OutputModule::RunData_::printVectors

  ( DataPrinter&       printer,
    const Specs_&      specs,
    const Properties&  globdat )

{
  using jive::util::VectorManager;

  if ( ! output_ )
  {
    return;
  }

  const idx_t         vecCount = specs.names.size ();

  Ref<VectorManager>  vman;
  Ref<ItemGroup>      igroup;
  Ref<ItemSet>        items;
  Ref<DofSpace>       dofs;

  Vector              vec;

  String              path;
  String              name;
  String              label;

  idx_t               ivec;


  for ( ivec = 0; ivec < vecCount; ivec++ )
  {
    if ( ! dofs || specs.paths[ivec] != path )
    {
      path = specs.paths[ivec];

      if ( ! path.size() )
      {
        dofs  = DofSpace::get  ( globdat, context );
        items = dofs->getItems ();
      }
      else
      {
        items = ItemSet ::get ( path,  globdat, context );
        dofs  = DofSpace::get ( items, globdat, context );
      }

      vman = VectorManager::get ( dofs, globdat );

      setIntWidth_ ( items->size() );
    }

    name = specs.names[ivec];

    if ( ! vman->findVector( vec, name ) )
    {
      warning_ (
        String::format ( "undefined vector: `%s\'", name )
      );

      continue;
    }

    label = StringUtils::expand ( specs.labels[ivec], istep );
    name  = specs.groups[ivec];

    try
    {
      if ( ! name.size() )
      {
        printer.printVector ( *output_, label, vec,
                              *dofs,    globdat );
      }
      else
      {
        igroup = ItemGroup::get ( name, items, globdat, context );

        getSubVector_ ( vec, igroup->getIndices(), *dofs );

        printer.printSparseVector ( *output_, label,
                                    subvec_,  idofs_,
                                    *dofs,    globdat );
      }
    }
    catch ( IOException& ex )
    {
      ex.setContext ( context );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   printTables
//-----------------------------------------------------------------------


void OutputModule::RunData_::printTables

  ( DataPrinter&       printer,
    const Specs_&      specs,
    const Properties&  globdat )

{
  using jem::where;
  using jem::Limits;
  using jive::model::Model;
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( ! output_ )
  {
    return;
  }

  const idx_t     tabCount = specs.names.size ();

  Ref<Model>      model;
  Ref<DofSpace>   dofs;
  Ref<ItemSet>    items;
  Ref<ItemGroup>  igroup;
  Ref<Table>      table;

  Vector          weights;

  String          name;
  String          path;
  String          label;

  bool            updated;
  idx_t           itab;


  for ( itab = 0; itab < tabCount; itab++ )
  {
    if ( ! items || specs.paths[itab] != path )
    {
      path = specs.paths[itab];

      if ( ! path.size() )
      {
        dofs  = DofSpace::get  ( globdat, context );
        items = dofs->getItems ();
      }
      else
      {
        items = ItemSet::get   ( path, globdat, context );
      }

      setIntWidth_ ( items->size() );
    }

    name   = specs.groups[itab];
    igroup = nullptr;

    if ( name.size() )
    {
      igroup = ItemGroup::get ( name, items, globdat, context );
    }

    name  = specs.names[itab];
    table = Table::find ( name, items, globdat );

    if ( ! table )
    {
      Ref<XTable>  xtable = getTableFor_ ( items.get() );


      weights.resize ( items->size() );

      weights = 0.0;
      model   = Model::get ( globdat, context );

      xtable->clearAll ();

      params_.set ( ActionParams::TABLE,         xtable  );
      params_.set ( ActionParams::TABLE_NAME,    name    );
      params_.set ( ActionParams::TABLE_WEIGHTS, weights );

      if ( igroup )
      {
        params_.set   ( ActionParams::TABLE_ROWS, igroup );
      }
      else
      {
        params_.erase ( ActionParams::TABLE_ROWS );
      }

      updated = model->takeAction ( Actions::GET_TABLE,
                                    params_, globdat );

      params_.erase ( ActionParams::TABLE         );
      params_.erase ( ActionParams::TABLE_WEIGHTS );

      if ( ! updated )
      {
        warning_ (
          String::format (
            "table `%s\' has not been updated (skipping)", name
          )
        );

        continue;
      }

      weights = where ( abs( weights ) < Limits<double>::TINY_VALUE,
                        1.0, 1.0 / weights );
      table   = xtable;

      xtable->scaleRows ( weights );
    }

    label = StringUtils::expand ( specs.labels[itab], istep );

    try
    {
      printer.printTable ( *output_, label, *table, globdat );
    }
    catch ( IOException& ex )
    {
      ex.setContext ( context );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   warning_
//-----------------------------------------------------------------------


void OutputModule::RunData_::warning_ ( const String& msg )
{
  using jem::System;

  const int  MAX_WARNINGS = 20;


  if ( warnings > MAX_WARNINGS )
  {
    return;
  }

  if ( warnings < MAX_WARNINGS )
  {
    print ( System::warn(), context, " : ", msg, "\n" );
  }
  else
  {
    print ( System::warn(), context,
            " : no more warnings will be printed\n" );
  }

  warnings++;
}


//-----------------------------------------------------------------------
//   setIntWidth_
//-----------------------------------------------------------------------


inline void OutputModule::RunData_::setIntWidth_ ( idx_t imax )
{
  output_->nformat.setIntegerWidth ( jem::io::getWidthOf( imax ) );
}


//-----------------------------------------------------------------------
//   getTableFor_
//-----------------------------------------------------------------------


Ref<XTable> OutputModule::RunData_::getTableFor_ ( ItemSet* items )
{
  using jive::util::SparseTable;

  int  i;


  for ( i = 0; i < TABLE_CACHE_SIZE_; i++ )
  {
    if ( tableCache_[i] &&
         tableCache_[i]->getRowItems() == items )
    {
      return tableCache_[i];
    }
  }

  Ref<XTable>  newTable =

    newInstance<SparseTable> ( "output", items );

  for ( i = 0; i < TABLE_CACHE_SIZE_; i++ )
  {
    if ( ! tableCache_[i] )
    {
      tableCache_[i] = newTable;

      return newTable;
    }
  }

  // Pseudo random index (ahem).

  i = (int) ((lint) newTable.get() % (lint) TABLE_CACHE_SIZE_);

  tableCache_[i] = newTable;

  return newTable;
}


//-----------------------------------------------------------------------
//   getSubVector_
//-----------------------------------------------------------------------


void OutputModule::RunData_::getSubVector_

  ( const Vector&     vec,
    const IdxVector&  iitems,
    const DofSpace&   dofs )

{
  const idx_t  iitemCount = iitems.size      ();
  const idx_t  dofCount   = vec   .size      ();
  const idx_t  typeCount  = dofs  .typeCount ();

  IdxVector    itypes;

  idx_t        i, k, n;


  idofs_.resize ( dofCount + typeCount );
  itypes.ref    ( idofs_[slice(dofCount,END)] );

  n = 0;

  for ( i = 0; i < iitemCount; i++ )
  {
    k  = dofs.getDofsForItem ( idofs_[slice(n,dofCount)],
                               itypes,
                               iitems[i] );
    n += k;
  }

  idofs_ .ref    ( idofs_[slice(BEGIN,n)] );
  subvec_.resize ( n );

  subvec_ = vec[idofs_];
}


//=======================================================================
//   class OutputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  OutputModule::TYPE_NAME = "Output";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


OutputModule::OutputModule

  ( const String&     name,
    Ref<DataPrinter>  printer ) :

    Super    (    name ),
    printer_ ( printer )

{
  if ( ! printer_ )
  {
    printer_ = newInstance<StdDataPrinter> (
      util::joinNames ( myName_, "printer" )
    );
  }

  fflags_    = 0;
  precision_ = 6;
  pageWidth_ = 200;
  bufsize_   = -1_idx;
  saveCond_  = FuncUtils::newCond ();
}


OutputModule::~OutputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status OutputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  rundat_ = nullptr;

  printer_->init ( conf, props, globdat );

  // No errors, so switch to running mode

  rundat_ = newInstance<RunData_> ( getContext(), globdat );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status OutputModule::run ( const Properties& globdat )
{
  using jem::mp::allmax;
  using jive::util::Globdat;

  const int  SAVE_VECTORS = 1 << 0;
  const int  SAVE_TABLES  = 1 << 1;

  if ( ! rundat_ )
  {
    return DONE;
  }

  RunData_&  d = * rundat_;

  int  todo = 0;

  try
  {
    todo = (int) FuncUtils::evalCond ( *saveCond_, globdat );
  }
  catch ( jem::Exception& ex )
  {
    ex.setContext ( d.context );
    throw;
  }

  if ( ! allmax( *d.mpContext, todo ) )
  {
    return OK;
  }

  todo = 0;

  if ( vecSpecs_ )
  {
    todo |= SAVE_VECTORS;
  }

  if ( tabSpecs_ )
  {
    todo |= SAVE_TABLES;
  }

  if ( todo )
  {
    globdat.find ( d.istep, Globdat::TIME_STEP );

    String  fname = StringUtils::expand ( fileName_, d.istep );

    d.openFile ( fname, fflags_, precision_, pageWidth_, bufsize_ );

    if ( todo & SAVE_VECTORS )
    {
      d.printVectors ( *printer_, *vecSpecs_, globdat );
    }

    if ( todo & SAVE_TABLES )
    {
      d.printTables  ( *printer_, *tabSpecs_, globdat );
    }

    d.flushFile ();
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void OutputModule::shutdown ( const Properties& globdat )
{
  if ( rundat_ )
  {
    rundat_->closeFile ();
    rundat_ = nullptr;

    printer_->shutdown ( globdat );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void OutputModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;

  if ( ! props.contains( myName_ ) )
  {
    return;
  }

  Properties    myProps = props.findProps ( myName_ );

  String        fname   = fileName_;
  bool          append  = (fflags_ & FileFlags::APPEND) != 0;

  StringVector  specs;
  OpenFlags     flags;


  myProps.find ( append,   PropNames::APPEND  );
  myProps.find ( fname,    PropNames::FILE    );
  myProps.find ( bufsize_, PropNames::BUFSIZE );

  flags = 0;

  if ( append )
  {
    flags |= FileFlags::APPEND;
  }

  checkFile_ ( fname, flags );

  if ( fname != fileName_ && rundat_ )
  {
    rundat_->closeFile ();
  }

  // Data are valid, so commit the changes

  fileName_ = fname;
  fflags_   = flags;

  if ( myProps.find( precision_, PropNames::PRECISION, 0, 20 ) )
  {
    if ( rundat_ )
    {
      rundat_->setPrecision ( precision_ );
    }
  }

  if ( myProps.find( pageWidth_, PropNames::PAGE_WIDTH,
                     4,          maxOf( pageWidth_ ) ) )
  {
    if ( rundat_ )
    {
      rundat_->setPageWidth ( pageWidth_ );
    }
  }

  if ( myProps.find( specs, PropNames::VECTORS ) )
  {
    Ref<Specs_>  tmp;

    try
    {
      tmp = newInstance<Specs_> ( specs );
    }
    catch ( ParseException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::VECTORS ) );
      throw;
    }

    vecSpecs_ = tmp;

    if ( rundat_ )
    {
      rundat_->warnings = 0;
    }
  }

  if ( myProps.find( specs, PropNames::TABLES ) )
  {
    Ref<Specs_>  tmp;

    try
    {
      tmp = newInstance<Specs_> ( specs );
    }
    catch ( ParseException& ex )
    {
      ex.setContext ( myProps.getContext( PropNames::TABLES ) );
      throw;
    }

    for ( idx_t i = 0; i < tmp->paths.size(); i++ )
    {
      if ( ! tmp->paths[i].size() )
      {
        myProps.propertyError (
          PropNames::TABLES,
          String::format (
            "no item set specified for table `%s\'",
            tmp->names[i]
          )
        );
      }
    }

    tabSpecs_ = tmp;

    if ( rundat_ )
    {
      rundat_->warnings = 0;
    }
  }

  FuncUtils::configCond ( saveCond_, PropNames::SAVE_COND,
                          myProps,   globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void OutputModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  bool        append = ((fflags_ & FileFlags::APPEND) != 0);


  myConf.set ( PropNames::FILE,       fileName_  );
  myConf.set ( PropNames::APPEND,     append     );
  myConf.set ( PropNames::PRECISION,  precision_ );
  myConf.set ( PropNames::PAGE_WIDTH, pageWidth_ );
  myConf.set ( PropNames::BUFSIZE,    bufsize_   );

  if ( vecSpecs_ )
  {
    myConf.set ( PropNames::VECTORS, vecSpecs_->getSpecs() );
  }
  else
  {
    myConf.set ( PropNames::VECTORS, StringVector() );
  }

  if ( tabSpecs_ )
  {
    myConf.set ( PropNames::TABLES, tabSpecs_->getSpecs() );
  }
  else
  {
    myConf.set ( PropNames::TABLES, StringVector() );
  }

  FuncUtils::getConfig ( myConf, saveCond_, PropNames::SAVE_COND );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> OutputModule::makeNew

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


void OutputModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   checkFile_
//-----------------------------------------------------------------------


void OutputModule::checkFile_

  ( const String&  fname,
    OpenFlags      flags ) const

{
  if ( (flags & FileFlags::APPEND) && util::isGzipFile( fname ) )
  {
    throw jem::IllegalInputException (
      getContext (),
      "can not append to a compressed output file "
      "(set `append\' to false, or modify the file name)"
    );
  }
}


JIVE_END_PACKAGE( app )
