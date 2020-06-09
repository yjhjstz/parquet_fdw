MODULE_big = parquet_fdw
OBJS = parquet_impl.o parquet_fdw.o stream_writer.o deparse.o option.o shippable.o
PGFILEDESC = "parquet_fdw - foreign data wrapper for parquet"

PG_CPPFLAGS = -I$(libpq_srcdir) -I/opt/annconda3/envs/rapids/include/
SHLIB_LINK_INTERNAL = $(libpq)
SHLIB_LINK = -lm -lstdc++ -lparquet -larrow

EXTENSION = parquet_fdw
DATA = parquet_fdw--0.1.sql

REGRESS = parquet_fdw

EXTRA_CLEAN = sql/parquet_fdw.sql expected/parquet_fdw.out


# parquet_impl.cpp requires C++ 11.
override PG_CXXFLAGS += -std=c++11 -O0 -g


# pass CCFLAGS (when defined) to both C and C++ compilers.
ifdef CCFLAGS
	override PG_CXXFLAGS += $(CCFLAGS)
	override PG_CFLAGS += $(CCFLAGS)
endif

ifdef USE_PGXS
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
else
subdir = contrib/parquet_fdw
top_builddir = ../..
include $(top_builddir)/src/Makefile.global
include $(top_srcdir)/contrib/contrib-global.mk
endif

# XXX: PostgreSQL below 11 does not automatically add -fPIC or equivalent to C++
# flags when building a shared library, have to do it here explicitely.
ifeq ($(shell test $(VERSION_NUM) -lt 110000; echo $$?), 0)
	override CXXFLAGS += $(CFLAGS_SL)
endif


# XXX: a hurdle to use common compiler flags when building bytecode from C++
# files. should be not unnecessary, but src/Makefile.global omits passing those
# flags for an unnknown reason.
%.bc : %.cpp
	$(COMPILE.cxx.bc) $(CXXFLAGS) $(CPPFLAGS)  -o $@ $<
