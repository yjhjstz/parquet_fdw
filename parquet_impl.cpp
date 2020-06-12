/*
 * Parquet processing implementation
 */

#include <list>
#include <set>
#include <utility>
#include <iostream>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/array.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"
#include "parquet/exception.h"
#include "parquet/file_reader.h"
#include "parquet/statistics.h"

//#include "parquet/schema.h"
//#include "parquet/stream_reader.h"
//#include "parquet/stream_writer.h"
#include "stream_writer.h"

extern "C"
{
#include "postgres.h"

#include "access/htup_details.h"
#include "access/parallel.h"
#include "access/sysattr.h"
#include "access/nbtree.h"
#include "catalog/pg_type.h"
#include "commands/defrem.h"
#include "commands/explain.h"
#include "executor/tuptable.h"
#include "foreign/foreign.h"
#include "foreign/fdwapi.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/makefuncs.h"
#include "nodes/parsenodes.h"
#include "nodes/pathnodes.h"

#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "parser/parse_coerce.h"
#include "parser/parse_oper.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/date.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "utils/selfuncs.h"
#include "storage/lmgr.h"


#if PG_VERSION_NUM < 120000
#include "nodes/relation.h"
#include "optimizer/var.h"
#else
#include "access/table.h"
#include "access/relation.h"
#include "optimizer/optimizer.h"
#endif

#include "postgres_fdw.h"
}

#define SEGMENT_SIZE (1024 * 1024)

#define to_postgres_timestamp(tstype, i, ts)                    \
    switch ((tstype)->unit()) {                                 \
        case arrow::TimeUnit::SECOND:                           \
            ts = time_t_to_timestamptz((i)); break;             \
        case arrow::TimeUnit::MILLI:                            \
            ts = time_t_to_timestamptz((i) / 1000); break;      \
        case arrow::TimeUnit::MICRO:                            \
            ts = time_t_to_timestamptz((i) / 1000000); break;   \
        case arrow::TimeUnit::NANO:                             \
            ts = time_t_to_timestamptz((i) / 1000000000); break;\
        default:                                                \
            elog(ERROR, "Timestamp of unknown precision: %d",   \
                 (tstype)->unit());                             \
    }

static void find_cmp_func(FmgrInfo *finfo, Oid type1, Oid type2);
static Datum bytes_to_postgres_type(const char *bytes, arrow::DataType *arrow_type);
static void add_foreign_grouping_paths(PlannerInfo *root,
                     RelOptInfo *input_rel,
                     RelOptInfo *grouped_rel,
                     GroupPathExtraData *extra);
static void add_foreign_ordered_paths(PlannerInfo *root,
                    RelOptInfo *input_rel,
                    RelOptInfo *ordered_rel);
static void add_foreign_final_paths(PlannerInfo *root,
                  RelOptInfo *input_rel,
                  RelOptInfo *final_rel,
                  FinalPathExtraData *extra);
static void estimate_path_cost_size(PlannerInfo *root,
            RelOptInfo *foreignrel,
            List *param_join_conds,
            List *pathkeys,
            PgFdwPathExtraData *fpextra,
            double *p_rows, int *p_width,
            Cost *p_startup_cost, Cost *p_total_cost);
static void adjust_foreign_grouping_path_cost(PlannerInfo *root,
                        List *pathkeys,
                        double retrieved_rows,
                        double width,
                        double limit_tuples,
                        Cost *p_startup_cost,
                        Cost *p_run_cost);
static void add_paths_with_pathkeys_for_rel(PlannerInfo *root, RelOptInfo *rel,
                      Path *epq_path);
bool parquet_fdw_use_threads = true;

/*
 * Restriction
 */
struct RowGroupFilter
{
    AttrNumber  attnum;
    Const      *value;
    int         strategy;
};

/*
 * Plain C struct for fdw_state
 */
struct ParquetFdwPlanState
{
    char       *filename;
    Bitmapset  *attrs_sorted;

    bool        use_mmap;
    bool        use_threads;
    List       *rowgroups;
    uint64      ntuples;

    /*
     * True means that the relation can be pushed down. Always true for simple
     * foreign scan.
     */
    bool    pushdown_safe;

    /*
     * Restriction clauses, divided into safe and unsafe to pushdown subsets.
     * All entries in these lists should have RestrictInfo wrappers; that
     * improves efficiency of selectivity and cost estimation.
     */
    List     *remote_conds;
    List     *local_conds;

    /* Actual remote restriction clauses for scan (sans RestrictInfos) */
    List     *final_remote_exprs;

    /* Bitmap of attr numbers we need to fetch from the remote server. */
    Bitmapset  *attrs_used;

    /* True means that the query_pathkeys is safe to push down */
    bool    qp_is_pushdown_safe;

    /* Cost and selectivity of local_conds. */
    QualCost  local_conds_cost;
    Selectivity local_conds_sel;

    /* Selectivity of join conditions */
    Selectivity joinclause_sel;

    /* Estimated size and cost for a scan, join, or grouping/aggregation. */
    double    rows;
    int     width;
    Cost    startup_cost;
    Cost    total_cost;

    /*
     * Estimated number of rows fetched from the foreign server, and costs
     * excluding costs for transferring those rows from the foreign server.
     * These are only used by estimate_path_cost_size().
     */
    double    retrieved_rows;
    Cost    rel_startup_cost;
    Cost    rel_total_cost;

    /* Options extracted from catalogs. */
    bool    use_remote_estimate;
    Cost    fdw_startup_cost;
    Cost    fdw_tuple_cost;
    List     *shippable_extensions; /* OIDs of whitelisted extensions */

    /* Cached catalog information. */
    ForeignTable *table;
    ForeignServer *server;
    UserMapping *user;      /* only set in use_remote_estimate mode */

    int     fetch_size;   /* fetch size for this remote table */

    /*
     * Name of the relation while EXPLAINing ForeignScan. It is used for join
     * relations but is set for all relations. For join relation, the name
     * indicates which foreign tables are being joined and the join type used.
     */
    StringInfo  relation_name;

    /* Join information */
    RelOptInfo *outerrel;
    RelOptInfo *innerrel;
    JoinType  jointype;
    /* joinclauses contains only JOIN/ON conditions for an outer join */
    List     *joinclauses;  /* List of RestrictInfo */

    /* Upper relation information */
    UpperRelationKind stage;

    /* Grouping information */
    List     *grouped_tlist;

    /* Subquery information */
    bool    make_outerrel_subquery; /* do we deparse outerrel as a
                       * subquery? */
    bool    make_innerrel_subquery; /* do we deparse innerrel as a
                       * subquery? */
    Relids    lower_subquery_rels;  /* all relids appearing in lower
                       * subqueries */

    /*
     * Index of the relation.  It is used to create an alias to a subquery
     * representing the relation.
     */
    int     relation_index;
};

struct ChunkInfo
{
    int     chunk;      /* current chunk number */
    int64   pos;        /* current pos within chunk */
    int64   len;        /* current chunk length */
};

struct ParallelCoordinator
{
    pg_atomic_uint32 next_rowgroup; 
};

class ParquetFdwExecutionState
{
public:
    std::unique_ptr<parquet::arrow::FileReader> reader;

    std::shared_ptr<arrow::Schema>  schema;

    /* Arrow column indices that are used in query */
    std::vector<int>                indices;

    /*
     * Mapping between slot attributes and arrow result set columns.
     * Corresponds to 'indices' vector.
     */
    std::vector<int>                map;

    /*
     * Cast functions from dafult postgres type defined in `to_postgres_type`
     * to actual table column type.
     */
    std::vector<FmgrInfo *>         castfuncs;

    /* Current row group */
    std::shared_ptr<arrow::Table>   table;

    /*
     * Plain pointers to inner the structures of row group. It's needed to
     * prevent excessive shared_ptr management.
     */
    std::vector<arrow::Array *>     chunks;
    std::vector<arrow::DataType *>  types;

    bool           *has_nulls;          /* per-column info on nulls */

    int             row_group;          /* current row group index */
    uint32_t        row;                /* current row within row group */
    uint32_t        num_rows;           /* total rows in row group */
    std::vector<ChunkInfo> chunk_info;  /* current chunk and position per-column */

    /*
     * Filters built from query restrictions that help to filter out row
     * groups.
     */
    std::list<RowGroupFilter>       filters;

    /*
     * List of row group indexes to scan
     */
    std::vector<int>                rowgroups;

    /*
     * Special memory segment to speed up bytea/Text allocations.
     */
    MemoryContext                   segments_cxt;
    char                           *segment_start_ptr;
    char                           *segment_cur_ptr;
    char                           *segment_last_ptr;
    std::list<char *>               garbage_segments;

    /* Callback to delete this state object in case of ERROR */
    MemoryContextCallback           callback;
    bool                            autodestroy;

    /* Coordinator for parallel query execution */
    ParallelCoordinator            *coordinator;

    /* Wether object is properly initialized */
    bool     initialized;

    ParquetFdwExecutionState(const char *filename, bool use_mmap)
        : row_group(-1), row(0), num_rows(0), coordinator(NULL),
          initialized(false)
    {
        parquet::arrow::FileReader::Make(
                arrow::default_memory_pool(),
                parquet::ParquetFileReader::OpenFile(filename, use_mmap),
                &reader
        );
    }
};


class ParquetInsertState
{
public:
    std::shared_ptr<::arrow::io::FileOutputStream>  parquet_file;

    std::shared_ptr<parquet::schema::GroupNode> schema;

    std::shared_ptr<parquet::StreamWriter2> stream_writer;

    /* Arrow column indices that are used in query */
    std::vector<int>                indices;

    /*
     * Mapping between slot attributes and arrow result set columns.
     * Corresponds to 'indices' vector.
     */
    std::vector<int>                map;

    /*
     * Cast functions from dafult postgres type defined in `to_postgres_type`
     * to actual table column type.
     */
    std::vector<FmgrInfo *>         castfuncs;

    /* Current row group */
    std::shared_ptr<arrow::Table>   table;

    /*
     * Plain pointers to inner the structures of row group. It's needed to
     * prevent excessive shared_ptr management.
     */
    std::vector<arrow::Array *>     chunks;
    std::vector<arrow::DataType *>  types;

    bool           *has_nulls;          /* per-column info on nulls */

    int             row_group;          /* current row group index */
    uint32_t        row;                /* current row within row group */
    uint32_t        num_rows;           /* total rows in row group */
    std::vector<ChunkInfo> chunk_info;  /* current chunk and position per-column */

    /*
     * Filters built from query restrictions that help to filter out row
     * groups.
     */
    std::list<RowGroupFilter>       filters;

    /*
     * List of row group indexes to scan
     */
    std::vector<int>                rowgroups;

    /*
     * Special memory segment to speed up bytea/Text allocations.
     */
    MemoryContext                   memcxt;
   
    /* Callback to delete this state object in case of ERROR */
    MemoryContextCallback           callback;
    bool                            autodestroy;

    /* Coordinator for parallel query execution */
    ParallelCoordinator            *coordinator;

    /* Wether object is properly initialized */
    bool     initialized;

    ParquetInsertState(const char *filename)
        : row_group(-1), row(0), num_rows(0), coordinator(NULL),
          initialized(false)
    {
        // PARQUET_ASSIGN_OR_THROW(parquet_file,
        //     arrow::io::FileOutputStream::Open(filename, false));
        arrow::io::FileOutputStream::Open(filename, false, &parquet_file);
    }
};

static void
destroy_parquet_state(void *arg)
{
    ParquetFdwExecutionState *festate = (ParquetFdwExecutionState *) arg;

    if (festate->autodestroy)
        delete festate;
}

static ParquetFdwExecutionState *
create_parquet_state(const char *filename,
                     TupleDesc tupleDesc,
                     MemoryContext parent_cxt,
                     std::set<int> &attrs_used,
                     bool use_mmap,
                     bool use_threads)
{
    ParquetFdwExecutionState *festate;

    festate = new ParquetFdwExecutionState(filename, use_mmap);
    auto schema = festate->reader->parquet_reader()->metadata()->schema();
    parquet::ArrowReaderProperties props;

    if (!parquet::arrow::FromParquetSchema(schema, props, &festate->schema).ok())
        elog(ERROR, "parquet_fdw: error reading parquet schema");


    parquet::schema::PrintSchema(schema->schema_root().get(), std::cout);

    /* Enable parallel columns decoding/decompression if needed */
    festate->reader->set_use_threads(use_threads && parquet_fdw_use_threads);
 
    /* Create mapping between tuple descriptor and parquet columns. */
    festate->map.resize(tupleDesc->natts);
    for (int i = 0; i < tupleDesc->natts; i++)
    {
        AttrNumber attnum = i + 1 - FirstLowInvalidHeapAttributeNumber;

        festate->map[i] = -1;

        /* Skip columns we don't intend to use in query */
        if (attrs_used.find(attnum) == attrs_used.end())
            continue;

        for (int k = 0; k < schema->num_columns(); k++)
        {
            parquet::schema::NodePtr node = schema->Column(k)->schema_node();
            std::vector<std::string> path = node->path()->ToDotVector();

            /*
             * Compare postgres attribute name to the top level column name in
             * parquet.
             *
             * XXX If we will ever want to support structs then this should be
             * changed.
             */
            //elog(INFO, "dot path %s .", path[0].c_str());
            if (strcmp(NameStr(TupleDescAttr(tupleDesc, i)->attname),
                       path[0].c_str()) == 0)
            {
                /* Found mapping! */
                festate->indices.push_back(k);

                /* index of last element */
                festate->map[i] = festate->indices.size() - 1; 

                festate->types.push_back(festate->schema->field(k)->type().get());
                break;
            }
        }
    }

    festate->has_nulls = (bool *) palloc(sizeof(bool) * festate->map.size());

    festate->segments_cxt = AllocSetContextCreate(parent_cxt,
                                                  "parquet_fdw tuple data",
                                                  ALLOCSET_DEFAULT_SIZES);
    festate->segment_start_ptr = NULL;
    festate->segment_cur_ptr = NULL;
    festate->segment_last_ptr = NULL;

    /*
     * Enable automatic execution state destruction by using memory context
     * callback
     */
    festate->callback.func = destroy_parquet_state;
    festate->callback.arg = (void *) festate;
    MemoryContextRegisterResetCallback(festate->segments_cxt,
                                       &festate->callback);
    festate->autodestroy = true;

    return festate;
}



/*
 * C interface functions
 */

static Bitmapset *
parse_attributes_list(char *start, Oid relid)
{
    Bitmapset *attrs = NULL;
    char      *token;
    const char *delim = " ";
    AttrNumber attnum;

    while ((token = strtok(start, delim)) != NULL)
    {
        if ((attnum = get_attnum(relid, token)) == InvalidAttrNumber)
            elog(ERROR, "paruqet_fdw: invalid attribute name '%s'", token);
        attrs = bms_add_member(attrs, attnum);
        start = NULL;
    }

    return attrs;
}


static void
apply_server_options(PgFdwRelationInfo *fpinfo)
{
  ListCell   *lc;

  foreach(lc, fpinfo->server->options)
  {
    DefElem    *def = (DefElem *) lfirst(lc);

    if (strcmp(def->defname, "use_remote_estimate") == 0)
      fpinfo->use_remote_estimate = defGetBoolean(def);
    else if (strcmp(def->defname, "fdw_startup_cost") == 0)
      fpinfo->fdw_startup_cost = strtod(defGetString(def), NULL);
    else if (strcmp(def->defname, "fdw_tuple_cost") == 0)
      fpinfo->fdw_tuple_cost = strtod(defGetString(def), NULL);
    else if (strcmp(def->defname, "extensions") == 0)
      fpinfo->shippable_extensions =
        ExtractExtensionList(defGetString(def), false);
    else if (strcmp(def->defname, "fetch_size") == 0)
      fpinfo->fetch_size = strtol(defGetString(def), NULL, 10);
  }
}

/*
 * Parse options from foreign table and apply them to fpinfo.
 *
 * New options might also require tweaking merge_fdw_options().
 */
static void
apply_table_options(PgFdwRelationInfo *fpinfo)
{
  ListCell   *lc;

  foreach(lc, fpinfo->table->options)
  {
      DefElem    *def = (DefElem *) lfirst(lc);

      if (strcmp(def->defname, "filename") == 0)
            fpinfo->filename = defGetString(def);
      else if (strcmp(def->defname, "use_mmap") == 0)
      { 
          if (!parse_bool(defGetString(def), &fpinfo->use_mmap))
              ereport(ERROR,
                      (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                       errmsg("invalid value for boolean option \"%s\": %s",
                              def->defname, defGetString(def))));
      }
      else if (strcmp(def->defname, "use_threads") == 0)
      {
          if (!parse_bool(defGetString(def), &fpinfo->use_threads))
              ereport(ERROR,
                      (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                       errmsg("invalid value for boolean option \"%s\": %s",
                              def->defname, defGetString(def))));
      } else if (strcmp(def->defname, "use_remote_estimate") == 0) {
            fpinfo->use_remote_estimate = defGetBoolean(def);
      } else if (strcmp(def->defname, "fetch_size") == 0) {
            fpinfo->fetch_size = strtol(defGetString(def), NULL, 10);
      } else
          elog(ERROR, "unknown option '%s'", def->defname);
  }
}


static void
merge_fdw_options(PgFdwRelationInfo *fpinfo,
          const PgFdwRelationInfo *fpinfo_o,
          const PgFdwRelationInfo *fpinfo_i)
{
  /* We must always have fpinfo_o. */
  Assert(fpinfo_o);

  /* fpinfo_i may be NULL, but if present the servers must both match. */
  Assert(!fpinfo_i ||
       fpinfo_i->server->serverid == fpinfo_o->server->serverid);

  /*
   * Copy the server specific FDW options.  (For a join, both relations come
   * from the same server, so the server options should have the same value
   * for both relations.)
   */
  fpinfo->fdw_startup_cost = fpinfo_o->fdw_startup_cost;
  fpinfo->fdw_tuple_cost = fpinfo_o->fdw_tuple_cost;
  fpinfo->shippable_extensions = fpinfo_o->shippable_extensions;
  fpinfo->use_remote_estimate = fpinfo_o->use_remote_estimate;
  fpinfo->fetch_size = fpinfo_o->fetch_size;

  /* Merge the table level options from either side of the join. */
  if (fpinfo_i)
  {
    /*
     * We'll prefer to use remote estimates for this join if any table
     * from either side of the join is using remote estimates.  This is
     * most likely going to be preferred since they're already willing to
     * pay the price of a round trip to get the remote EXPLAIN.  In any
     * case it's not entirely clear how we might otherwise handle this
     * best.
     */
    fpinfo->use_remote_estimate = fpinfo_o->use_remote_estimate ||
      fpinfo_i->use_remote_estimate;

    /*
     * Set fetch size to maximum of the joining sides, since we are
     * expecting the rows returned by the join to be proportional to the
     * relation sizes.
     */
    fpinfo->fetch_size = Max(fpinfo_o->fetch_size, fpinfo_i->fetch_size);
  }
}

static void
get_table_options(Oid relid, ParquetFdwPlanState *fdw_private)
{
	  ForeignTable *table;
    ListCell     *lc;

    fdw_private->use_mmap = false;
    fdw_private->use_threads = false;
    table = GetForeignTable(relid);
    
    foreach(lc, table->options)
    {
		    DefElem    *def = (DefElem *) lfirst(lc);

        if (strcmp(def->defname, "filename") == 0)
            fdw_private->filename = defGetString(def);
        else if (strcmp(def->defname, "sorted") == 0)
        {
            fdw_private->attrs_sorted =
                parse_attributes_list(defGetString(def), relid);
        }
        else if (strcmp(def->defname, "use_mmap") == 0)
        { 
            if (!parse_bool(defGetString(def), &fdw_private->use_mmap))
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                         errmsg("invalid value for boolean option \"%s\": %s",
                                def->defname, defGetString(def))));
        }
        else if (strcmp(def->defname, "use_threads") == 0)
        {
            if (!parse_bool(defGetString(def), &fdw_private->use_threads))
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                         errmsg("invalid value for boolean option \"%s\": %s",
                                def->defname, defGetString(def))));
        } else
            elog(ERROR, "unknown option '%s'", def->defname);
    }
}

extern "C" void
parquetGetForeignRelSize(PlannerInfo *root,
					  RelOptInfo *baserel,
					  Oid foreigntableid)
{

    // ParquetFdwPlanState *fdw_private;

    // fdw_private = (ParquetFdwPlanState *) palloc0(sizeof(ParquetFdwPlanState));
    // get_table_options(foreigntableid, fdw_private);
    // baserel->fdw_private = fdw_private;

    PgFdwRelationInfo *fpinfo;
    ListCell   *lc;
    RangeTblEntry *rte = planner_rt_fetch(baserel->relid, root);
    const char *ns;
    const char *relname;
    const char *refname;

    /*
     * We use PgFdwRelationInfo to pass various information to subsequent
     * functions.
     */
    fpinfo = (PgFdwRelationInfo *) palloc0(sizeof(ParquetFdwPlanState));
    baserel->fdw_private = (void *) fpinfo;

    /* Base foreign tables need to be pushed down always. */
    fpinfo->pushdown_safe = true;

    /* Look up foreign-table catalog info. */
    fpinfo->table = GetForeignTable(foreigntableid);
    fpinfo->server = GetForeignServer(fpinfo->table->serverid);

    /*
     * Extract user-settable option values.  Note that per-table setting of
     * use_remote_estimate overrides per-server setting.
     */
    fpinfo->use_remote_estimate = false;
    fpinfo->fdw_startup_cost = DEFAULT_FDW_STARTUP_COST;
    fpinfo->fdw_tuple_cost = DEFAULT_FDW_TUPLE_COST;
    fpinfo->shippable_extensions = NIL;
    fpinfo->fetch_size = 100;

    apply_server_options(fpinfo);
    apply_table_options(fpinfo);

    /*
     * If the table or the server is configured to use remote estimates,
     * identify which user to do remote access as during planning.  This
     * should match what ExecCheckRTEPerms() does.  If we fail due to lack of
     * permissions, the query would have failed at runtime anyway.
     */
    if (fpinfo->use_remote_estimate)
    {
      Oid     userid = rte->checkAsUser ? rte->checkAsUser : GetUserId();

      fpinfo->user = GetUserMapping(userid, fpinfo->server->serverid);
    }
    else
      fpinfo->user = NULL;

    /*
     * Identify which baserestrictinfo clauses can be sent to the remote
     * server and which can't.
     */
    classifyConditions(root, baserel, baserel->baserestrictinfo,
               &fpinfo->remote_conds, &fpinfo->local_conds);

    /*
     * Identify which attributes will need to be retrieved from the remote
     * server.  These include all attrs needed for joins or final output, plus
     * all attrs used in the local_conds.  (Note: if we end up using a
     * parameterized scan, it's possible that some of the join clauses will be
     * sent to the remote and thus we wouldn't really need to retrieve the
     * columns used in them.  Doesn't seem worth detecting that case though.)
     */
    fpinfo->attrs_used = NULL;
    pull_varattnos((Node *) baserel->reltarget->exprs, baserel->relid,
             &fpinfo->attrs_used);
    foreach(lc, fpinfo->local_conds)
    {
      RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

      pull_varattnos((Node *) rinfo->clause, baserel->relid,
               &fpinfo->attrs_used);
    }

    /*
     * Compute the selectivity and cost of the local_conds, so we don't have
     * to do it over again for each path.  The best we can do for these
     * conditions is to estimate selectivity on the basis of local statistics.
     */
    fpinfo->local_conds_sel = clauselist_selectivity(root,
                             fpinfo->local_conds,
                             baserel->relid,
                             JOIN_INNER,
                             NULL);

    cost_qual_eval(&fpinfo->local_conds_cost, fpinfo->local_conds, root);

    /*
     * Set # of retrieved rows and cached relation costs to some negative
     * value, so that we can detect when they are set to some sensible values,
     * during one (usually the first) of the calls to estimate_path_cost_size.
     */
    fpinfo->retrieved_rows = -1;
    fpinfo->rel_startup_cost = -1;
    fpinfo->rel_total_cost = -1;

    /*
     * If the table or the server is configured to use remote estimates,
     * connect to the foreign server and execute EXPLAIN to estimate the
     * number of rows selected by the restriction clauses, as well as the
     * average row width.  Otherwise, estimate using whatever statistics we
     * have locally, in a way similar to ordinary tables.
     */
    if (fpinfo->use_remote_estimate)
    {
      /*
       * Get cost/size estimates with help of remote server.  Save the
       * values in fpinfo so we don't need to do it again to generate the
       * basic foreign path.
       */
      estimate_path_cost_size(root, baserel, NIL, NIL, NULL,
                  &fpinfo->rows, &fpinfo->width,
                  &fpinfo->startup_cost, &fpinfo->total_cost);

      /* Report estimated baserel size to planner. */
      baserel->rows = fpinfo->rows;
      baserel->reltarget->width = fpinfo->width;
    }
    else
    {
      /*
       * If the foreign table has never been ANALYZEd, it will have relpages
       * and reltuples equal to zero, which most likely has nothing to do
       * with reality.  We can't do a whole lot about that if we're not
       * allowed to consult the remote server, but we can use a hack similar
       * to plancat.c's treatment of empty relations: use a minimum size
       * estimate of 10 pages, and divide by the column-datatype-based width
       * estimate to get the corresponding number of tuples.
       */
      if (baserel->pages == 0 && baserel->tuples == 0)
      {
        baserel->pages = 10;
        baserel->tuples =
          (10 * BLCKSZ) / (baserel->reltarget->width +
                   MAXALIGN(SizeofHeapTupleHeader));
      }

      /* Estimate baserel size as best we can with local statistics. */
      set_baserel_size_estimates(root, baserel);

      /* Fill in basically-bogus cost estimates for use later. */
      estimate_path_cost_size(root, baserel, NIL, NIL, NULL,
                  &fpinfo->rows, &fpinfo->width,
                  &fpinfo->startup_cost, &fpinfo->total_cost);
    }

    /*
     * Set the name of relation in fpinfo, while we are constructing it here.
     * It will be used to build the string describing the join relation in
     * EXPLAIN output. We can't know whether VERBOSE option is specified or
     * not, so always schema-qualify the foreign table name.
     */
    fpinfo->relation_name = makeStringInfo();
    ns = get_namespace_name(get_rel_namespace(foreigntableid));
    relname = get_rel_name(foreigntableid);
    refname = rte->eref->aliasname;
    appendStringInfo(fpinfo->relation_name, "%s.%s",
             quote_identifier(ns),
             quote_identifier(relname));
    if (*refname && strcmp(refname, relname) != 0)
      appendStringInfo(fpinfo->relation_name, " %s",
               quote_identifier(rte->eref->aliasname));


    /* No outer and inner relations. */
    fpinfo->make_outerrel_subquery = false;
    fpinfo->make_innerrel_subquery = false;
    fpinfo->lower_subquery_rels = NULL;
    /* Set the relation index. */
    fpinfo->relation_index = baserel->relid;
}

/*
 * extract_rowgroup_filters
 *      Build a list of expressions we can use to filter out row groups.
 */
static void
extract_rowgroup_filters(List *scan_clauses,
                         std::list<RowGroupFilter> &filters)
{
    ListCell *lc;

    foreach (lc, scan_clauses)
    {
        TypeCacheEntry *tce;
        Expr       *clause = (Expr *) lfirst(lc);
        OpExpr     *expr;
        Expr       *left, *right;
        int         strategy;
        Const      *c;
        Var        *v;
        Oid         opno;

        if (IsA(clause, RestrictInfo))
            clause = ((RestrictInfo *) clause)->clause;

        if (IsA(clause, OpExpr))
        {
            expr = (OpExpr *) clause;

            /* Only interested in binary opexprs */
            if (list_length(expr->args) != 2)
                continue;

            left = (Expr *) linitial(expr->args);
            right = (Expr *) lsecond(expr->args);

            /*
             * Looking for expressions like "EXPR OP CONST" or "CONST OP EXPR"
             *
             * XXX Currently only Var as expression is supported. Will be
             * extended in future.
             */
            if (IsA(right, Const))
            {
                if (!IsA(left, Var))
                    continue;
                v = (Var *) left;
                c = (Const *) right;
                opno = expr->opno;
            }
            else if (IsA(left, Const))
            {
                /* reverse order (CONST OP VAR) */
                if (!IsA(right, Var))
                    continue;
                v = (Var *) right;
                c = (Const *) left;
                opno = get_commutator(expr->opno);
            }
            else
                continue;

            /* TODO */
            tce = lookup_type_cache(exprType((Node *) left),
                                    TYPECACHE_BTREE_OPFAMILY);
            strategy = get_op_opfamily_strategy(opno, tce->btree_opf);

            /* Not a btree family operator? */
            if (strategy == 0)
                continue;
        }
        else if (IsA(clause, Var))
        {
            /*
             * Trivial expression containing only a single boolean Var. This
             * also covers cases "BOOL_VAR = true"
             * */
            v = (Var *) clause;
            strategy = BTEqualStrategyNumber;
            c = (Const *) makeBoolConst(true, false);
        }
        else if (IsA(clause, BoolExpr))
        {
            /*
             * Similar to previous case but for expressions like "!BOOL_VAR" or
             * "BOOL_VAR = false"
             */
            BoolExpr *boolExpr = (BoolExpr *) clause;

            if (boolExpr->args && list_length(boolExpr->args) != 1)
                continue;

            if (!IsA(linitial(boolExpr->args), Var))
                continue;

            v = (Var *) linitial(boolExpr->args);
            strategy = BTEqualStrategyNumber;
            c = (Const *) makeBoolConst(false, false);
        }
        else
            continue;

        RowGroupFilter f
        {
            .attnum = v->varattno,
            .value = c,
            .strategy = strategy,
        };

        filters.push_back(f);
    }
}

static Oid
to_postgres_type(int arrow_type)
{
    switch (arrow_type)
    {
        case arrow::Type::BOOL:
            return BOOLOID;
        case arrow::Type::INT32:
            return INT4OID;
        case arrow::Type::INT64:
            return INT8OID;
        case arrow::Type::FLOAT:
            return FLOAT4OID;
        case arrow::Type::DOUBLE:
            return FLOAT8OID;
        case arrow::Type::STRING:
            return TEXTOID;
        case arrow::Type::BINARY:
            return BYTEAOID;
        case arrow::Type::TIMESTAMP:
            return TIMESTAMPOID;
        case arrow::Type::DATE32:
            return DATEOID;
        default:
            return InvalidOid;
    }
}

/*
 * row_group_matches_filter
 *      Check if min/max values of the column of the row group match filter.
 */
static bool
row_group_matches_filter(parquet::Statistics *stats,
                         arrow::DataType *arrow_type,
                         RowGroupFilter *filter)
{
    FmgrInfo finfo;
    Datum    val = filter->value->constvalue;
    int      collid = filter->value->constcollid;
    int      strategy = filter->strategy;

    find_cmp_func(&finfo,
                  filter->value->consttype,
                  to_postgres_type(arrow_type->id()));

    switch (filter->strategy)
    {
        case BTLessStrategyNumber:
        case BTLessEqualStrategyNumber:
            {
                Datum   lower;
                int     cmpres;
                bool    satisfies;

                lower = bytes_to_postgres_type(stats->EncodeMin().c_str(),
                                               arrow_type);
                cmpres = FunctionCall2Coll(&finfo, collid, val, lower);

                satisfies =
                    (strategy == BTLessStrategyNumber      && cmpres > 0) ||
                    (strategy == BTLessEqualStrategyNumber && cmpres >= 0);

                if (!satisfies)
                    return false;
                break;
            }

        case BTGreaterStrategyNumber:
        case BTGreaterEqualStrategyNumber:
            {
                Datum   upper;
                int     cmpres;
                bool    satisfies;

                upper = bytes_to_postgres_type(stats->EncodeMax().c_str(),
                                               arrow_type);
                cmpres = FunctionCall2Coll(&finfo, collid, val, upper);

                satisfies =
                    (strategy == BTGreaterStrategyNumber      && cmpres < 0) ||
                    (strategy == BTGreaterEqualStrategyNumber && cmpres <= 0);

                if (!satisfies)
                    return false;
                break;
            }

        case BTEqualStrategyNumber:
            {
                Datum   lower,
                        upper;

                lower = bytes_to_postgres_type(stats->EncodeMin().c_str(),
                                               arrow_type);
                upper = bytes_to_postgres_type(stats->EncodeMax().c_str(),
                                               arrow_type);

                int l = FunctionCall2Coll(&finfo, collid, val, lower);
                int u = FunctionCall2Coll(&finfo, collid, val, upper);

                if (l < 0 || u > 0)
                    return false;
            }

        default:
            /* should not happen */
            Assert(true);
    }

    return true;
}

/*
 * extract_rowgroups_list
 *      Analyze query predicates and using min/max statistics determine which
 *      row groups satisfy clauses. Store resulting row group list to
 *      fdw_private.
 */
static void
extract_rowgroups_list(PlannerInfo *root, RelOptInfo *baserel)
{
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::list<RowGroupFilter>       filters;
    RangeTblEntry  *rte;
    Relation        rel;
    TupleDesc       tupleDesc;
    auto            fdw_private = (ParquetFdwPlanState *) baserel->fdw_private;

    /*
     * Open relation to be able to access tuple descriptor
     */
    rte = root->simple_rte_array[baserel->relid];
    rel = heap_open(rte->relid, AccessShareLock);
    tupleDesc = RelationGetDescr(rel);

    /* Analyze query clauses and extract ones that can be of interest to us*/
    extract_rowgroup_filters(baserel->baserestrictinfo, filters);

    /* Open parquet file to read meta information */
    try
    {
        parquet::arrow::FileReader::Make(
                arrow::default_memory_pool(),
                parquet::ParquetFileReader::OpenFile(fdw_private->filename, false),
                &reader
        );

    }
    catch(const std::exception& e)
    {
        elog(ERROR, "parquet_fdw: parquet initialization failed: %s", e.what());
    }
    auto meta = reader->parquet_reader()->metadata();
    parquet::ArrowReaderProperties  props;
    std::shared_ptr<arrow::Schema>  schema;
    parquet::arrow::FromParquetSchema(meta->schema(), props, &schema);

    /* Check each row group whether it matches the filters */
    for (int r = 0; r < reader->num_row_groups(); r++)
    {
        bool match = true;
        auto rowgroup = meta->RowGroup(r);

        for (auto it = filters.begin(); it != filters.end(); it++)
        {
            RowGroupFilter &filter = *it;
            AttrNumber      attnum;
            const char     *attname;

            attnum = filter.attnum - 1;
            attname = NameStr(TupleDescAttr(tupleDesc, attnum)->attname);

            /*
             * Search for the column with the same name as filtered attribute
             */
            for (int k = 0; k < rowgroup->num_columns(); k++)
            {
                auto    column = rowgroup->ColumnChunk(k);
                std::vector<std::string> path = column->path_in_schema()->ToDotVector();

                if (strcmp(attname, path[0].c_str()) == 0)
                {
                    /* Found it! */
                    std::shared_ptr<parquet::Statistics>  stats;
                    std::shared_ptr<arrow::Field>       field;
                    std::shared_ptr<arrow::DataType>    type;

                    stats = column->statistics();

                    /* Convert to arrow field to get appropriate type */
                    field = schema->field(k);
                    type = field->type();

                    /*
                     * If at least one filter doesn't match rowgroup exclude
                     * the current row group and proceed with the next one.
                     */
                    if (stats &&
                        !row_group_matches_filter(stats.get(), type.get(), &filter))
                    {
                        match = false;
                        elog(DEBUG1, "parquet_fdw: skip rowgroup %d", r + 1);
                    }
                    break;
                }
            }  /* loop over columns */

            if (!match)
                break;

        }  /* loop over filters */
        
        /* All the filters match this rowgroup */
        if (match)
        {
            fdw_private->rowgroups = lappend_int(fdw_private->rowgroups, r);
            fdw_private->ntuples += rowgroup->num_rows();
        } 
    }  /* loop over rowgroups */

    heap_close(rel, AccessShareLock);
}

static void
estimate_costs(PlannerInfo *root, RelOptInfo *baserel, Cost *startup_cost,
               Cost *run_cost, Cost *total_cost)
{
    auto    fdw_private = (ParquetFdwPlanState *) baserel->fdw_private;
    double  ntuples;

    /* Use statistics if we have it */
    if (baserel->tuples)
    {
        ntuples = baserel->tuples *
            clauselist_selectivity(root,
                                   baserel->baserestrictinfo,
                                   0,
                                   JOIN_INNER,
                                   NULL);

    }
    else
    {
        /*
         * If there is no statistics then use estimate based on rows number
         * in the selected row groups.
         */
        ntuples = fdw_private->ntuples;
    }

    /*
     * Here we assume that parquet tuple cost is the same as regular tuple cost
     * even though this is probably not true in many cases. Maybe we'll come up
     * with a smarter idea later.
     */
    *run_cost = ntuples * cpu_tuple_cost;
  	*startup_cost = baserel->baserestrictcost.startup;
  	*total_cost = *startup_cost + *run_cost;

    baserel->rows = ntuples;
}

static void
extract_used_attributes(RelOptInfo *baserel)
{
    ParquetFdwPlanState *fdw_private = (ParquetFdwPlanState *) baserel->fdw_private;
    ListCell *lc;

    pull_varattnos((Node *) baserel->reltarget->exprs,
                   baserel->relid,
                   &fdw_private->attrs_used);

    foreach(lc, baserel->baserestrictinfo)
    {
        RestrictInfo *rinfo = (RestrictInfo *) lfirst(lc);

        pull_varattnos((Node *) rinfo->clause,
                       baserel->relid,
                       &fdw_private->attrs_used);
    }

    if (bms_is_empty(fdw_private->attrs_used))
    {
        bms_free(fdw_private->attrs_used);
        fdw_private->attrs_used = bms_make_singleton(1 - FirstLowInvalidHeapAttributeNumber);
    }
}

/*
 * Prepare for processing of parameters used in remote query.
 */
static void
prepare_query_params(PlanState *node,
           List *fdw_exprs,
           int numParams,
           FmgrInfo **param_flinfo,
           List **param_exprs,
           const char ***param_values)
{
  int     i;
  ListCell   *lc;

  Assert(numParams > 0);

  /* Prepare for output conversion of parameters used in remote query. */
  *param_flinfo = (FmgrInfo *) palloc0(sizeof(FmgrInfo) * numParams);

  i = 0;
  foreach(lc, fdw_exprs)
  {
    Node     *param_expr = (Node *) lfirst(lc);
    Oid     typefnoid;
    bool    isvarlena;

    getTypeOutputInfo(exprType(param_expr), &typefnoid, &isvarlena);
    fmgr_info(typefnoid, &(*param_flinfo)[i]);
    i++;
  }

  /*
   * Prepare remote-parameter expressions for evaluation.  (Note: in
   * practice, we expect that all these expressions will be just Params, so
   * we could possibly do something more efficient than using the full
   * expression-eval machinery for this.  But probably there would be little
   * benefit, and it'd require postgres_fdw to know more than is desirable
   * about Param evaluation.)
   */
  *param_exprs = ExecInitExprList(fdw_exprs, node);

  /* Allocate buffer for text form of query parameters. */
  *param_values = (const char **) palloc0(numParams * sizeof(char *));
}

extern "C" void
parquetGetForeignPaths(PlannerInfo *root,
					RelOptInfo *baserel,
					Oid foreigntableid)
{
  #if 0
  	ParquetFdwPlanState *fdw_private;
  	Cost		startup_cost;
  	Cost		total_cost;
    Cost        run_cost;
    List       *pathkeys = NIL;

    fdw_private = (ParquetFdwPlanState *) baserel->fdw_private;

    /*
     * Extract list of row groups that match query clauses. Also calculate
     * approximate number of rows in result set based on total number of tuples
     * in those row groups. It isn't very precise but it is best we got.
     */
    extract_rowgroups_list(root, baserel);
    estimate_costs(root, baserel, &startup_cost, &run_cost, &total_cost);

    /* Collect used attributes to reduce number of read columns during scan */
    extract_used_attributes(baserel);

    /* Build pathkeys based on attrs_sorted */
    int attnum = -1;
    while ((attnum = bms_next_member(fdw_private->attrs_sorted, attnum)) >= 0)
    {
        Oid         relid = root->simple_rte_array[baserel->relid]->relid;
        Oid         typid,
                    collid;
        int32       typmod;
        Oid         sort_op;
        Var        *var;
        List       *attr_pathkey;

        /* Build an expression (simple var) */
        get_atttypetypmodcoll(relid, attnum, &typid, &typmod, &collid);
        var = makeVar(baserel->relid, attnum, typid, typmod, collid, 0);

        /* Lookup sorting operator for the attribute type */
        get_sort_group_operators(typid,
                                 true, false, false,
                                 &sort_op, NULL, NULL,
                                 NULL);

        attr_pathkey = build_expression_pathkey(root, (Expr *) var, NULL,
                                                sort_op, baserel->relids,
                                                true);
        pathkeys = list_concat(pathkeys, attr_pathkey);
    }

	/*
	 * Create a ForeignPath node and add it as only possible path.  We use the
	 * fdw_private list of the path to carry the convert_selectively option;
	 * it will be propagated into the fdw_private list of the Plan node.
	 */
	add_path(baserel, (Path *)
	create_foreignscan_path(root, baserel,
									 NULL,	/* default pathtarget */
									 baserel->rows,
									 startup_cost,
									 total_cost,
									 pathkeys,
									 baserel->lateral_relids,
									 NULL,	/* no extra plan */
									 (List *) fdw_private)); // todo(yang)

    if (baserel->consider_parallel > 0)
    {
        Path *parallel_path = (Path *)
                 create_foreignscan_path(root, baserel,
                                         NULL,	/* default pathtarget */
                                         baserel->rows,
                                         startup_cost,
                                         total_cost,
									                       NULL,
                                         baserel->lateral_relids,
                                         NULL,	/* no extra plan */
                                         (List *) fdw_private);

        int num_workers = max_parallel_workers_per_gather;

        parallel_path->rows = fdw_private->ntuples / (num_workers + 1);
        parallel_path->total_cost       = startup_cost + run_cost / num_workers;
        parallel_path->parallel_workers = num_workers;
        parallel_path->parallel_aware   = true;
        parallel_path->parallel_safe    = true;
        add_partial_path(baserel, parallel_path);
    }
  #endif
    PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) baserel->fdw_private;
    ForeignPath *path;
    //List     *ppi_list;
    //ListCell   *lc;

    /*
     * Create simplest ForeignScan path node and add it to baserel.  This path
     * corresponds to SeqScan path of regular tables (though depending on what
     * baserestrict conditions we were able to send to remote, there might
     * actually be an indexscan happening there).  We already did all the work
     * to estimate cost and size of this path.
     *
     * Although this path uses no join clauses, it could still have required
     * parameterization due to LATERAL refs in its tlist.
     */
    path = create_foreignscan_path(root, baserel,
                     NULL,  /* default pathtarget */
                     fpinfo->rows,
                     fpinfo->startup_cost,
                     fpinfo->total_cost,
                     NIL, /* no pathkeys */
                     baserel->lateral_relids,
                     NULL,  /* no extra plan */
                     NIL);  /* no fdw_private list */
    add_path(baserel, (Path *) path);

    /* Add paths with pathkeys */
    add_paths_with_pathkeys_for_rel(root, baserel, NULL);

}

extern "C" ForeignScan *
parquetGetForeignPlan(PlannerInfo *root,
                      RelOptInfo *foreignrel,
                      Oid foreigntableid,
                      ForeignPath *best_path,
                      List *tlist,
                      List *scan_clauses,
                      Plan *outer_plan)
{
    PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) foreignrel->fdw_private;
	  Index    scan_relid;
    List     *fdw_private;
    List     *remote_exprs = NIL;
    List     *local_exprs = NIL;
    List     *params_list = NIL;
    List     *fdw_scan_tlist = NIL;
    List     *fdw_recheck_quals = NIL;
    List     *retrieved_attrs;
    StringInfoData sql;
    bool    has_final_sort = false;
    bool    has_limit = false;
    ListCell   *lc;

    /*
     * Get FDW private data created by postgresGetForeignUpperPaths(), if any.
     */
    if (best_path->fdw_private)
    {
      has_final_sort = intVal(list_nth(best_path->fdw_private,
                       FdwPathPrivateHasFinalSort));
      has_limit = intVal(list_nth(best_path->fdw_private,
                    FdwPathPrivateHasLimit));
    }

    if (IS_SIMPLE_REL(foreignrel))
    {
      /*
       * For base relations, set scan_relid as the relid of the relation.
       */
      scan_relid = foreignrel->relid;

      /*
       * In a base-relation scan, we must apply the given scan_clauses.
       *
       * Separate the scan_clauses into those that can be executed remotely
       * and those that can't.  baserestrictinfo clauses that were
       * previously determined to be safe or unsafe by classifyConditions
       * are found in fpinfo->remote_conds and fpinfo->local_conds. Anything
       * else in the scan_clauses list will be a join clause, which we have
       * to check for remote-safety.
       *
       * Note: the join clauses we see here should be the exact same ones
       * previously examined by postgresGetForeignPaths.  Possibly it'd be
       * worth passing forward the classification work done then, rather
       * than repeating it here.
       *
       * This code must match "extract_actual_clauses(scan_clauses, false)"
       * except for the additional decision about remote versus local
       * execution.
       */
      foreach(lc, scan_clauses)
      {
        RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

        /* Ignore any pseudoconstants, they're dealt with elsewhere */
        if (rinfo->pseudoconstant)
          continue;

        if (list_member_ptr(fpinfo->remote_conds, rinfo))
          remote_exprs = lappend(remote_exprs, rinfo->clause);
        else if (list_member_ptr(fpinfo->local_conds, rinfo))
          local_exprs = lappend(local_exprs, rinfo->clause);
        else if (is_foreign_expr(root, foreignrel, rinfo->clause))
          remote_exprs = lappend(remote_exprs, rinfo->clause);
        else
          local_exprs = lappend(local_exprs, rinfo->clause);
      }

      /*
       * For a base-relation scan, we have to support EPQ recheck, which
       * should recheck all the remote quals.
       */
      fdw_recheck_quals = remote_exprs;
    }
    else
    {
      /*
       * Join relation or upper relation - set scan_relid to 0.
       */
      scan_relid = 0;

      /*
       * For a join rel, baserestrictinfo is NIL and we are not considering
       * parameterization right now, so there should be no scan_clauses for
       * a joinrel or an upper rel either.
       */
      Assert(!scan_clauses);

      /*
       * Instead we get the conditions to apply from the fdw_private
       * structure.
       */
      remote_exprs = extract_actual_clauses(fpinfo->remote_conds, false);
      local_exprs = extract_actual_clauses(fpinfo->local_conds, false);

      /*
       * We leave fdw_recheck_quals empty in this case, since we never need
       * to apply EPQ recheck clauses.  In the case of a joinrel, EPQ
       * recheck is handled elsewhere --- see postgresGetForeignJoinPaths().
       * If we're planning an upperrel (ie, remote grouping or aggregation)
       * then there's no EPQ to do because SELECT FOR UPDATE wouldn't be
       * allowed, and indeed we *can't* put the remote clauses into
       * fdw_recheck_quals because the unaggregated Vars won't be available
       * locally.
       */

      /* Build the list of columns to be fetched from the foreign server. */
      fdw_scan_tlist = build_tlist_to_deparse(foreignrel);

      /*
       * Ensure that the outer plan produces a tuple whose descriptor
       * matches our scan tuple slot.  Also, remove the local conditions
       * from outer plan's quals, lest they be evaluated twice, once by the
       * local plan and once by the scan.
       */
      if (outer_plan)
      {
        ListCell   *lc;

        /*
         * Right now, we only consider grouping and aggregation beyond
         * joins. Queries involving aggregates or grouping do not require
         * EPQ mechanism, hence should not have an outer plan here.
         */
        Assert(!IS_UPPER_REL(foreignrel));

        /*
         * First, update the plan's qual list if possible.  In some cases
         * the quals might be enforced below the topmost plan level, in
         * which case we'll fail to remove them; it's not worth working
         * harder than this.
         */
        foreach(lc, local_exprs)
        {
          Node     *qual = (Node*)lfirst(lc);

          outer_plan->qual = list_delete(outer_plan->qual, qual);

          /*
           * For an inner join the local conditions of foreign scan plan
           * can be part of the joinquals as well.  (They might also be
           * in the mergequals or hashquals, but we can't touch those
           * without breaking the plan.)
           */
          if (IsA(outer_plan, NestLoop) ||
            IsA(outer_plan, MergeJoin) ||
            IsA(outer_plan, HashJoin))
          {
            Join     *join_plan = (Join *) outer_plan;

            if (join_plan->jointype == JOIN_INNER)
              join_plan->joinqual = list_delete(join_plan->joinqual,
                                qual);
          }
        }

        /*
         * Now fix the subplan's tlist --- this might result in inserting
         * a Result node atop the plan tree.
         */
        outer_plan = change_plan_targetlist(outer_plan, fdw_scan_tlist,
                          best_path->path.parallel_safe);
      }
    }

    /*
     * Build the query string to be sent for execution, and identify
     * expressions to be sent as parameters.
     */
    initStringInfo(&sql);
    deparseSelectStmtForRel(&sql, root, foreignrel, fdw_scan_tlist,
                remote_exprs, best_path->path.pathkeys,
                has_final_sort, has_limit, false,
                &retrieved_attrs, &params_list);

    /* Remember remote_exprs for possible use by postgresPlanDirectModify */
    fpinfo->final_remote_exprs = remote_exprs;

    /*
     * Build the fdw_private list that will be available to the executor.
     * Items in the list must match order in enum FdwScanPrivateIndex.
     */
    fdw_private = list_make4(makeString(fpinfo->filename),
                 makeString(sql.data),
                 retrieved_attrs,
                 makeInteger(fpinfo->fetch_size));
    if (IS_JOIN_REL(foreignrel) || IS_UPPER_REL(foreignrel))
      fdw_private = lappend(fdw_private,
                  makeString(fpinfo->relation_name->data));

    /*
     * Create the ForeignScan node for the given relation.
     *
     * Note that the remote parameter expressions are stored in the fdw_exprs
     * field of the finished plan node; we can't keep them in private state
     * because then they wouldn't be subject to later planner processing.
     */
    return make_foreignscan(tlist,
                local_exprs,
                scan_relid,
                params_list,
                fdw_private,
                fdw_scan_tlist,
                fdw_recheck_quals,
                outer_plan);
}

extern "C" void
parquetBeginForeignScan(ForeignScanState *node, int eflags)
{
  #if 0
    ParquetFdwExecutionState *festate; 
	  ForeignScan    *plan = (ForeignScan *) node->ss.ps.plan;
	  EState         *estate = node->ss.ps.state;
    List           *fdw_private = plan->fdw_private;
    List           *attrs_used_list;
    List           *rowgroups_list;
    ListCell       *lc;
    char           *filename;
    std::set<int>   attrs_used;
    bool            use_mmap; 
    bool            use_threads;

    /* Unwrap fdw_private */
    filename = strVal((Value *) linitial(fdw_private));

    attrs_used_list = (List *) lsecond(fdw_private);
    foreach (lc, attrs_used_list)
        attrs_used.insert(lfirst_int(lc));

    use_mmap = (bool) intVal((Value *) lthird(fdw_private));
    use_threads = (bool) intVal((Value *) lfourth(fdw_private));

    MemoryContext cxt = estate->es_query_cxt;
    TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
    TupleDesc tupleDesc = slot->tts_tupleDescriptor;

    try
    {
        festate = create_parquet_state(filename,
                                       tupleDesc,
                                       cxt,
                                       attrs_used,
                                       use_mmap,
                                       use_threads);
    }
    catch(const std::exception& e)
    {
        elog(ERROR, "parquet_fdw: parquet initialization failed: %s", e.what());
    }

    rowgroups_list = (List *) llast(fdw_private);
    foreach (lc, rowgroups_list)
        festate->rowgroups.push_back(lfirst_int(lc));

    node->fdw_state = festate;
  #endif
    ForeignScan *fsplan = (ForeignScan *) node->ss.ps.plan;
    EState     *estate = node->ss.ps.state;
    PgFdwScanState *fsstate;
    RangeTblEntry *rte;
    Oid     userid;
    ForeignTable *table;
    UserMapping *user;
    int     rtindex;
    int     numParams;

    /*
     * Do nothing in EXPLAIN (no ANALYZE) case.  node->fdw_state stays NULL.
     */
    if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
      return;

    /*
     * We'll save private state in node->fdw_state.
     */
    fsstate = (PgFdwScanState *) palloc0(sizeof(PgFdwScanState));
    node->fdw_state = (void *) fsstate;

    /*
     * Identify which user to do the remote access as.  This should match what
     * ExecCheckRTEPerms() does.  In case of a join or aggregate, use the
     * lowest-numbered member RTE as a representative; we would get the same
     * result from any.
     */
    if (fsplan->scan.scanrelid > 0)
      rtindex = fsplan->scan.scanrelid;
    else
      rtindex = bms_next_member(fsplan->fs_relids, -1);
    rte = exec_rt_fetch(rtindex, estate);
    userid = rte->checkAsUser ? rte->checkAsUser : GetUserId();

    /* Get info about foreign table. */
    table = GetForeignTable(rte->relid);
    user = GetUserMapping(userid, table->serverid);

    /*
     * Get connection to the foreign server.  Connection manager will
     * establish new connection if necessary.
     */
    //fsstate->conn = GetConnection(user, false);

    /* Assign a unique ID for my cursor */
    //fsstate->cursor_number = GetCursorNumber(fsstate->conn);
    fsstate->cursor_exists = false;

    fsstate->filename = strVal(list_nth(fsplan->fdw_private,
                     FdwScanPrivateFilename));
    /* Get private info created by planner functions. */
    fsstate->query = strVal(list_nth(fsplan->fdw_private,
                     FdwScanPrivateSelectSql));
    fsstate->retrieved_attrs = (List *) list_nth(fsplan->fdw_private,
                           FdwScanPrivateRetrievedAttrs);
    fsstate->fetch_size = intVal(list_nth(fsplan->fdw_private,
                        FdwScanPrivateFetchSize));

    /* Create contexts for batches of tuples and per-tuple temp workspace. */
    fsstate->batch_cxt = AllocSetContextCreate(estate->es_query_cxt,
                           "postgres_fdw tuple data",
                           ALLOCSET_DEFAULT_SIZES);
    fsstate->temp_cxt = AllocSetContextCreate(estate->es_query_cxt,
                          "postgres_fdw temporary data",
                          ALLOCSET_SMALL_SIZES);

    /*
     * Get info we'll need for converting data fetched from the foreign server
     * into local representation and error reporting during that process.
     */
    if (fsplan->scan.scanrelid > 0)
    {
      fsstate->rel = node->ss.ss_currentRelation;
      fsstate->tupdesc = RelationGetDescr(fsstate->rel);
    }
    else
    {
      fsstate->rel = NULL;
      fsstate->tupdesc = node->ss.ss_ScanTupleSlot->tts_tupleDescriptor;
    }

    fsstate->attinmeta = TupleDescGetAttInMetadata(fsstate->tupdesc);

    /*
     * Prepare for processing of parameters used in remote query, if any.
     */
    numParams = list_length(fsplan->fdw_exprs);
    fsstate->numParams = numParams;
    if (numParams > 0)
      prepare_query_params((PlanState *) node,
                 fsplan->fdw_exprs,
                 numParams,
                 &fsstate->param_flinfo,
                 &fsstate->param_exprs,
                 &fsstate->param_values);

}

static int
get_arrow_list_elem_type(arrow::DataType *type)
{
    auto children = type->children();

    Assert(children.size() == 1);
    return children[0]->type()->id();
}

/*
 * initialize_castfuncs
 *      Check wether implicit cast will be required and prepare cast function
 *      call. For arrays find cast functions for its elements.
 */
static void
initialize_castfuncs(ParquetFdwExecutionState *festate, TupleDesc tupleDesc)
{
    festate->castfuncs.resize(festate->map.size());

    for (uint i = 0; i < festate->map.size(); ++i)
    {
        int arrow_col = festate->map[i];

        if (festate->map[i] < 0)
        {
            /* Null column */
            festate->castfuncs[i] = NULL;
            continue;
        }

        arrow::DataType *type = festate->types[arrow_col];
        int     type_id = type->id();
        int     src_type,
                dst_type;
        bool    src_is_list,
                dst_is_array;
        Oid     funcid;
        CoercionPathType ct;

        /* Find underlying type of list */
        src_is_list = (type_id == arrow::Type::LIST);
        if (src_is_list)
            type_id = get_arrow_list_elem_type(type);

        src_type = to_postgres_type(type_id);
        dst_type = TupleDescAttr(tupleDesc, i)->atttypid;

        if (!OidIsValid(src_type))
            elog(ERROR, "parquet_fdw: unsupported column type: %s",
                 type->name().c_str());

        /* Find underlying type of array */
        dst_is_array = type_is_array(dst_type);
        if (dst_is_array)
            dst_type = get_element_type(dst_type);

        /* Make sure both types are compatible */
        if (src_is_list != dst_is_array)
        {
            ereport(ERROR,
                    (errcode(ERRCODE_FDW_INVALID_DATA_TYPE),
                     errmsg("parquet_fdw: incompatible types in column \"%s\"",
                            festate->table->field(arrow_col)->name().c_str()),
                     errhint(src_is_list ?
                         "parquet column is of type list while postgres type is scalar" :
                         "parquet column is of scalar type while postgres type is array")));
        }

        if (IsBinaryCoercible(src_type, dst_type))
        {
            festate->castfuncs[i] = NULL;
            continue;
        }

        ct = find_coercion_pathway(dst_type,
                                   src_type,
                                   COERCION_EXPLICIT,
                                   &funcid);
        switch (ct)
        {
            case COERCION_PATH_FUNC:
                {
                    MemoryContext   oldctx;
                    
                    oldctx = MemoryContextSwitchTo(CurTransactionContext);
                    festate->castfuncs[i] = (FmgrInfo *) palloc0(sizeof(FmgrInfo));
                    fmgr_info(funcid, festate->castfuncs[i]);
                    MemoryContextSwitchTo(oldctx);
                    break;
                }
            case COERCION_PATH_RELABELTYPE:
            case COERCION_PATH_COERCEVIAIO:  /* TODO: double check that we
                                              * shouldn't do anything here*/
                /* Cast is not needed */
                festate->castfuncs[i] = NULL;
                break;
            default:
                elog(ERROR, "parquet_fdw: cast function is not found");
        }
    }
    festate->initialized = true;
}

/*
 * fast_alloc
 *      Preallocate a big memory segment and distribute blocks from it. When
 *      segment is exhausted it is added to garbage_segments list and freed
 *      on the next executor's iteration. If requested size is bigger that
 *      SEGMENT_SIZE then just palloc is used.
 */
static inline void *
fast_alloc(ParquetFdwExecutionState *festate, long size)
{
    void   *ret;

    Assert(size >= 0);

    /* If allocation is bigger than segment then just palloc */
    if (size > SEGMENT_SIZE)
        return palloc(size);

    size = MAXALIGN(size);

    /* If there is not enough space in current segment create a new one */
    if (festate->segment_last_ptr - festate->segment_cur_ptr < size)
    {
        MemoryContext oldcxt;
        
        /*
         * Recycle the last segment at the next iteration (if there
         * was one)
         */
        if (festate->segment_start_ptr)
            festate->garbage_segments.
                push_back(festate->segment_start_ptr);

        oldcxt = MemoryContextSwitchTo(festate->segments_cxt);
        festate->segment_start_ptr = (char *) palloc(SEGMENT_SIZE);
        festate->segment_cur_ptr = festate->segment_start_ptr;
        festate->segment_last_ptr =
            festate->segment_start_ptr + SEGMENT_SIZE - 1;
        MemoryContextSwitchTo(oldcxt);
    }

    ret = (void *) festate->segment_cur_ptr;
    festate->segment_cur_ptr += size;

    return ret;
}

/*
 * read_primitive_type
 *      Returns primitive type value from arrow array
 */
static Datum
read_primitive_type(ParquetFdwExecutionState *festate,
                    arrow::Array *array,
                    int type_id, int64_t i,
                    FmgrInfo *castfunc)
{
    Datum   res;

    /* Get datum depending on the column type */
    switch (type_id)
    {
        case arrow::Type::BOOL:
        {
            arrow::BooleanArray *boolarray = (arrow::BooleanArray *) array;

            res = BoolGetDatum(boolarray->Value(i));
            break;
        }
        case arrow::Type::INT32:
        {
            arrow::Int32Array *intarray = (arrow::Int32Array *) array;
            int value = intarray->Value(i);

            res = Int32GetDatum(value);
            break;
        }
        case arrow::Type::INT64:
        {
            arrow::Int64Array *intarray = (arrow::Int64Array *) array;
            int64 value = intarray->Value(i);

            res = Int64GetDatum(value);
            break;
        }
        case arrow::Type::FLOAT:
        {
            arrow::FloatArray *farray = (arrow::FloatArray *) array;
            float value = farray->Value(i);

            res = Float4GetDatum(value);
            break;
        }
        case arrow::Type::DOUBLE:
        {
            arrow::DoubleArray *darray = (arrow::DoubleArray *) array;
            double value = darray->Value(i);

            res = Float8GetDatum(value);
            break;
        }
        case arrow::Type::STRING:
        case arrow::Type::BINARY:
        {
            arrow::BinaryArray *binarray = (arrow::BinaryArray *) array;

            int32_t vallen = 0;
            const char *value = reinterpret_cast<const char*>(binarray->GetValue(i, &vallen));

            /* Build bytea */
            int64 bytea_len = vallen + VARHDRSZ;
            bytea *b = (bytea *) fast_alloc(festate, bytea_len);
            SET_VARSIZE(b, bytea_len);
            memcpy(VARDATA(b), value, vallen);

            res = PointerGetDatum(b);
            break;
        }
        case arrow::Type::TIMESTAMP:
        {
            /* TODO: deal with timezones */
            TimestampTz ts;
            arrow::TimestampArray *tsarray = (arrow::TimestampArray *) array;
            auto tstype = (arrow::TimestampType *) array->type().get();

            to_postgres_timestamp(tstype, tsarray->Value(i), ts);
            res = TimestampGetDatum(ts);
            break;
        }
        case arrow::Type::DATE32:
        {
            arrow::Date32Array *tsarray = (arrow::Date32Array *) array;
            int32 d = tsarray->Value(i);

            /*
             * Postgres date starts with 2000-01-01 while unix date (which
             * Parquet is using) starts with 1970-01-01. So we need to do
             * simple calculations here.
             */
            res = DateADTGetDatum(d + (UNIX_EPOCH_JDATE - POSTGRES_EPOCH_JDATE));
            break;
        }
        /* TODO: add other types */
        default:
            elog(ERROR,
                 "parquet_fdw: unsupported column type: %d",
                 type_id);
    }

    /* Call cast function if needed */
    if (castfunc != NULL)
        return FunctionCall1(castfunc, res);

    return res;
}

/*
 * GetPrimitiveValues
 *      Get plain C value array. Copy-pasted from Arrow.
 */
template <typename T>
inline const T* GetPrimitiveValues(const arrow::Array& arr) {
  if (arr.length() == 0) {
    return nullptr;
  }
  const auto& prim_arr = arrow::internal::checked_cast<const arrow::PrimitiveArray&>(arr);
  const T* raw_values = reinterpret_cast<const T*>(prim_arr.values()->data());
  return raw_values + arr.offset();
}

/* 
 * copy_to_c_array
 *      memcpy plain values from Arrow array to a C array.
 */
template<typename T> inline void
copy_to_c_array(T *values, const arrow::Array *array, int elem_size)
{
    const T *in = GetPrimitiveValues<T>(*array);

    memcpy(values, in, elem_size * array->length());
}

/*
 * nested_list_get_datum
 *      Returns postgres array build from elements of array. Only one
 *      dimensional arrays are supported.
 */
static Datum
nested_list_get_datum(ParquetFdwExecutionState *festate,
                      arrow::Array *array, int type_id,
                      Oid elem_type, FmgrInfo *castfunc)
{
    ArrayType  *res;
    Datum      *values;
    bool       *nulls = NULL;
    int16       elem_len;
    bool        elem_byval;
    char        elem_align;
    int         dims[1];
    int         lbs[1];

    values = (Datum *) fast_alloc(festate, sizeof(Datum) * array->length());
    get_typlenbyvalalign(elem_type, &elem_len, &elem_byval, &elem_align);

    /* Fill values and nulls arrays */
    if (array->null_count() == 0 && type_id == arrow::Type::INT64)
    {
        /*
         * Ok, there are no nulls, so probably we could just memcpy the
         * entire array.
         *
         * Warning: the code below is based on the assumption that Datum is
         * 8 bytes long, which is true for most contemporary systems but this
         * will not work on some exotic or really old systems. In this case
         * the entire "if" branch should just be removed.
         */
        copy_to_c_array<int64_t>((int64_t *) values, array, elem_len);
        goto construct_array;
    }
    /* Fill values and nulls arrays */
    if (array->null_count() == 0 && type_id == arrow::Type::FLOAT)
    {
        /*
         * Ok, there are no nulls, so probably we could just memcpy the
         * entire array.
         *
         * Warning: the code below is based on the assumption that Datum is
         * 8 bytes long, which is true for most contemporary systems but this
         * will not work on some exotic or really old systems. In this case
         * the entire "if" branch should just be removed.
         */
        copy_to_c_array<float>((float *) values, array, elem_len);
        goto construct_array;
    }
    for (int64_t i = 0; i < array->length(); ++i)
    {
        if (!array->IsNull(i))
            values[i] = read_primitive_type(festate, array, type_id, i, castfunc);
        else
        {
            if (!nulls)
            {
                Size size = sizeof(bool) * array->length();

                nulls = (bool *) fast_alloc(festate, size);
                memset(nulls, 0, size);
            }
            nulls[i] = true;
        }
    }

construct_array:
    /* Construct one dimensional array */
    dims[0] = array->length();
    lbs[0] = 1;
    res = construct_md_array(values, nulls, 1, dims, lbs,
                             elem_type, elem_len, elem_byval, elem_align);

    return PointerGetDatum(res);
}

/*
 * populate_slot
 *      Fill slot with the values from parquet row.
 *
 * If `fake` set to true the actual reading and populating the slot is skipped.
 * The purpose of this feature is to correctly skip rows to collect sparse
 * samples.
 */
static void
populate_slot(ParquetFdwExecutionState *festate,
              TupleTableSlot *slot,
              bool fake=false)
{
    /* Fill slot values */
    for (int attr = 0; attr < slot->tts_tupleDescriptor->natts; attr++)
    {
        int arrow_col = festate->map[attr];
        /*
         * We only fill slot attributes if column was referred in targetlist
         * or clauses. In other cases mark attribute as NULL.
         * */
        if (arrow_col >= 0)
        {
            ChunkInfo &chunkInfo = festate->chunk_info[arrow_col];
            arrow::Array       *array = festate->chunks[arrow_col];
            arrow::DataType    *arrow_type = festate->types[arrow_col];
            int                 arrow_type_id = arrow_type->id();

            chunkInfo.len = array->length();

            if (chunkInfo.pos >= chunkInfo.len)
            {
                auto column = festate->table->column(arrow_col);

                /* There are no more chunks */
                if (++chunkInfo.chunk >= column->num_chunks())
                    break;

                array = column->chunk(chunkInfo.chunk).get();
                festate->chunks[arrow_col] = array;
                chunkInfo.pos = 0;
                chunkInfo.len = array->length();
            }

            /* Don't do actual reading data into slot in fake mode */
            if (fake)
                continue;

            /* Currently only primitive types and lists are supported */
            if (arrow_type_id != arrow::Type::LIST)
            {
                if (festate->has_nulls[arrow_col] && array->IsNull(chunkInfo.pos))
                {
                    slot->tts_isnull[attr] = true;
                }
                else
                {
                    slot->tts_values[attr] = 
                        read_primitive_type(festate,
                                            array,
                                            arrow_type_id,
                                            chunkInfo.pos,
                                            festate->castfuncs[attr]);
                    slot->tts_isnull[attr] = false;
                }
            }
            else
            {
                Oid     pg_type_id;

                pg_type_id = TupleDescAttr(slot->tts_tupleDescriptor, attr)->atttypid;
                if (!type_is_array(pg_type_id))
                    elog(ERROR,
                         "parquet_fdw: cannot convert parquet column of type "
                         "LIST to postgres column of scalar type");

                /* Figure out the base element types */
                pg_type_id = get_element_type(pg_type_id);
                arrow_type_id = get_arrow_list_elem_type(arrow_type);

                int64 pos = chunkInfo.pos;
                arrow::ListArray   *larray = (arrow::ListArray *) array;

                if (festate->has_nulls[arrow_col] && array->IsNull(pos))
                {
                    slot->tts_isnull[attr] = true;
                }
                else
                {
                    std::shared_ptr<arrow::Array> slice =
                        larray->values()->Slice(larray->value_offset(pos),
                                                larray->value_length(pos));

                    slot->tts_values[attr] =
                        nested_list_get_datum(festate,
                                              slice.get(),
                                              arrow_type_id,
                                              pg_type_id,
                                              festate->castfuncs[attr]);
                    slot->tts_isnull[attr] = false;
                }
            }

            chunkInfo.pos++;
        }
        else
        {
            slot->tts_isnull[attr] = true;
        }
    }
}

/*
 * bytes_to_postgres_type
 *      Convert min/max values from column statistics stored in parquet file as
 *      plain bytes to postgres Datum.
 */
static Datum
bytes_to_postgres_type(const char *bytes, arrow::DataType *arrow_type)
{
    switch(arrow_type->id())
    {
        case arrow::Type::BOOL:
            return BoolGetDatum(*(bool *) bytes);
        case arrow::Type::INT32:
            return Int32GetDatum(*(int32 *) bytes);
        case arrow::Type::INT64:
            return Int64GetDatum(*(int64 *) bytes);
        case arrow::Type::FLOAT:
            return Int32GetDatum(*(float *) bytes);
        case arrow::Type::DOUBLE:
            return Int64GetDatum(*(double *) bytes);
        case arrow::Type::STRING:
        case arrow::Type::BINARY:
            return CStringGetTextDatum(bytes);
        case arrow::Type::TIMESTAMP:
            {
                TimestampTz ts;
                auto tstype = (arrow::TimestampType *) arrow_type;

                to_postgres_timestamp(tstype, *(int64 *) bytes, ts);
                return TimestampGetDatum(ts);
            }
        case arrow::Type::DATE32:
            return DateADTGetDatum(*(int32 *) bytes +
                                   (UNIX_EPOCH_JDATE - POSTGRES_EPOCH_JDATE));
        default:
            return PointerGetDatum(NULL);
    }
}

/*
 * find_cmp_func
 *      Find comparison function for two given types.
 */
static void
find_cmp_func(FmgrInfo *finfo, Oid type1, Oid type2)
{
    Oid cmp_proc_oid;
    TypeCacheEntry *tce_1, *tce_2;

    tce_1 = lookup_type_cache(type1, TYPECACHE_BTREE_OPFAMILY);
    tce_2 = lookup_type_cache(type2, TYPECACHE_BTREE_OPFAMILY);

    cmp_proc_oid = get_opfamily_proc(tce_1->btree_opf,
                                     tce_1->btree_opintype,
                                     tce_2->btree_opintype,
                                     BTORDER_PROC);
    fmgr_info(cmp_proc_oid, finfo);
}

static bool
read_next_rowgroup(ParquetFdwExecutionState *festate, TupleDesc tupleDesc)
{
    ParallelCoordinator        *coord;
    arrow::Status               status;

    coord = festate->coordinator;

    /*
     * Use atomic increment for parallel query or just regular one for single
     * threaded execution.
     */
    if (coord)
        festate->row_group = pg_atomic_fetch_add_u32(&coord->next_rowgroup, 1);
    else
        festate->row_group++;

    /*
     * row_group cannot be less than zero at this point so it is safe to cast
     * it to unsigned int
     */
    if ((uint) festate->row_group >= festate->rowgroups.size())
        return false;

    int  rowgroup = festate->rowgroups[festate->row_group];
    auto rowgroup_meta = festate->reader
                            ->parquet_reader()
                            ->metadata()
                            ->RowGroup(rowgroup);

    /* Determine which columns has null values */
    for (uint i = 0; i < festate->map.size(); i++)
    {
        std::shared_ptr<parquet::Statistics>  stats;
        int arrow_col = festate->map[i];

        if (arrow_col < 0)
            continue;

        stats = rowgroup_meta
            ->ColumnChunk(festate->indices[arrow_col])
            ->statistics();

        if (stats)
            festate->has_nulls[arrow_col] = (stats->null_count() > 0);
        else
            festate->has_nulls[arrow_col] = true;
    }

    status = festate->reader
        ->RowGroup(rowgroup)
        ->ReadTable(festate->indices, &festate->table);

    if (!status.ok())
        throw std::runtime_error(status.message().c_str());

    if (!festate->table)
        throw std::runtime_error("got empty table");

    /* Fill festate->columns and festate->types */
    /* TODO: don't clear each time */
    festate->chunk_info.clear();
    festate->chunks.clear();
    for (int i = 0; i < tupleDesc->natts; i++)
    {
        if (festate->map[i] >= 0)
        {
            ChunkInfo chunkInfo = { .chunk = 0, .pos = 0, .len = 0 };
            auto column = festate->table->column(festate->map[i]);

            festate->chunk_info.push_back(chunkInfo);
            festate->chunks.push_back(column->chunk(0).get());
        }
    }

    festate->row = 0;
    festate->num_rows = festate->table->num_rows();

    return true;
}

extern "C" TupleTableSlot *
parquetIterateForeignScan(ForeignScanState *node)
{
  #if 0
    ParquetFdwExecutionState   *festate = (ParquetFdwExecutionState *) node->fdw_state;
	TupleTableSlot             *slot = node->ss.ss_ScanTupleSlot;

	ExecClearTuple(slot);

    /* recycle old segments if any */
    if (!festate->garbage_segments.empty())
    {
        for (auto it : festate->garbage_segments)
            pfree(it);
        festate->garbage_segments.clear();
        elog(DEBUG1, "parquet_fdw: garbage segments recycled");
    }

    if (festate->row >= festate->num_rows)
    {
        /* Read next row group */
        try
        {
            if (!read_next_rowgroup(festate, slot->tts_tupleDescriptor))
                return slot;
        }
        catch(const std::exception& e)
        {
            elog(ERROR,
                 "parquet_fdw: failed to read row group %d: %s",
                 festate->row_group, e.what());
        }

        /* Lookup cast funcs */
        if (!festate->initialized)
            initialize_castfuncs(festate, slot->tts_tupleDescriptor);
    }

    populate_slot(festate, slot);
    festate->row++;
    ExecStoreVirtualTuple(slot);

    return slot;
  #endif
    PgFdwScanState *fsstate = (PgFdwScanState *) node->fdw_state;
    TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;

    /*
     * If this is the first call after Begin or ReScan, we need to create the
     * cursor on the remote side.
     */
    if (!fsstate->cursor_exists) {
      create_cursor(node);
    }

    /*
     * Get some more tuples, if we've run out.
     */
    if (fsstate->next_tuple >= fsstate->num_tuples)
    {
      /* No point in another fetch if we already detected EOF, though. */
      if (!fsstate->eof_reached) {
        fetch_more_data(node);
      }
      /* If we didn't get any tuples, must be end of data. */
      if (fsstate->next_tuple >= fsstate->num_tuples)
        return ExecClearTuple(slot);
    }

    /*
     * Return the next tuple.
     */
    ExecStoreHeapTuple(fsstate->tuples[fsstate->next_tuple++],
               slot,
               false);

    return slot;
}

extern "C" void
parquetEndForeignScan(ForeignScanState *node)
{
  #if 0
    ParquetFdwExecutionState *festate = (ParquetFdwExecutionState *) node->fdw_state;

    /* 
     * Disable autodestruction to prevent double freeing of the execution
     * state object. I could just remove `delete festate` below and let the
     * memory context callback do its job. But it is more obvious for readers
     * to see an explicit destruction of the execution state.
     */
    festate->autodestroy = false;

    delete festate;
  #endif
    PgFdwScanState *fsstate = (PgFdwScanState *) node->fdw_state;

    /* if fsstate is NULL, we are in EXPLAIN; nothing to do */
    if (fsstate == NULL)
      return;

    /* Close the cursor if open, to prevent accumulation of cursors */
    if (fsstate->cursor_exists) {
      close_cursor(fsstate->conn, fsstate->cursor_number);
    }

    /* Release remote connection */
    //ReleaseConnection(fsstate->conn);
    fsstate->conn = NULL;


}

extern "C" void
parquetReScanForeignScan(ForeignScanState *node)
{
#if 0
    ParquetFdwExecutionState   *festate = (ParquetFdwExecutionState *) node->fdw_state;

    festate->row_group = 0;
    festate->row = 0;
    festate->num_rows = 0;
#endif
    PgFdwScanState *fsstate = (PgFdwScanState *) node->fdw_state;
    char    sql[64];
    //PGresult   *res;

    /* If we haven't created the cursor yet, nothing to do. */
    if (!fsstate->cursor_exists)
      return;

    /*
     * If any internal parameters affecting this node have changed, we'd
     * better destroy and recreate the cursor.  Otherwise, rewinding it should
     * be good enough.  If we've only fetched zero or one batch, we needn't
     * even rewind the cursor, just rescan what we have.
     */
    if (fsstate->fetch_ct_2 > 1)
    {
      snprintf(sql, sizeof(sql), "MOVE BACKWARD ALL IN c%u",
           fsstate->cursor_number);
    }
    else
    {
      /* Easy: just rescan what we already have in memory, if anything */
      fsstate->next_tuple = 0;
      return;
    }

    /*
     * We don't use a PG_TRY block here, so be careful not to throw error
     * without releasing the PGresult.
     */
    // res = pgfdw_exec_query(fsstate->conn, sql);
    // if (PQresultStatus(res) != PGRES_COMMAND_OK)
    //   pgfdw_report_error(ERROR, res, fsstate->conn, true, sql);
    // PQclear(res);

    /* Now force a fresh FETCH. */
    fsstate->tuples = NULL;
    fsstate->num_tuples = 0;
    fsstate->next_tuple = 0;
    fsstate->fetch_ct_2 = 0;
    fsstate->eof_reached = false;
}

static int
parquetAcquireSampleRowsFunc(Relation relation, int elevel,
                             HeapTuple *rows, int targrows,
                             double *totalrows,
                             double *totaldeadrows)
{
    ParquetFdwExecutionState   *festate;
    ParquetFdwPlanState         fdw_private;
    TupleDesc       tupleDesc = RelationGetDescr(relation);
    TupleTableSlot *slot;
    std::set<int>   attrs_used;
    int cnt = 0;

    get_table_options(RelationGetRelid(relation), &fdw_private);

    for (int i = 0; i < tupleDesc->natts; ++i)
        attrs_used.insert(i + 1 - FirstLowInvalidHeapAttributeNumber);

    /* Open parquet file and build execution state */
    try
    {
        festate = create_parquet_state(fdw_private.filename,
                                       tupleDesc,
                                       CurrentMemoryContext,
                                       attrs_used,
                                       false,
                                       fdw_private.use_threads);
        festate->autodestroy = false;
    }
    catch(const std::exception& e)
    {
        elog(ERROR, "parquet_fdw: parquet initialization failed: %s", e.what());
    }

    PG_TRY();
    {
        auto meta = festate->reader->parquet_reader()->metadata();
        int ratio = meta->num_rows() / targrows;

        /* Set ratio to at least 1 to avoid devision by zero issue */
        ratio = ratio < 1 ? 1 : ratio;

        /* We need to scan all rowgroups */
        for (int i = 0; i < meta->num_row_groups(); ++i)
            festate->rowgroups.push_back(i);

#if PG_VERSION_NUM < 120000
        slot = MakeSingleTupleTableSlot(tupleDesc);
#else
        slot = MakeSingleTupleTableSlot(tupleDesc, &TTSOpsHeapTuple);
#endif

        initialize_castfuncs(festate, tupleDesc);

        while (true)
        {
            CHECK_FOR_INTERRUPTS();

            if (cnt >= targrows)
                break;

            /* recycle old segments if any */
            if (!festate->garbage_segments.empty())
            {
                for (auto it : festate->garbage_segments)
                    pfree(it);
                festate->garbage_segments.clear();
                elog(DEBUG1, "parquet_fdw: garbage segments recycled");
            }

            if (festate->row >= festate->num_rows)
            {
                /* Read next row group */
                try
                {
                    if (!read_next_rowgroup(festate, tupleDesc))
                        break;
                }
                catch(const std::exception& e)
                {
                    elog(ERROR,
                         "parquet_fdw: failed to read row group %d: %s",
                         festate->row_group, e.what());
                }
            }

            bool fake = (festate->row % ratio) != 0;

            populate_slot(festate, slot, fake);

            if (!fake)
            {
                rows[cnt++] = heap_form_tuple(tupleDesc,
                                              slot->tts_values,
                                              slot->tts_isnull);
            }

            festate->row++;
        }

        *totalrows = meta->num_rows();
        *totaldeadrows = 0;

        ExecDropSingleTupleTableSlot(slot);
    }
    PG_CATCH();
    {
        elog(LOG, "Cancelled");
        delete festate;
        PG_RE_THROW();
    }
    PG_END_TRY();

    delete festate;

    return cnt - 1;
}

extern "C" bool
parquetAnalyzeForeignTable (Relation relation,
                            AcquireSampleRowsFunc *func,
                            BlockNumber *totalpages)
{
    *func = parquetAcquireSampleRowsFunc;
    return true;
}

/*
 * parquetExplainForeignScan
 *      Additional explain information, namely row groups list.
 */
extern "C" void
parquetExplainForeignScan(ForeignScanState *node, ExplainState *es)
{
  #if 0
    List	   *fdw_private;
    List       *rowgroups;
    ListCell   *lc;
    StringInfoData str;
    bool        is_first = true;

    initStringInfo(&str);

	   fdw_private = ((ForeignScan *) node->ss.ps.plan)->fdw_private;
    rowgroups = (List *) llast(fdw_private);

    foreach(lc, rowgroups)
    {
        /*
         * As parquet-tools use 1 based indexing for row groups it's probably
         * a good idea to output row groups numbers in the same way.
         */
        int rowgroup = lfirst_int(lc) + 1;

        if (is_first)
        {
            appendStringInfo(&str, "%i", rowgroup);
            is_first = false;
        }
        else
            appendStringInfo(&str, ", %i", rowgroup);
    }

    ExplainPropertyText("Row groups", str.data, es);
  #endif
    List     *fdw_private;
    char     *sql;
    char     *relations;

    fdw_private = ((ForeignScan *) node->ss.ps.plan)->fdw_private;

    /*
     * Add names of relation handled by the foreign scan when the scan is a
     * join
     */
    if (list_length(fdw_private) > FdwScanPrivateRelations)
    {
      relations = strVal(list_nth(fdw_private, FdwScanPrivateRelations));
      ExplainPropertyText("Relations", relations, es);
    }

    /*
     * Add remote query, when VERBOSE option is specified.
     */
    if (es->verbose)
    {
      sql = strVal(list_nth(fdw_private, FdwScanPrivateSelectSql));
      ExplainPropertyText("Remote SQL", sql, es);
    }
}

/* Parallel query execution */

extern "C" bool
parquetIsForeignScanParallelSafe(PlannerInfo *root, RelOptInfo *rel,
                                 RangeTblEntry *rte)
{
    /* Use parallel execution only when statistics are collected */
    return (rel->tuples > 0);
}

extern "C" Size
parquetEstimateDSMForeignScan(ForeignScanState *node, ParallelContext *pcxt)
{
    return sizeof(ParallelCoordinator);
}

extern "C" void
parquetInitializeDSMForeignScan(ForeignScanState *node, ParallelContext *pcxt,
                                void *coordinate)
{
    ParallelCoordinator        *coord = (ParallelCoordinator *) coordinate;
    ParquetFdwExecutionState   *festate;

    pg_atomic_write_u32(&coord->next_rowgroup, 0);
    festate = (ParquetFdwExecutionState *) node->fdw_state;
    festate->coordinator = coord;
}

extern "C" void
parquetReInitializeDSMForeignScan(ForeignScanState *node,
                                  ParallelContext *pcxt, void *coordinate)
{
    ParallelCoordinator    *coord = (ParallelCoordinator *) coordinate;

    pg_atomic_write_u32(&coord->next_rowgroup, 0);
}

extern "C" void
parquetInitializeWorkerForeignScan(ForeignScanState *node,
                                   shm_toc *toc,
                                   void *coordinate)
{
    ParallelCoordinator        *coord   = (ParallelCoordinator *) coordinate;
    ParquetFdwExecutionState   *festate;

    festate = (ParquetFdwExecutionState *) node->fdw_state;
    festate->coordinator = coord;
}

extern "C" void
parquetShutdownForeignScan(ForeignScanState *node)
{
    ParquetFdwExecutionState   *festate;

    festate = (ParquetFdwExecutionState *) node->fdw_state;
    festate->coordinator = NULL;
}

/*
 * extract_parquet_fields
 *      Read parquet file and return a list of its fields
 */
std::list<std::pair<std::string, Oid> >
extract_parquet_fields(ImportForeignSchemaStmt *stmt, const char *path)
{
    std::list<std::pair<std::string, Oid> >     res;
    std::unique_ptr<parquet::arrow::FileReader> reader;
    std::shared_ptr<arrow::Schema>              schema;

    try
    {
        parquet::arrow::FileReader::Make(
                arrow::default_memory_pool(),
                parquet::ParquetFileReader::OpenFile(path, false),
                &reader
        );

    }
    catch(const std::exception& e)
    {
        elog(ERROR, "parquet_fdw: parquet initialization failed: %s", e.what());
    }

    PG_TRY();
    {
        auto meta = reader->parquet_reader()->metadata();
        parquet::ArrowReaderProperties props;

        if (!parquet::arrow::FromParquetSchema(meta->schema(), props, &schema).ok())
            elog(ERROR, "parquet_fdw: error reading parquet schema");

        for (int k = 0; k < schema->num_fields(); ++k)
        {
            std::shared_ptr<arrow::Field>       field;
            std::shared_ptr<arrow::DataType>    type;
            Oid     pg_type;

            /* Convert to arrow field to get appropriate type */
            field = schema->field(k);
            type = field->type();

            if (type->id() == arrow::Type::LIST)
            {
                int subtype_id;
                Oid pg_subtype;

                if (type->children().size() != 1)
                    elog(ERROR, "parquet_fdw: lists of structs are not supported");

                subtype_id = get_arrow_list_elem_type(type.get());
                pg_subtype = to_postgres_type(subtype_id);

                pg_type = get_array_type(pg_subtype);
            }
            else
            {
                pg_type = to_postgres_type(type->id());
            }

            if (pg_type != InvalidOid)
            {
                res.push_back(std::pair<std::string, Oid>(field->name(), pg_type));
            }
            else
            {
                elog(ERROR,
                     "parquet_fdw: cannot convert field '%s' of type '%s' in %s",
                     field->name().c_str(), type->name().c_str(), path);
            }
        }
    }
    PG_CATCH();
    {
        /* Destroy the reader on error */
        reader.reset();
        PG_RE_THROW();
    }
    PG_END_TRY();

    return res;
}

/*
 * autodiscover_parquet_file
 *      Builds CREATE FOREIGN TABLE query based on specified parquet file
 */
static char *
autodiscover_parquet_file(ImportForeignSchemaStmt *stmt, char *filename)
{
    char           *path = psprintf("%s/%s", stmt->remote_schema, filename);
    StringInfoData  str;
    auto            fields = extract_parquet_fields(stmt, path);
    bool            is_first = true;
    char           *ext;
    ListCell       *lc;

    initStringInfo(&str);
    appendStringInfo(&str, "CREATE FOREIGN TABLE ");

    /* append table name */
    ext = strrchr(filename, '.');
    *ext = '\0';
    if (stmt->local_schema)
        appendStringInfo(&str, "%s.%s (",
                         stmt->local_schema, quote_identifier(filename));
    else
        appendStringInfo(&str, "%s (", quote_identifier(filename));
    *ext = '.';

    /* append columns */
    for (auto field: fields)
    {
        std::string &name = field.first;
        Oid pg_type = field.second;

        const char *type_name = format_type_be(pg_type);

        if (!is_first)
            appendStringInfo(&str, ", %s %s", name.c_str(), type_name);
        else
        {
            appendStringInfo(&str, "%s %s", name.c_str(), type_name);
            is_first = false;
        }
    }
    appendStringInfo(&str, ") SERVER %s ", stmt->server_name);
    appendStringInfo(&str, "OPTIONS (filename '%s'", path);

    /* append options */
    foreach (lc, stmt->options)
    {
		DefElem    *def = (DefElem *) lfirst(lc);

        appendStringInfo(&str, ", %s '%s'", def->defname, defGetString(def));
    }
    appendStringInfo(&str, ")");

    elog(DEBUG1, "parquet_fdw: %s", str.data);

    return str.data;
}

extern "C" List *
parquetImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid)
{
    struct dirent  *f;
    DIR            *d;
    List           *cmds = NIL;

    d = AllocateDir(stmt->remote_schema);
    if (!d)
    {
        int e = errno;

        elog(ERROR, "parquet_fdw: failed to open directory '%s': %s",
             stmt->remote_schema,
             strerror(e));
    }

    while ((f = readdir(d)) != NULL)
    {

        /* TODO: use lstat if d_type == DT_UNKNOWN */
        if (f->d_type == DT_REG)
        {
            ListCell   *lc;
            bool        skip = false;
            char       *filename = f->d_name;

            /* check that file extension is "parquet" */
            char *ext = strrchr(filename, '.');

            if (ext && strcmp(ext + 1, "parquet") != 0)
                continue;

            /*
             * Set terminal symbol to be able to run strcmp on filename
             * without file extension
             */
            *ext = '\0';

            foreach (lc, stmt->table_list)
            {
                RangeVar *rv = (RangeVar *) lfirst(lc);

                switch (stmt->list_type)
                {
                    case FDW_IMPORT_SCHEMA_LIMIT_TO:
                        if (strcmp(filename, rv->relname) != 0)
                        {
                            skip = true;
                            break;
                        }
                        break;
                    case FDW_IMPORT_SCHEMA_EXCEPT:
                        if (strcmp(filename, rv->relname) == 0)
                        {
                            skip = true;
                            break;
                        }
                        break;
                    default:
                        ;
                }
            }
            if (skip)
                continue;

            /* Return dot back */
            *ext = '.';
            cmds = lappend(cmds, autodiscover_parquet_file(stmt, filename));
        }

    }
    FreeDir(d);

    return cmds;
}

extern "C" List *
parquetPlanForeignModify(PlannerInfo *root,
                       ModifyTable *plan,
                       Index resultRelation,
                       int subplan_index)
{
    //elog(INFO,"foreign parquetPlanForeignModify");   
    if (plan->operation != CMD_INSERT)
        elog(ERROR, "not a supported operation on arrow_fdw foreign tables");

    return NIL;
}


extern "C" void
parquetExplainForeignModify(ModifyTableState *mtstate,
                          ResultRelInfo *rinfo,
                          List *fdw_private,
                          int subplan_index,
                          struct ExplainState *es)
{
    /* print something */
}


static std::pair<parquet::Type::type, parquet::ConvertedType::type>
to_parquet_type(int oid)
{
    // todo (yang)
    switch (oid)
    {
        case BOOLOID:
            return std::make_pair(parquet::Type::BOOLEAN, parquet::ConvertedType::NONE);
        case INT4OID:
        case INT4ARRAYOID:
            return std::make_pair(parquet::Type::INT32, parquet::ConvertedType::INT_32);
        case INT8OID:
        case INT8ARRAYOID:
            return std::make_pair(parquet::Type::INT64, parquet::ConvertedType::INT_64);
        case FLOAT4OID:
        case FLOAT4ARRAYOID:
            return std::make_pair(parquet::Type::FLOAT, parquet::ConvertedType::NONE);
        case FLOAT8OID:
            return std::make_pair(parquet::Type::DOUBLE, parquet::ConvertedType::NONE);
        case BYTEAOID:
            return std::make_pair(parquet::Type::BYTE_ARRAY, parquet::ConvertedType::NONE);
        case TEXTOID:
            return std::make_pair(parquet::Type::BYTE_ARRAY, parquet::ConvertedType::UTF8);
        case TIMESTAMPOID:
            return std::make_pair(parquet::Type::INT64, parquet::ConvertedType::TIMESTAMP_MICROS);
        case DATEOID:
            return std::make_pair(parquet::Type::INT32, parquet::ConvertedType::DATE);
        default:
            elog(ERROR, "unsupported oid %d", oid);
    }
}

static std::shared_ptr<parquet::schema::GroupNode> SetupSchema(TupleDesc tupdesc) {
    int     j;
    parquet::schema::NodeVector fields;

    for (j=0; j < tupdesc->natts; j++)
    {

        Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
        auto type_pair = to_parquet_type(attr->atttypid);

        if (type_is_array(attr->atttypid)) {
            auto element = parquet::schema::PrimitiveNode::Make("item",
                                             parquet::Repetition::OPTIONAL, type_pair.first, type_pair.second);
            auto array = parquet::schema::GroupNode::Make("list", parquet::Repetition::REPEATED,  {element});

            fields.push_back(parquet::schema::GroupNode::Make(NameStr(attr->attname), parquet::Repetition::OPTIONAL, {array},
                                     parquet::ConvertedType::LIST));
            // fields.push_back(parquet::schema::PrimitiveNode::Make(NameStr(attr->attname), parquet::Repetition::REPEATED,
            //                          type_pair.first, type_pair.second));
        } else {
            fields.push_back(parquet::schema::PrimitiveNode::Make(NameStr(attr->attname), parquet::Repetition::REQUIRED,
                                     type_pair.first, type_pair.second));
        }
        
    }
    
    // Create a GroupNode named 'schema' using the primitive nodes defined above
    // This GroupNode is the root node of the schema tree
    return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
}



static ParquetInsertState *
create_insert_state(const char *filename,
                     TupleDesc tupleDesc,
                     MemoryContext parent_cxt)
{
    ParquetInsertState *festate;

    festate = new ParquetInsertState(filename);
    
    /* Create mapping between tuple descriptor and parquet columns. */
    festate->schema = SetupSchema(tupleDesc);
    

    parquet::schema::PrintSchema(festate->schema.get(), std::cout);

    festate->memcxt = AllocSetContextCreate(parent_cxt,
                                                  "parquet_fdw tuple data",
                                                  ALLOCSET_DEFAULT_SIZES);

    //setup schema
    parquet::WriterProperties::Builder builder;
    std::shared_ptr<parquet::WriterProperties> props = builder.build();
    
    builder.compression(parquet::Compression::SNAPPY);

    festate->stream_writer = std::shared_ptr<parquet::StreamWriter2>(new parquet::StreamWriter2(\
        parquet::ParquetFileWriter::Open(festate->parquet_file, festate->schema, props)));
    
    return festate;
}



extern "C" void
parquetBeginForeignInsert(ModifyTableState *mtstate,
        ResultRelInfo *rrinfo)
{
    //elog(INFO,"foreign parquetBeginForeignInsert");   
    Oid  foreignTableOid = InvalidOid;
    TupleDesc tupleDescriptor = NULL;
    Relation relation = rrinfo->ri_RelationDesc;
    ParquetFdwPlanState     private_fdw;
    
    foreignTableOid = RelationGetRelid(relation);
    //relation = heap_open(foreignTableOid, ShareUpdateExclusiveLock);
    
    get_table_options(foreignTableOid, &private_fdw);
    tupleDescriptor = RelationGetDescr(relation);

    rrinfo->ri_FdwState = (void *)create_insert_state(private_fdw.filename, tupleDescriptor,
                                                CurrentMemoryContext);
}

extern "C" void
parquetBeginForeignModify(ModifyTableState *modifyTableState,
                         ResultRelInfo *rrinfo, List *fdwPrivate,
                         int subplanIndex, int executorFlags)
{
   //elog(INFO,"foreign parquetBeginForeignModify");   
    /* if Explain with no Analyze, do nothing */
    if (executorFlags & EXEC_FLAG_EXPLAIN_ONLY)
    {
        return;
    }

    Assert (modifyTableState->operation == CMD_INSERT);
    
    parquetBeginForeignInsert(modifyTableState, rrinfo);
}


#define ARRPTR(x)  ( (float *) ARR_DATA_PTR(x) )
#define ARRNELEMS(x)  ArrayGetNItems( ARR_NDIM(x), ARR_DIMS(x))

extern "C" TupleTableSlot *
parquetExecForeignInsert(EState *estate,
                       ResultRelInfo *rrinfo,
                       TupleTableSlot *slot,
                       TupleTableSlot *planSlot)
{
    Relation        frel = rrinfo->ri_RelationDesc;
    TupleDesc       tupdesc = RelationGetDescr(frel);
    ParquetInsertState *aw_state = (ParquetInsertState*) rrinfo->ri_FdwState;
    auto schema = aw_state->schema;
    auto os = aw_state->stream_writer;
    MemoryContext   oldcxt;
    int             j;

    slot_getallattrs(slot);
    oldcxt = MemoryContextSwitchTo(aw_state->memcxt);
    for (j=0; j < tupdesc->natts; j++)
    {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, j);
        Oid         valtype = attr->atttypid;
        Datum       datum = slot->tts_values[j];
        bool        isnull = slot->tts_isnull[j];

        try
        {
            if (isnull)
            {
                os->SkipColumns(1);
            }
            else if (attr->attbyval)
            {
               switch (valtype)
                {
                    case BOOLOID:
                    {
                        bool val = DatumGetBool(datum);
                        *os << val;
                        break;
                    }
                    case INT2OID:
                    {
                        const std::shared_ptr<parquet::schema::Node>& field = schema->field(j);
                        parquet::ConvertedType::type type = field->converted_type();
                        if (type == parquet::ConvertedType::INT_16) {
                            *os << DatumGetInt16(datum);
                        } else if (type == parquet::ConvertedType::UINT_16) {
                            *os << DatumGetUInt16(datum);
                        }
                        break;
                    }
                    case INT4OID:
                    {
                        const std::shared_ptr<parquet::schema::Node>& field = schema->field(j);
                        parquet::ConvertedType::type type = field->converted_type();
                        if (type == parquet::ConvertedType::INT_32) {
                            *os << DatumGetInt32(datum);
                        } else if (type == parquet::ConvertedType::UINT_32) {
                            *os << DatumGetUInt32(datum);
                        }
                        break;
                    }
                    case INT8OID:
                    {
                        const std::shared_ptr<parquet::schema::Node>& field = schema->field(j);
                        parquet::ConvertedType::type type = field->converted_type();
                        if (type == parquet::ConvertedType::INT_64) {
                            *os << (int64_t)DatumGetInt64(datum);
                        } else if (type == parquet::ConvertedType::UINT_64) {
                            *os << (uint64_t)DatumGetUInt64(datum);
                        }
                        break;
                    }
                    case FLOAT4OID:
                    {
                        float val = DatumGetFloat4(datum);
                        *os << val;
                        break;
                    }
                    case FLOAT8OID:
                    {
                        double val = DatumGetFloat8(datum);
                        *os << val;
                        break;
                    }
                    case TEXTOID:
                    {
                        char *s = TextDatumGetCString(datum);
                        *os << s;
                        break;
                    }
                    case DATEOID:
                    {
                        // Timestamp t = date2timestamp_no_overflow(DatumGetDateADT(datum));
                        // pg_time_t d = timestamptz_to_time_t(t);

                        
                        break;
                    }
                    case TIMESTAMPOID:
                    {
                        //pg_time_t d = timestamptz_to_time_t(DatumGetTimestamp(datum));

                        break;
                    }
                    default:
                    {
                        throw std::runtime_error("unexpected type " +
                                std::to_string(valtype) + " type ");
                    }
                }
            }
            else if (attr->attlen == -1)
            { 
                if (type_is_array(attr->atttypid)) {
                    Assert(attr->atttypid == FLOAT4ARRAYOID);
                    ArrayType* arr = DatumGetArrayTypeP(datum);
                    size_t num = ARRNELEMS(arr);
                    os->WriteFloatArray(ARRPTR(arr), num);
                } else {
                    //todo text
                    size_t     vl_len = VARSIZE_ANY_EXHDR(datum);
                    char   *vl_ptr = VARDATA_ANY(datum);
                    os->WriteVariableLength(vl_ptr, vl_len);
                }
                
            }
            else
            {
                elog(ERROR, "parquet_fdw: unsupported type format");
            }
        } 
        catch(const std::exception& e)
        {
            elog(ERROR, "parquet_fdw: parquet insert failed: %s", e.what());
        }
    }
    os->EndRow();

    MemoryContextSwitchTo(oldcxt);

    return slot;
}



extern "C" void
parquetEndForeignInsert(EState *estate,
                        ResultRelInfo *rrinfo)
{

    //elog(INFO, "parquetEndForeignInsert");
    ParquetInsertState *aw_state = (ParquetInsertState*) rrinfo->ri_FdwState;

    // if (aw_state) {
    //     delete aw_state;
    // }
    // aw_state = NULL;
    aw_state->stream_writer.reset();

}


extern "C" void
parquetEndForeignModify(EState *estate,
                      ResultRelInfo *rrinfo)
{
    //elog(INFO, "parquetEndForeignModify");
    parquetEndForeignInsert(estate, rrinfo);
}


/*
 * Assess whether the join between inner and outer relations can be pushed down
 * to the foreign server. As a side effect, save information we obtain in this
 * function to PgFdwRelationInfo passed in.
 */
static bool
foreign_join_ok(PlannerInfo *root, RelOptInfo *joinrel, JoinType jointype,
        RelOptInfo *outerrel, RelOptInfo *innerrel,
        JoinPathExtraData *extra)
{
  PgFdwRelationInfo *fpinfo;
  PgFdwRelationInfo *fpinfo_o;
  PgFdwRelationInfo *fpinfo_i;
  ListCell   *lc;
  List     *joinclauses;

  /*
   * We support pushing down INNER, LEFT, RIGHT and FULL OUTER joins.
   * Constructing queries representing SEMI and ANTI joins is hard, hence
   * not considered right now.
   */
  if (jointype != JOIN_INNER && jointype != JOIN_LEFT &&
    jointype != JOIN_RIGHT && jointype != JOIN_FULL)
    return false;

  /*
   * If either of the joining relations is marked as unsafe to pushdown, the
   * join can not be pushed down.
   */
  fpinfo = (PgFdwRelationInfo *) joinrel->fdw_private;
  fpinfo_o = (PgFdwRelationInfo *) outerrel->fdw_private;
  fpinfo_i = (PgFdwRelationInfo *) innerrel->fdw_private;
  if (!fpinfo_o || !fpinfo_o->pushdown_safe ||
    !fpinfo_i || !fpinfo_i->pushdown_safe)
    return false;

  /*
   * If joining relations have local conditions, those conditions are
   * required to be applied before joining the relations. Hence the join can
   * not be pushed down.
   */
  if (fpinfo_o->local_conds || fpinfo_i->local_conds)
    return false;

  /*
   * Merge FDW options.  We might be tempted to do this after we have deemed
   * the foreign join to be OK.  But we must do this beforehand so that we
   * know which quals can be evaluated on the foreign server, which might
   * depend on shippable_extensions.
   */
  fpinfo->server = fpinfo_o->server;
  merge_fdw_options(fpinfo, fpinfo_o, fpinfo_i);

  /*
   * Separate restrict list into join quals and pushed-down (other) quals.
   *
   * Join quals belonging to an outer join must all be shippable, else we
   * cannot execute the join remotely.  Add such quals to 'joinclauses'.
   *
   * Add other quals to fpinfo->remote_conds if they are shippable, else to
   * fpinfo->local_conds.  In an inner join it's okay to execute conditions
   * either locally or remotely; the same is true for pushed-down conditions
   * at an outer join.
   *
   * Note we might return failure after having already scribbled on
   * fpinfo->remote_conds and fpinfo->local_conds.  That's okay because we
   * won't consult those lists again if we deem the join unshippable.
   */
  joinclauses = NIL;
  foreach(lc, extra->restrictlist)
  {
    RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
    bool    is_remote_clause = is_foreign_expr(root, joinrel,
                             rinfo->clause);

    if (IS_OUTER_JOIN(jointype) &&
      !RINFO_IS_PUSHED_DOWN(rinfo, joinrel->relids))
    {
      if (!is_remote_clause)
        return false;
      joinclauses = lappend(joinclauses, rinfo);
    }
    else
    {
      if (is_remote_clause)
        fpinfo->remote_conds = lappend(fpinfo->remote_conds, rinfo);
      else
        fpinfo->local_conds = lappend(fpinfo->local_conds, rinfo);
    }
  }

  /*
   * deparseExplicitTargetList() isn't smart enough to handle anything other
   * than a Var.  In particular, if there's some PlaceHolderVar that would
   * need to be evaluated within this join tree (because there's an upper
   * reference to a quantity that may go to NULL as a result of an outer
   * join), then we can't try to push the join down because we'll fail when
   * we get to deparseExplicitTargetList().  However, a PlaceHolderVar that
   * needs to be evaluated *at the top* of this join tree is OK, because we
   * can do that locally after fetching the results from the remote side.
   */
  foreach(lc, root->placeholder_list)
  {
    PlaceHolderInfo *phinfo = (PlaceHolderInfo*)lfirst(lc);
    Relids    relids;

    /* PlaceHolderInfo refers to parent relids, not child relids. */
    relids = IS_OTHER_REL(joinrel) ?
      joinrel->top_parent_relids : joinrel->relids;

    if (bms_is_subset(phinfo->ph_eval_at, relids) &&
      bms_nonempty_difference(relids, phinfo->ph_eval_at))
      return false;
  }

  /* Save the join clauses, for later use. */
  fpinfo->joinclauses = joinclauses;

  fpinfo->outerrel = outerrel;
  fpinfo->innerrel = innerrel;
  fpinfo->jointype = jointype;

  /*
   * By default, both the input relations are not required to be deparsed as
   * subqueries, but there might be some relations covered by the input
   * relations that are required to be deparsed as subqueries, so save the
   * relids of those relations for later use by the deparser.
   */
  fpinfo->make_outerrel_subquery = false;
  fpinfo->make_innerrel_subquery = false;
  Assert(bms_is_subset(fpinfo_o->lower_subquery_rels, outerrel->relids));
  Assert(bms_is_subset(fpinfo_i->lower_subquery_rels, innerrel->relids));
  fpinfo->lower_subquery_rels = bms_union(fpinfo_o->lower_subquery_rels,
                      fpinfo_i->lower_subquery_rels);

  /*
   * Pull the other remote conditions from the joining relations into join
   * clauses or other remote clauses (remote_conds) of this relation
   * wherever possible. This avoids building subqueries at every join step.
   *
   * For an inner join, clauses from both the relations are added to the
   * other remote clauses. For LEFT and RIGHT OUTER join, the clauses from
   * the outer side are added to remote_conds since those can be evaluated
   * after the join is evaluated. The clauses from inner side are added to
   * the joinclauses, since they need to be evaluated while constructing the
   * join.
   *
   * For a FULL OUTER JOIN, the other clauses from either relation can not
   * be added to the joinclauses or remote_conds, since each relation acts
   * as an outer relation for the other.
   *
   * The joining sides can not have local conditions, thus no need to test
   * shippability of the clauses being pulled up.
   */
  switch (jointype)
  {
    case JOIN_INNER:
      fpinfo->remote_conds = list_concat(fpinfo->remote_conds,
                         list_copy(fpinfo_i->remote_conds));
      fpinfo->remote_conds = list_concat(fpinfo->remote_conds,
                         list_copy(fpinfo_o->remote_conds));
      break;

    case JOIN_LEFT:
      fpinfo->joinclauses = list_concat(fpinfo->joinclauses,
                        list_copy(fpinfo_i->remote_conds));
      fpinfo->remote_conds = list_concat(fpinfo->remote_conds,
                         list_copy(fpinfo_o->remote_conds));
      break;

    case JOIN_RIGHT:
      fpinfo->joinclauses = list_concat(fpinfo->joinclauses,
                        list_copy(fpinfo_o->remote_conds));
      fpinfo->remote_conds = list_concat(fpinfo->remote_conds,
                         list_copy(fpinfo_i->remote_conds));
      break;

    case JOIN_FULL:

      /*
       * In this case, if any of the input relations has conditions, we
       * need to deparse that relation as a subquery so that the
       * conditions can be evaluated before the join.  Remember it in
       * the fpinfo of this relation so that the deparser can take
       * appropriate action.  Also, save the relids of base relations
       * covered by that relation for later use by the deparser.
       */
      if (fpinfo_o->remote_conds)
      {
        fpinfo->make_outerrel_subquery = true;
        fpinfo->lower_subquery_rels =
          bms_add_members(fpinfo->lower_subquery_rels,
                  outerrel->relids);
      }
      if (fpinfo_i->remote_conds)
      {
        fpinfo->make_innerrel_subquery = true;
        fpinfo->lower_subquery_rels =
          bms_add_members(fpinfo->lower_subquery_rels,
                  innerrel->relids);
      }
      break;

    default:
      /* Should not happen, we have just checked this above */
      elog(ERROR, "unsupported join type %d", jointype);
  }

  /*
   * For an inner join, all restrictions can be treated alike. Treating the
   * pushed down conditions as join conditions allows a top level full outer
   * join to be deparsed without requiring subqueries.
   */
  if (jointype == JOIN_INNER)
  {
    Assert(!fpinfo->joinclauses);
    fpinfo->joinclauses = fpinfo->remote_conds;
    fpinfo->remote_conds = NIL;
  }

  /* Mark that this join can be pushed down safely */
  fpinfo->pushdown_safe = true;

  /* Get user mapping */
  if (fpinfo->use_remote_estimate)
  {
    if (fpinfo_o->use_remote_estimate)
      fpinfo->user = fpinfo_o->user;
    else
      fpinfo->user = fpinfo_i->user;
  }
  else
    fpinfo->user = NULL;

  /*
   * Set # of retrieved rows and cached relation costs to some negative
   * value, so that we can detect when they are set to some sensible values,
   * during one (usually the first) of the calls to estimate_path_cost_size.
   */
  fpinfo->retrieved_rows = -1;
  fpinfo->rel_startup_cost = -1;
  fpinfo->rel_total_cost = -1;

  /*
   * Set the string describing this join relation to be used in EXPLAIN
   * output of corresponding ForeignScan.
   */
  fpinfo->relation_name = makeStringInfo();
  appendStringInfo(fpinfo->relation_name, "(%s) %s JOIN (%s)",
           fpinfo_o->relation_name->data,
           get_jointype_name(fpinfo->jointype),
           fpinfo_i->relation_name->data);

  /*
   * Set the relation index.  This is defined as the position of this
   * joinrel in the join_rel_list list plus the length of the rtable list.
   * Note that since this joinrel is at the end of the join_rel_list list
   * when we are called, we can get the position by list_length.
   */
  Assert(fpinfo->relation_index == 0);  /* shouldn't be set yet */
  fpinfo->relation_index =
    list_length(root->parse->rtable) + list_length(root->join_rel_list);

  return true;
}


/*
 * postgresGetForeignJoinPaths
 *    Add possible ForeignPath to joinrel, if join is safe to push down.
 */
extern "C" void
parquetGetForeignJoinPaths(PlannerInfo *root,
              RelOptInfo *joinrel,
              RelOptInfo *outerrel,
              RelOptInfo *innerrel,
              JoinType jointype,
              JoinPathExtraData *extra)
{
  PgFdwRelationInfo *fpinfo;
  ForeignPath *joinpath;
  double    rows;
  int     width;
  Cost    startup_cost;
  Cost    total_cost;
  Path     *epq_path;   /* Path to create plan to be executed when
                 * EvalPlanQual gets triggered. */

  /*
   * Skip if this join combination has been considered already.
   */
  if (joinrel->fdw_private)
    return;

  /*
   * This code does not work for joins with lateral references, since those
   * must have parameterized paths, which we don't generate yet.
   */
  if (!bms_is_empty(joinrel->lateral_relids))
    return;

  /*
   * Create unfinished PgFdwRelationInfo entry which is used to indicate
   * that the join relation is already considered, so that we won't waste
   * time in judging safety of join pushdown and adding the same paths again
   * if found safe. Once we know that this join can be pushed down, we fill
   * the entry.
   */
  fpinfo = (PgFdwRelationInfo *) palloc0(sizeof(PgFdwRelationInfo));
  fpinfo->pushdown_safe = false;
  joinrel->fdw_private = fpinfo;
  /* attrs_used is only for base relations. */
  fpinfo->attrs_used = NULL;

  /*
   * If there is a possibility that EvalPlanQual will be executed, we need
   * to be able to reconstruct the row using scans of the base relations.
   * GetExistingLocalJoinPath will find a suitable path for this purpose in
   * the path list of the joinrel, if one exists.  We must be careful to
   * call it before adding any ForeignPath, since the ForeignPath might
   * dominate the only suitable local path available.  We also do it before
   * calling foreign_join_ok(), since that function updates fpinfo and marks
   * it as pushable if the join is found to be pushable.
   */
  if (root->parse->commandType == CMD_DELETE ||
    root->parse->commandType == CMD_UPDATE ||
    root->rowMarks)
  {
    epq_path = GetExistingLocalJoinPath(joinrel);
    if (!epq_path)
    {
      elog(DEBUG3, "could not push down foreign join because a local path suitable for EPQ checks was not found");
      return;
    }
  }
  else
    epq_path = NULL;

  if (!foreign_join_ok(root, joinrel, jointype, outerrel, innerrel, extra))
  {
    /* Free path required for EPQ if we copied one; we don't need it now */
    if (epq_path)
      pfree(epq_path);
    return;
  }

  /*
   * Compute the selectivity and cost of the local_conds, so we don't have
   * to do it over again for each path. The best we can do for these
   * conditions is to estimate selectivity on the basis of local statistics.
   * The local conditions are applied after the join has been computed on
   * the remote side like quals in WHERE clause, so pass jointype as
   * JOIN_INNER.
   */
  fpinfo->local_conds_sel = clauselist_selectivity(root,
                           fpinfo->local_conds,
                           0,
                           JOIN_INNER,
                           NULL);
  cost_qual_eval(&fpinfo->local_conds_cost, fpinfo->local_conds, root);

  /*
   * If we are going to estimate costs locally, estimate the join clause
   * selectivity here while we have special join info.
   */
  if (!fpinfo->use_remote_estimate)
    fpinfo->joinclause_sel = clauselist_selectivity(root, fpinfo->joinclauses,
                            0, fpinfo->jointype,
                            extra->sjinfo);

  /* Estimate costs for bare join relation */
  estimate_path_cost_size(root, joinrel, NIL, NIL, NULL,
              &rows, &width, &startup_cost, &total_cost);
  /* Now update this information in the joinrel */
  joinrel->rows = rows;
  joinrel->reltarget->width = width;
  fpinfo->rows = rows;
  fpinfo->width = width;
  fpinfo->startup_cost = startup_cost;
  fpinfo->total_cost = total_cost;

  /*
   * Create a new join path and add it to the joinrel which represents a
   * join between foreign tables.
   */
  joinpath = create_foreign_join_path(root,
                    joinrel,
                    NULL, /* default pathtarget */
                    rows,
                    startup_cost,
                    total_cost,
                    NIL,  /* no pathkeys */
                    joinrel->lateral_relids,
                    epq_path,
                    NIL); /* no fdw_private */

  /* Add generated path into joinrel by add_path(). */
  add_path(joinrel, (Path *) joinpath);

  /* Consider pathkeys for the join relation */
  add_paths_with_pathkeys_for_rel(root, joinrel, epq_path);

  /* XXX Consider parameterized paths for the join relation */
}

extern "C" void
parquetGetForeignUpperPaths(PlannerInfo *root, UpperRelationKind stage,
               RelOptInfo *input_rel, RelOptInfo *output_rel,
               void *extra)
{
    PgFdwRelationInfo *fpinfo;

    /*
     * If input rel is not safe to pushdown, then simply return as we cannot
     * perform any post-join operations on the foreign server.
     */
    if (!input_rel->fdw_private ||
      !((PgFdwRelationInfo *) input_rel->fdw_private)->pushdown_safe)
      return;

    /* Ignore stages we don't support; and skip any duplicate calls. */
    if ((stage != UPPERREL_GROUP_AGG &&
       stage != UPPERREL_ORDERED &&
       stage != UPPERREL_FINAL) ||
      output_rel->fdw_private)
      return;

    fpinfo = (PgFdwRelationInfo *) palloc0(sizeof(PgFdwRelationInfo));
    fpinfo->pushdown_safe = false;
    fpinfo->stage = stage;
    output_rel->fdw_private = fpinfo;

    switch (stage)
    {
      case UPPERREL_GROUP_AGG:
        add_foreign_grouping_paths(root, input_rel, output_rel,
                      (GroupPathExtraData *) extra);
        break;
      case UPPERREL_ORDERED:
        add_foreign_ordered_paths(root, input_rel, output_rel);
        break;
      case UPPERREL_FINAL:
        add_foreign_final_paths(root, input_rel, output_rel,
                    (FinalPathExtraData *) extra);
        break;
      default:
        elog(ERROR, "unexpected upper relation: %d", (int) stage);
        break;
    }
}


/*
 * Assess whether the aggregation, grouping and having operations can be pushed
 * down to the foreign server.  As a side effect, save information we obtain in
 * this function to ParquetFdwPlanState of the input relation.
 */
static bool
foreign_grouping_ok(PlannerInfo *root, RelOptInfo *grouped_rel,
          Node *havingQual)
{
  Query    *query = root->parse;
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) grouped_rel->fdw_private;
  PathTarget *grouping_target = grouped_rel->reltarget;
  PgFdwRelationInfo *ofpinfo;
  ListCell   *lc;
  int     i;
  List     *tlist = NIL;

  /* We currently don't support pushing Grouping Sets. */
  if (query->groupingSets)
    return false;

  /* Get the fpinfo of the underlying scan relation. */
  ofpinfo = (PgFdwRelationInfo *) fpinfo->outerrel->fdw_private;

  /*
   * If underlying scan relation has any local conditions, those conditions
   * are required to be applied before performing aggregation.  Hence the
   * aggregate cannot be pushed down.
   */
  if (ofpinfo->local_conds)
    return false;

  /*
   * Examine grouping expressions, as well as other expressions we'd need to
   * compute, and check whether they are safe to push down to the foreign
   * server.  All GROUP BY expressions will be part of the grouping target
   * and thus there is no need to search for them separately.  Add grouping
   * expressions into target list which will be passed to foreign server.
   *
   * A tricky fine point is that we must not put any expression into the
   * target list that is just a foreign param (that is, something that
   * deparse.c would conclude has to be sent to the foreign server).  If we
   * do, the expression will also appear in the fdw_exprs list of the plan
   * node, and setrefs.c will get confused and decide that the fdw_exprs
   * entry is actually a reference to the fdw_scan_tlist entry, resulting in
   * a broken plan.  Somewhat oddly, it's OK if the expression contains such
   * a node, as long as it's not at top level; then no match is possible.
   */
  i = 0;
  foreach(lc, grouping_target->exprs)
  {
    Expr     *expr = (Expr *) lfirst(lc);
    Index   sgref = get_pathtarget_sortgroupref(grouping_target, i);
    ListCell   *l;

    /* Check whether this expression is part of GROUP BY clause */
    if (sgref && get_sortgroupref_clause_noerr(sgref, query->groupClause))
    {
      TargetEntry *tle;

      /*
       * If any GROUP BY expression is not shippable, then we cannot
       * push down aggregation to the foreign server.
       */
      if (!is_foreign_expr(root, grouped_rel, expr))
        return false;

      /*
       * If it would be a foreign param, we can't put it into the tlist,
       * so we have to fail.
       */
      if (is_foreign_param(root, grouped_rel, expr))
        return false;

      /*
       * Pushable, so add to tlist.  We need to create a TLE for this
       * expression and apply the sortgroupref to it.  We cannot use
       * add_to_flat_tlist() here because that avoids making duplicate
       * entries in the tlist.  If there are duplicate entries with
       * distinct sortgrouprefs, we have to duplicate that situation in
       * the output tlist.
       */
      tle = makeTargetEntry(expr, list_length(tlist) + 1, NULL, false);
      tle->ressortgroupref = sgref;
      tlist = lappend(tlist, tle);
    }
    else
    {
      /*
       * Non-grouping expression we need to compute.  Can we ship it
       * as-is to the foreign server?
       */
      if (is_foreign_expr(root, grouped_rel, expr) &&
        !is_foreign_param(root, grouped_rel, expr))
      {
        /* Yes, so add to tlist as-is; OK to suppress duplicates */
        tlist = add_to_flat_tlist(tlist, list_make1(expr));
      }
      else
      {
        /* Not pushable as a whole; extract its Vars and aggregates */
        List     *aggvars;

        aggvars = pull_var_clause((Node *) expr,
                      PVC_INCLUDE_AGGREGATES);

        /*
         * If any aggregate expression is not shippable, then we
         * cannot push down aggregation to the foreign server.  (We
         * don't have to check is_foreign_param, since that certainly
         * won't return true for any such expression.)
         */
        if (!is_foreign_expr(root, grouped_rel, (Expr *) aggvars))
          return false;

        /*
         * Add aggregates, if any, into the targetlist.  Plain Vars
         * outside an aggregate can be ignored, because they should be
         * either same as some GROUP BY column or part of some GROUP
         * BY expression.  In either case, they are already part of
         * the targetlist and thus no need to add them again.  In fact
         * including plain Vars in the tlist when they do not match a
         * GROUP BY column would cause the foreign server to complain
         * that the shipped query is invalid.
         */
        foreach(l, aggvars)
        {
          Expr     *expr = (Expr *) lfirst(l);

          if (IsA(expr, Aggref))
            tlist = add_to_flat_tlist(tlist, list_make1(expr));
        }
      }
    }

    i++;
  }

  /*
   * Classify the pushable and non-pushable HAVING clauses and save them in
   * remote_conds and local_conds of the grouped rel's fpinfo.
   */
  if (havingQual)
  {
    ListCell   *lc;

    foreach(lc, (List *) havingQual)
    {
      Expr     *expr = (Expr *) lfirst(lc);
      RestrictInfo *rinfo;

      /*
       * Currently, the core code doesn't wrap havingQuals in
       * RestrictInfos, so we must make our own.
       */
      Assert(!IsA(expr, RestrictInfo));
      rinfo = make_restrictinfo(expr,
                    true,
                    false,
                    false,
                    root->qual_security_level,
                    grouped_rel->relids,
                    NULL,
                    NULL);
      if (is_foreign_expr(root, grouped_rel, expr))
        fpinfo->remote_conds = lappend(fpinfo->remote_conds, rinfo);
      else
        fpinfo->local_conds = lappend(fpinfo->local_conds, rinfo);
    }
  }

  /*
   * If there are any local conditions, pull Vars and aggregates from it and
   * check whether they are safe to pushdown or not.
   */
  if (fpinfo->local_conds)
  {
    List     *aggvars = NIL;
    ListCell   *lc;

    foreach(lc, fpinfo->local_conds)
    {
      RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);

      aggvars = list_concat(aggvars,
                  pull_var_clause((Node *) rinfo->clause,
                          PVC_INCLUDE_AGGREGATES));
    }

    foreach(lc, aggvars)
    {
      Expr     *expr = (Expr *) lfirst(lc);

      /*
       * If aggregates within local conditions are not safe to push
       * down, then we cannot push down the query.  Vars are already
       * part of GROUP BY clause which are checked above, so no need to
       * access them again here.  Again, we need not check
       * is_foreign_param for a foreign aggregate.
       */
      if (IsA(expr, Aggref))
      {
        if (!is_foreign_expr(root, grouped_rel, expr))
          return false;

        tlist = add_to_flat_tlist(tlist, list_make1(expr));
      }
    }
  }

  /* Store generated targetlist */
  fpinfo->grouped_tlist = tlist;

  /* Safe to pushdown */
  fpinfo->pushdown_safe = true;

  /*
   * Set # of retrieved rows and cached relation costs to some negative
   * value, so that we can detect when they are set to some sensible values,
   * during one (usually the first) of the calls to estimate_path_cost_size.
   */
  fpinfo->retrieved_rows = -1;
  fpinfo->rel_startup_cost = -1;
  fpinfo->rel_total_cost = -1;

  /*
   * Set the string describing this grouped relation to be used in EXPLAIN
   * output of corresponding ForeignScan.
   */
  fpinfo->relation_name = makeStringInfo();
  appendStringInfo(fpinfo->relation_name, "Aggregate on (%s)",
           ofpinfo->relation_name->data);

  return true;
}

/*
 * add_foreign_grouping_paths
 *    Add foreign path for grouping and/or aggregation.
 *
 * Given input_rel represents the underlying scan.  The paths are added to the
 * given grouped_rel.
 */
static void
add_foreign_grouping_paths(PlannerInfo *root, RelOptInfo *input_rel,
               RelOptInfo *grouped_rel,
               GroupPathExtraData *extra)
{
  Query    *parse = root->parse;
  PgFdwRelationInfo *ifpinfo = (PgFdwRelationInfo *)input_rel->fdw_private;
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *)grouped_rel->fdw_private;
  ForeignPath *grouppath;
  double    rows;
  int     width;
  Cost    startup_cost;
  Cost    total_cost;

  /* Nothing to be done, if there is no grouping or aggregation required. */
  if (!parse->groupClause && !parse->groupingSets && !parse->hasAggs &&
    !root->hasHavingQual)
    return;

  Assert(extra->patype == PARTITIONWISE_AGGREGATE_NONE ||
       extra->patype == PARTITIONWISE_AGGREGATE_FULL);

  /* save the input_rel as outerrel in fpinfo */
  fpinfo->outerrel = input_rel;

  /*
   * Copy foreign table, foreign server, user mapping, FDW options etc.
   * details from the input relation's fpinfo.
   */
  fpinfo->table = ifpinfo->table;
  fpinfo->server = ifpinfo->server;
  fpinfo->user = ifpinfo->user;
  merge_fdw_options(fpinfo, ifpinfo, NULL);

  /*
   * Assess if it is safe to push down aggregation and grouping.
   *
   * Use HAVING qual from extra. In case of child partition, it will have
   * translated Vars.
   */
  if (!foreign_grouping_ok(root, grouped_rel, extra->havingQual))
    return;

  /*
   * Compute the selectivity and cost of the local_conds, so we don't have
   * to do it over again for each path.  (Currently we create just a single
   * path here, but in future it would be possible that we build more paths
   * such as pre-sorted paths as in postgresGetForeignPaths and
   * postgresGetForeignJoinPaths.)  The best we can do for these conditions
   * is to estimate selectivity on the basis of local statistics.
   */
  fpinfo->local_conds_sel = clauselist_selectivity(root,
                           fpinfo->local_conds,
                           0,
                           JOIN_INNER,
                           NULL);

  cost_qual_eval(&fpinfo->local_conds_cost, fpinfo->local_conds, root);

  /* Estimate the cost of push down */
  estimate_path_cost_size(root, grouped_rel, NIL, NIL, NULL,
              &rows, &width, &startup_cost, &total_cost);

  /* Now update this information in the fpinfo */
  fpinfo->rows = rows;
  fpinfo->width = width;
  fpinfo->startup_cost = startup_cost;
  fpinfo->total_cost = total_cost;

  /* Create and add foreign path to the grouping relation. */
  grouppath = create_foreign_upper_path(root,
                      grouped_rel,
                      grouped_rel->reltarget,
                      rows,
                      startup_cost,
                      total_cost,
                      NIL,  /* no pathkeys */
                      NULL,
                      NIL); /* no fdw_private */

  /* Add generated path into grouped_rel by add_path(). */
  add_path(grouped_rel, (Path *) grouppath);
}

/*
 * add_foreign_ordered_paths
 *    Add foreign paths for performing the final sort remotely.
 *
 * Given input_rel contains the source-data Paths.  The paths are added to the
 * given ordered_rel.
 */
static void
add_foreign_ordered_paths(PlannerInfo *root, RelOptInfo *input_rel,
              RelOptInfo *ordered_rel)
{
  Query    *parse = root->parse;
  PgFdwRelationInfo *ifpinfo = (PgFdwRelationInfo *)input_rel->fdw_private;
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *)ordered_rel->fdw_private;
  PgFdwPathExtraData *fpextra;
  double    rows;
  int     width;
  Cost    startup_cost;
  Cost    total_cost;
  List     *fdw_private;
  ForeignPath *ordered_path;
  ListCell   *lc;

  /* Shouldn't get here unless the query has ORDER BY */
  Assert(parse->sortClause);

  /* We don't support cases where there are any SRFs in the targetlist */
  if (parse->hasTargetSRFs)
    return;

  /* Save the input_rel as outerrel in fpinfo */
  fpinfo->outerrel = input_rel;

  /*
   * Copy foreign table, foreign server, user mapping, FDW options etc.
   * details from the input relation's fpinfo.
   */
  fpinfo->table = ifpinfo->table;
  fpinfo->server = ifpinfo->server;
  fpinfo->user = ifpinfo->user;
  merge_fdw_options(fpinfo, ifpinfo, NULL);

  /*
   * If the input_rel is a base or join relation, we would already have
   * considered pushing down the final sort to the remote server when
   * creating pre-sorted foreign paths for that relation, because the
   * query_pathkeys is set to the root->sort_pathkeys in that case (see
   * standard_qp_callback()).
   */
  if (input_rel->reloptkind == RELOPT_BASEREL ||
    input_rel->reloptkind == RELOPT_JOINREL)
  {
    Assert(root->query_pathkeys == root->sort_pathkeys);

    /* Safe to push down if the query_pathkeys is safe to push down */
    fpinfo->pushdown_safe = ifpinfo->qp_is_pushdown_safe;

    return;
  }

  /* The input_rel should be a grouping relation */
  Assert(input_rel->reloptkind == RELOPT_UPPER_REL &&
       ifpinfo->stage == UPPERREL_GROUP_AGG);

  /*
   * We try to create a path below by extending a simple foreign path for
   * the underlying grouping relation to perform the final sort remotely,
   * which is stored into the fdw_private list of the resulting path.
   */

  /* Assess if it is safe to push down the final sort */
  foreach(lc, root->sort_pathkeys)
  {
    PathKey    *pathkey = (PathKey *) lfirst(lc);
    EquivalenceClass *pathkey_ec = pathkey->pk_eclass;
    Expr     *sort_expr;

    /*
     * is_foreign_expr would detect volatile expressions as well, but
     * checking ec_has_volatile here saves some cycles.
     */
    if (pathkey_ec->ec_has_volatile)
      return;

    /* Get the sort expression for the pathkey_ec */
    sort_expr = find_em_expr_for_input_target(root,
                          pathkey_ec,
                          input_rel->reltarget);

    /* If it's unsafe to remote, we cannot push down the final sort */
    if (!is_foreign_expr(root, input_rel, sort_expr))
      return;
  }

  /* Safe to push down */
  fpinfo->pushdown_safe = true;

  /* Construct PgFdwPathExtraData */
  fpextra = (PgFdwPathExtraData *) palloc0(sizeof(PgFdwPathExtraData));
  fpextra->target = root->upper_targets[UPPERREL_ORDERED];
  fpextra->has_final_sort = true;

  /* Estimate the costs of performing the final sort remotely */
  estimate_path_cost_size(root, input_rel, NIL, root->sort_pathkeys, fpextra,
              &rows, &width, &startup_cost, &total_cost);

  /*
   * Build the fdw_private list that will be used by postgresGetForeignPlan.
   * Items in the list must match order in enum FdwPathPrivateIndex.
   */
  fdw_private = list_make2(makeInteger(true), makeInteger(false));

  /* Create foreign ordering path */
  ordered_path = create_foreign_upper_path(root,
                       input_rel,
                       root->upper_targets[UPPERREL_ORDERED],
                       rows,
                       startup_cost,
                       total_cost,
                       root->sort_pathkeys,
                       NULL,  /* no extra plan */
                       fdw_private);

  /* and add it to the ordered_rel */
  add_path(ordered_rel, (Path *) ordered_path);
}

/*
 * add_foreign_final_paths
 *    Add foreign paths for performing the final processing remotely.
 *
 * Given input_rel contains the source-data Paths.  The paths are added to the
 * given final_rel.
 */
static void
add_foreign_final_paths(PlannerInfo *root, RelOptInfo *input_rel,
            RelOptInfo *final_rel,
            FinalPathExtraData *extra)
{
  Query    *parse = root->parse;
  PgFdwRelationInfo *ifpinfo = (PgFdwRelationInfo *) input_rel->fdw_private;
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) final_rel->fdw_private;
  bool    has_final_sort = false;
  List     *pathkeys = NIL;
  PgFdwPathExtraData *fpextra;
  bool    save_use_remote_estimate = false;
  double    rows;
  int     width;
  Cost    startup_cost;
  Cost    total_cost;
  List     *fdw_private;
  ForeignPath *final_path;

  /*
   * Currently, we only support this for SELECT commands
   */
  if (parse->commandType != CMD_SELECT)
    return;

  /*
   * No work if there is no FOR UPDATE/SHARE clause and if there is no need
   * to add a LIMIT node
   */
  if (!parse->rowMarks && !extra->limit_needed)
    return;

  /* We don't support cases where there are any SRFs in the targetlist */
  if (parse->hasTargetSRFs)
    return;

  /* Save the input_rel as outerrel in fpinfo */
  fpinfo->outerrel = input_rel;

  /*
   * Copy foreign table, foreign server, user mapping, FDW options etc.
   * details from the input relation's fpinfo.
   */
  fpinfo->table = ifpinfo->table;
  fpinfo->server = ifpinfo->server;
  fpinfo->user = ifpinfo->user;
  merge_fdw_options(fpinfo, ifpinfo, NULL);

  /*
   * If there is no need to add a LIMIT node, there might be a ForeignPath
   * in the input_rel's pathlist that implements all behavior of the query.
   * Note: we would already have accounted for the query's FOR UPDATE/SHARE
   * (if any) before we get here.
   */
  if (!extra->limit_needed)
  {
    ListCell   *lc;

    Assert(parse->rowMarks);

    /*
     * Grouping and aggregation are not supported with FOR UPDATE/SHARE,
     * so the input_rel should be a base, join, or ordered relation; and
     * if it's an ordered relation, its input relation should be a base or
     * join relation.
     */
    Assert(input_rel->reloptkind == RELOPT_BASEREL ||
         input_rel->reloptkind == RELOPT_JOINREL ||
         (input_rel->reloptkind == RELOPT_UPPER_REL &&
        ifpinfo->stage == UPPERREL_ORDERED &&
        (ifpinfo->outerrel->reloptkind == RELOPT_BASEREL ||
         ifpinfo->outerrel->reloptkind == RELOPT_JOINREL)));

    foreach(lc, input_rel->pathlist)
    {
      Path     *path = (Path *) lfirst(lc);

      /*
       * apply_scanjoin_target_to_paths() uses create_projection_path()
       * to adjust each of its input paths if needed, whereas
       * create_ordered_paths() uses apply_projection_to_path() to do
       * that.  So the former might have put a ProjectionPath on top of
       * the ForeignPath; look through ProjectionPath and see if the
       * path underneath it is ForeignPath.
       */
      if (IsA(path, ForeignPath) ||
        (IsA(path, ProjectionPath) &&
         IsA(((ProjectionPath *) path)->subpath, ForeignPath)))
      {
        /*
         * Create foreign final path; this gets rid of a
         * no-longer-needed outer plan (if any), which makes the
         * EXPLAIN output look cleaner
         */
        final_path = create_foreign_upper_path(root,
                             path->parent,
                             path->pathtarget,
                             path->rows,
                             path->startup_cost,
                             path->total_cost,
                             path->pathkeys,
                             NULL,  /* no extra plan */
                             NULL); /* no fdw_private */

        /* and add it to the final_rel */
        add_path(final_rel, (Path *) final_path);

        /* Safe to push down */
        fpinfo->pushdown_safe = true;

        return;
      }
    }

    /*
     * If we get here it means no ForeignPaths; since we would already
     * have considered pushing down all operations for the query to the
     * remote server, give up on it.
     */
    return;
  }

  Assert(extra->limit_needed);

  /*
   * If the input_rel is an ordered relation, replace the input_rel with its
   * input relation
   */
  if (input_rel->reloptkind == RELOPT_UPPER_REL &&
    ifpinfo->stage == UPPERREL_ORDERED)
  {
    input_rel = ifpinfo->outerrel;
    ifpinfo = (PgFdwRelationInfo *) input_rel->fdw_private;
    has_final_sort = true;
    pathkeys = root->sort_pathkeys;
  }

  /* The input_rel should be a base, join, or grouping relation */
  Assert(input_rel->reloptkind == RELOPT_BASEREL ||
       input_rel->reloptkind == RELOPT_JOINREL ||
       (input_rel->reloptkind == RELOPT_UPPER_REL &&
      ifpinfo->stage == UPPERREL_GROUP_AGG));

  /*
   * We try to create a path below by extending a simple foreign path for
   * the underlying base, join, or grouping relation to perform the final
   * sort (if has_final_sort) and the LIMIT restriction remotely, which is
   * stored into the fdw_private list of the resulting path.  (We
   * re-estimate the costs of sorting the underlying relation, if
   * has_final_sort.)
   */

  /*
   * Assess if it is safe to push down the LIMIT and OFFSET to the remote
   * server
   */

  /*
   * If the underlying relation has any local conditions, the LIMIT/OFFSET
   * cannot be pushed down.
   */
  if (ifpinfo->local_conds)
    return;

  /*
   * Also, the LIMIT/OFFSET cannot be pushed down, if their expressions are
   * not safe to remote.
   */
  if (!is_foreign_expr(root, input_rel, (Expr *) parse->limitOffset) ||
    !is_foreign_expr(root, input_rel, (Expr *) parse->limitCount))
    return;

  /* Safe to push down */
  fpinfo->pushdown_safe = true;

  /* Construct PgFdwPathExtraData */
  fpextra = (PgFdwPathExtraData *) palloc0(sizeof(PgFdwPathExtraData));
  fpextra->target = root->upper_targets[UPPERREL_FINAL];
  fpextra->has_final_sort = has_final_sort;
  fpextra->has_limit = extra->limit_needed;
  fpextra->limit_tuples = extra->limit_tuples;
  fpextra->count_est = extra->count_est;
  fpextra->offset_est = extra->offset_est;

  /*
   * Estimate the costs of performing the final sort and the LIMIT
   * restriction remotely.  If has_final_sort is false, we wouldn't need to
   * execute EXPLAIN anymore if use_remote_estimate, since the costs can be
   * roughly estimated using the costs we already have for the underlying
   * relation, in the same way as when use_remote_estimate is false.  Since
   * it's pretty expensive to execute EXPLAIN, force use_remote_estimate to
   * false in that case.
   */
  if (!fpextra->has_final_sort)
  {
    save_use_remote_estimate = ifpinfo->use_remote_estimate;
    ifpinfo->use_remote_estimate = false;
  }
  estimate_path_cost_size(root, input_rel, NIL, pathkeys, fpextra,
              &rows, &width, &startup_cost, &total_cost);
  if (!fpextra->has_final_sort)
    ifpinfo->use_remote_estimate = save_use_remote_estimate;

  /*
   * Build the fdw_private list that will be used by postgresGetForeignPlan.
   * Items in the list must match order in enum FdwPathPrivateIndex.
   */
  fdw_private = list_make2(makeInteger(has_final_sort),
               makeInteger(extra->limit_needed));

  /*
   * Create foreign final path; this gets rid of a no-longer-needed outer
   * plan (if any), which makes the EXPLAIN output look cleaner
   */
  final_path = create_foreign_upper_path(root,
                       input_rel,
                       root->upper_targets[UPPERREL_FINAL],
                       rows,
                       startup_cost,
                       total_cost,
                       pathkeys,
                       NULL,  /* no extra plan */
                       fdw_private);

  /* and add it to the final_rel */
  add_path(final_rel, (Path *) final_path);
}


/*
 * estimate_path_cost_size
 *    Get cost and size estimates for a foreign scan on given foreign relation
 *    either a base relation or a join between foreign relations or an upper
 *    relation containing foreign relations.
 *
 * param_join_conds are the parameterization clauses with outer relations.
 * pathkeys specify the expected sort order if any for given path being costed.
 * fpextra specifies additional post-scan/join-processing steps such as the
 * final sort and the LIMIT restriction.
 *
 * The function returns the cost and size estimates in p_rows, p_width,
 * p_startup_cost and p_total_cost variables.
 */
static void
estimate_path_cost_size(PlannerInfo *root,
            RelOptInfo *foreignrel,
            List *param_join_conds,
            List *pathkeys,
            PgFdwPathExtraData *fpextra,
            double *p_rows, int *p_width,
            Cost *p_startup_cost, Cost *p_total_cost)
{
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) foreignrel->fdw_private;
  double    rows;
  double    retrieved_rows;
  int     width;
  Cost    startup_cost;
  Cost    total_cost;

  /* Make sure the core code has set up the relation's reltarget */
  Assert(foreignrel->reltarget);

  /*
   * If the table or the server is configured to use remote estimates,
   * connect to the foreign server and execute EXPLAIN to estimate the
   * number of rows selected by the restriction+join clauses.  Otherwise,
   * estimate rows using whatever statistics we have locally, in a way
   * similar to ordinary tables.
   */
  if (fpinfo->use_remote_estimate)
  {
    List     *remote_param_join_conds;
    List     *local_param_join_conds;
    StringInfoData sql;
    PGconn     *conn;
    Selectivity local_sel;
    QualCost  local_cost;
    List     *fdw_scan_tlist = NIL;
    List     *remote_conds;

    /* Required only to be passed to deparseSelectStmtForRel */
    List     *retrieved_attrs;

    /*
     * param_join_conds might contain both clauses that are safe to send
     * across, and clauses that aren't.
     */
    classifyConditions(root, foreignrel, param_join_conds,
               &remote_param_join_conds, &local_param_join_conds);

    /* Build the list of columns to be fetched from the foreign server. */
    if (IS_JOIN_REL(foreignrel) || IS_UPPER_REL(foreignrel))
      fdw_scan_tlist = build_tlist_to_deparse(foreignrel);
    else
      fdw_scan_tlist = NIL;

    /*
     * The complete list of remote conditions includes everything from
     * baserestrictinfo plus any extra join_conds relevant to this
     * particular path.
     */
    remote_conds = list_concat(list_copy(remote_param_join_conds),
                   fpinfo->remote_conds);

    /*
     * Construct EXPLAIN query including the desired SELECT, FROM, and
     * WHERE clauses. Params and other-relation Vars are replaced by dummy
     * values, so don't request params_list.
     */
    initStringInfo(&sql);
    appendStringInfoString(&sql, "EXPLAIN ");
    deparseSelectStmtForRel(&sql, root, foreignrel, fdw_scan_tlist,
                remote_conds, pathkeys,
                fpextra ? fpextra->has_final_sort : false,
                fpextra ? fpextra->has_limit : false,
                false, &retrieved_attrs, NULL);

    /* Get the remote estimate */
    // conn = GetConnection(fpinfo->user, false);
    // get_remote_estimate(sql.data, conn, &rows, &width,
    //           &startup_cost, &total_cost);
    // ReleaseConnection(conn);

    retrieved_rows = rows;

    /* Factor in the selectivity of the locally-checked quals */
    local_sel = clauselist_selectivity(root,
                       local_param_join_conds,
                       foreignrel->relid,
                       JOIN_INNER,
                       NULL);
    local_sel *= fpinfo->local_conds_sel;

    rows = clamp_row_est(rows * local_sel);

    /* Add in the eval cost of the locally-checked quals */
    startup_cost += fpinfo->local_conds_cost.startup;
    total_cost += fpinfo->local_conds_cost.per_tuple * retrieved_rows;
    cost_qual_eval(&local_cost, local_param_join_conds, root);
    startup_cost += local_cost.startup;
    total_cost += local_cost.per_tuple * retrieved_rows;

    /*
     * Add in tlist eval cost for each output row.  In case of an
     * aggregate, some of the tlist expressions such as grouping
     * expressions will be evaluated remotely, so adjust the costs.
     */
    startup_cost += foreignrel->reltarget->cost.startup;
    total_cost += foreignrel->reltarget->cost.startup;
    total_cost += foreignrel->reltarget->cost.per_tuple * rows;
    if (IS_UPPER_REL(foreignrel))
    {
      QualCost  tlist_cost;

      cost_qual_eval(&tlist_cost, fdw_scan_tlist, root);
      startup_cost -= tlist_cost.startup;
      total_cost -= tlist_cost.startup;
      total_cost -= tlist_cost.per_tuple * rows;
    }
  }
  else
  {
    Cost    run_cost = 0;

    /*
     * We don't support join conditions in this mode (hence, no
     * parameterized paths can be made).
     */
    Assert(param_join_conds == NIL);

    /*
     * We will come here again and again with different set of pathkeys or
     * additional post-scan/join-processing steps that caller wants to
     * cost.  We don't need to calculate the cost/size estimates for the
     * underlying scan, join, or grouping each time.  Instead, use those
     * estimates if we have cached them already.
     */
    if (fpinfo->rel_startup_cost >= 0 && fpinfo->rel_total_cost >= 0)
    {
      Assert(fpinfo->retrieved_rows >= 1);

      rows = fpinfo->rows;
      retrieved_rows = fpinfo->retrieved_rows;
      width = fpinfo->width;
      startup_cost = fpinfo->rel_startup_cost;
      run_cost = fpinfo->rel_total_cost - fpinfo->rel_startup_cost;

      /*
       * If we estimate the costs of a foreign scan or a foreign join
       * with additional post-scan/join-processing steps, the scan or
       * join costs obtained from the cache wouldn't yet contain the
       * eval costs for the final scan/join target, which would've been
       * updated by apply_scanjoin_target_to_paths(); add the eval costs
       * now.
       */
      if (fpextra && !IS_UPPER_REL(foreignrel))
      {
        /* Shouldn't get here unless we have LIMIT */
        Assert(fpextra->has_limit);
        Assert(foreignrel->reloptkind == RELOPT_BASEREL ||
             foreignrel->reloptkind == RELOPT_JOINREL);
        startup_cost += foreignrel->reltarget->cost.startup;
        run_cost += foreignrel->reltarget->cost.per_tuple * rows;
      }
    }
    else if (IS_JOIN_REL(foreignrel))
    {
      PgFdwRelationInfo *fpinfo_i;
      PgFdwRelationInfo *fpinfo_o;
      QualCost  join_cost;
      QualCost  remote_conds_cost;
      double    nrows;

      /* Use rows/width estimates made by the core code. */
      rows = foreignrel->rows;
      width = foreignrel->reltarget->width;

      /* For join we expect inner and outer relations set */
      Assert(fpinfo->innerrel && fpinfo->outerrel);

      fpinfo_i = (PgFdwRelationInfo *) fpinfo->innerrel->fdw_private;
      fpinfo_o = (PgFdwRelationInfo *) fpinfo->outerrel->fdw_private;

      /* Estimate of number of rows in cross product */
      nrows = fpinfo_i->rows * fpinfo_o->rows;

      /*
       * Back into an estimate of the number of retrieved rows.  Just in
       * case this is nuts, clamp to at most nrow.
       */
      retrieved_rows = clamp_row_est(rows / fpinfo->local_conds_sel);
      retrieved_rows = Min(retrieved_rows, nrows);

      /*
       * The cost of foreign join is estimated as cost of generating
       * rows for the joining relations + cost for applying quals on the
       * rows.
       */

      /*
       * Calculate the cost of clauses pushed down to the foreign server
       */
      cost_qual_eval(&remote_conds_cost, fpinfo->remote_conds, root);
      /* Calculate the cost of applying join clauses */
      cost_qual_eval(&join_cost, fpinfo->joinclauses, root);

      /*
       * Startup cost includes startup cost of joining relations and the
       * startup cost for join and other clauses. We do not include the
       * startup cost specific to join strategy (e.g. setting up hash
       * tables) since we do not know what strategy the foreign server
       * is going to use.
       */
      startup_cost = fpinfo_i->rel_startup_cost + fpinfo_o->rel_startup_cost;
      startup_cost += join_cost.startup;
      startup_cost += remote_conds_cost.startup;
      startup_cost += fpinfo->local_conds_cost.startup;

      /*
       * Run time cost includes:
       *
       * 1. Run time cost (total_cost - startup_cost) of relations being
       * joined
       *
       * 2. Run time cost of applying join clauses on the cross product
       * of the joining relations.
       *
       * 3. Run time cost of applying pushed down other clauses on the
       * result of join
       *
       * 4. Run time cost of applying nonpushable other clauses locally
       * on the result fetched from the foreign server.
       */
      run_cost = fpinfo_i->rel_total_cost - fpinfo_i->rel_startup_cost;
      run_cost += fpinfo_o->rel_total_cost - fpinfo_o->rel_startup_cost;
      run_cost += nrows * join_cost.per_tuple;
      nrows = clamp_row_est(nrows * fpinfo->joinclause_sel);
      run_cost += nrows * remote_conds_cost.per_tuple;
      run_cost += fpinfo->local_conds_cost.per_tuple * retrieved_rows;

      /* Add in tlist eval cost for each output row */
      startup_cost += foreignrel->reltarget->cost.startup;
      run_cost += foreignrel->reltarget->cost.per_tuple * rows;
    }
    else if (IS_UPPER_REL(foreignrel))
    {
      RelOptInfo *outerrel = fpinfo->outerrel;
      PgFdwRelationInfo *ofpinfo;
      AggClauseCosts aggcosts;
      double    input_rows;
      int     numGroupCols;
      double    numGroups = 1;

      /* The upper relation should have its outer relation set */
      Assert(outerrel);
      /* and that outer relation should have its reltarget set */
      Assert(outerrel->reltarget);

      /*
       * This cost model is mixture of costing done for sorted and
       * hashed aggregates in cost_agg().  We are not sure which
       * strategy will be considered at remote side, thus for
       * simplicity, we put all startup related costs in startup_cost
       * and all finalization and run cost are added in total_cost.
       */

      ofpinfo = (PgFdwRelationInfo *) outerrel->fdw_private;

      /* Get rows from input rel */
      input_rows = ofpinfo->rows;

      /* Collect statistics about aggregates for estimating costs. */
      MemSet(&aggcosts, 0, sizeof(AggClauseCosts));
      if (root->parse->hasAggs)
      {
        get_agg_clause_costs(root, (Node *) fpinfo->grouped_tlist,
                   AGGSPLIT_SIMPLE, &aggcosts);

        /*
         * The cost of aggregates in the HAVING qual will be the same
         * for each child as it is for the parent, so there's no need
         * to use a translated version of havingQual.
         */
        get_agg_clause_costs(root, (Node *) root->parse->havingQual,
                   AGGSPLIT_SIMPLE, &aggcosts);
      }

      /* Get number of grouping columns and possible number of groups */
      numGroupCols = list_length(root->parse->groupClause);
      numGroups = estimate_num_groups(root,
                      get_sortgrouplist_exprs(root->parse->groupClause,
                                  fpinfo->grouped_tlist),
                      input_rows, NULL);

      /*
       * Get the retrieved_rows and rows estimates.  If there are HAVING
       * quals, account for their selectivity.
       */
      if (root->parse->havingQual)
      {
        /* Factor in the selectivity of the remotely-checked quals */
        retrieved_rows =
          clamp_row_est(numGroups *
                  clauselist_selectivity(root,
                             fpinfo->remote_conds,
                             0,
                             JOIN_INNER,
                             NULL));
        /* Factor in the selectivity of the locally-checked quals */
        rows = clamp_row_est(retrieved_rows * fpinfo->local_conds_sel);
      }
      else
      {
        rows = retrieved_rows = numGroups;
      }

      /* Use width estimate made by the core code. */
      width = foreignrel->reltarget->width;

      /*-----
       * Startup cost includes:
       *    1. Startup cost for underneath input relation, adjusted for
       *       tlist replacement by apply_scanjoin_target_to_paths()
       *    2. Cost of performing aggregation, per cost_agg()
       *-----
       */
      startup_cost = ofpinfo->rel_startup_cost;
      startup_cost += outerrel->reltarget->cost.startup;
      startup_cost += aggcosts.transCost.startup;
      startup_cost += aggcosts.transCost.per_tuple * input_rows;
      startup_cost += aggcosts.finalCost.startup;
      startup_cost += (cpu_operator_cost * numGroupCols) * input_rows;

      /*-----
       * Run time cost includes:
       *    1. Run time cost of underneath input relation, adjusted for
       *       tlist replacement by apply_scanjoin_target_to_paths()
       *    2. Run time cost of performing aggregation, per cost_agg()
       *-----
       */
      run_cost = ofpinfo->rel_total_cost - ofpinfo->rel_startup_cost;
      run_cost += outerrel->reltarget->cost.per_tuple * input_rows;
      run_cost += aggcosts.finalCost.per_tuple * numGroups;
      run_cost += cpu_tuple_cost * numGroups;

      /* Account for the eval cost of HAVING quals, if any */
      if (root->parse->havingQual)
      {
        QualCost  remote_cost;

        /* Add in the eval cost of the remotely-checked quals */
        cost_qual_eval(&remote_cost, fpinfo->remote_conds, root);
        startup_cost += remote_cost.startup;
        run_cost += remote_cost.per_tuple * numGroups;
        /* Add in the eval cost of the locally-checked quals */
        startup_cost += fpinfo->local_conds_cost.startup;
        run_cost += fpinfo->local_conds_cost.per_tuple * retrieved_rows;
      }

      /* Add in tlist eval cost for each output row */
      startup_cost += foreignrel->reltarget->cost.startup;
      run_cost += foreignrel->reltarget->cost.per_tuple * rows;
    }
    else
    {
      Cost    cpu_per_tuple;

      /* Use rows/width estimates made by set_baserel_size_estimates. */
      rows = foreignrel->rows;
      width = foreignrel->reltarget->width;

      /*
       * Back into an estimate of the number of retrieved rows.  Just in
       * case this is nuts, clamp to at most foreignrel->tuples.
       */
      retrieved_rows = clamp_row_est(rows / fpinfo->local_conds_sel);
      retrieved_rows = Min(retrieved_rows, foreignrel->tuples);

      /*
       * Cost as though this were a seqscan, which is pessimistic.  We
       * effectively imagine the local_conds are being evaluated
       * remotely, too.
       */
      startup_cost = 0;
      run_cost = 0;
      run_cost += seq_page_cost * foreignrel->pages;

      startup_cost += foreignrel->baserestrictcost.startup;
      cpu_per_tuple = cpu_tuple_cost + foreignrel->baserestrictcost.per_tuple;
      run_cost += cpu_per_tuple * foreignrel->tuples;

      /* Add in tlist eval cost for each output row */
      startup_cost += foreignrel->reltarget->cost.startup;
      run_cost += foreignrel->reltarget->cost.per_tuple * rows;
    }

    /*
     * Without remote estimates, we have no real way to estimate the cost
     * of generating sorted output.  It could be free if the query plan
     * the remote side would have chosen generates properly-sorted output
     * anyway, but in most cases it will cost something.  Estimate a value
     * high enough that we won't pick the sorted path when the ordering
     * isn't locally useful, but low enough that we'll err on the side of
     * pushing down the ORDER BY clause when it's useful to do so.
     */
    if (pathkeys != NIL)
    {
      if (IS_UPPER_REL(foreignrel))
      {
        Assert(foreignrel->reloptkind == RELOPT_UPPER_REL &&
             fpinfo->stage == UPPERREL_GROUP_AGG);
        adjust_foreign_grouping_path_cost(root, pathkeys,
                          retrieved_rows, width,
                          fpextra->limit_tuples,
                          &startup_cost, &run_cost);
      }
      else
      {
        startup_cost *= DEFAULT_FDW_SORT_MULTIPLIER;
        run_cost *= DEFAULT_FDW_SORT_MULTIPLIER;
      }
    }

    total_cost = startup_cost + run_cost;

    /* Adjust the cost estimates if we have LIMIT */
    if (fpextra && fpextra->has_limit)
    {
      adjust_limit_rows_costs(&rows, &startup_cost, &total_cost,
                  fpextra->offset_est, fpextra->count_est);
      retrieved_rows = rows;
    }
  }

  /*
   * If this includes the final sort step, the given target, which will be
   * applied to the resulting path, might have different expressions from
   * the foreignrel's reltarget (see make_sort_input_target()); adjust tlist
   * eval costs.
   */
  if (fpextra && fpextra->has_final_sort &&
    fpextra->target != foreignrel->reltarget)
  {
    QualCost  oldcost = foreignrel->reltarget->cost;
    QualCost  newcost = fpextra->target->cost;

    startup_cost += newcost.startup - oldcost.startup;
    total_cost += newcost.startup - oldcost.startup;
    total_cost += (newcost.per_tuple - oldcost.per_tuple) * rows;
  }

  /*
   * Cache the retrieved rows and cost estimates for scans, joins, or
   * groupings without any parameterization, pathkeys, or additional
   * post-scan/join-processing steps, before adding the costs for
   * transferring data from the foreign server.  These estimates are useful
   * for costing remote joins involving this relation or costing other
   * remote operations on this relation such as remote sorts and remote
   * LIMIT restrictions, when the costs can not be obtained from the foreign
   * server.  This function will be called at least once for every foreign
   * relation without any parameterization, pathkeys, or additional
   * post-scan/join-processing steps.
   */
  if (pathkeys == NIL && param_join_conds == NIL && fpextra == NULL)
  {
    fpinfo->retrieved_rows = retrieved_rows;
    fpinfo->rel_startup_cost = startup_cost;
    fpinfo->rel_total_cost = total_cost;
  }

  /*
   * Add some additional cost factors to account for connection overhead
   * (fdw_startup_cost), transferring data across the network
   * (fdw_tuple_cost per retrieved row), and local manipulation of the data
   * (cpu_tuple_cost per retrieved row).
   */
  startup_cost += fpinfo->fdw_startup_cost;
  total_cost += fpinfo->fdw_startup_cost;
  total_cost += fpinfo->fdw_tuple_cost * retrieved_rows;
  total_cost += cpu_tuple_cost * retrieved_rows;

  /*
   * If we have LIMIT, we should prefer performing the restriction remotely
   * rather than locally, as the former avoids extra row fetches from the
   * remote that the latter might cause.  But since the core code doesn't
   * account for such fetches when estimating the costs of the local
   * restriction (see create_limit_path()), there would be no difference
   * between the costs of the local restriction and the costs of the remote
   * restriction estimated above if we don't use remote estimates (except
   * for the case where the foreignrel is a grouping relation, the given
   * pathkeys is not NIL, and the effects of a bounded sort for that rel is
   * accounted for in costing the remote restriction).  Tweak the costs of
   * the remote restriction to ensure we'll prefer it if LIMIT is a useful
   * one.
   */
  if (!fpinfo->use_remote_estimate &&
    fpextra && fpextra->has_limit &&
    fpextra->limit_tuples > 0 &&
    fpextra->limit_tuples < fpinfo->rows)
  {
    Assert(fpinfo->rows > 0);
    total_cost -= (total_cost - startup_cost) * 0.05 *
      (fpinfo->rows - fpextra->limit_tuples) / fpinfo->rows;
  }

  /* Return results. */
  *p_rows = rows;
  *p_width = width;
  *p_startup_cost = startup_cost;
  *p_total_cost = total_cost;
}


/*
 * Adjust the cost estimates of a foreign grouping path to include the cost of
 * generating properly-sorted output.
 */
static void
adjust_foreign_grouping_path_cost(PlannerInfo *root,
                  List *pathkeys,
                  double retrieved_rows,
                  double width,
                  double limit_tuples,
                  Cost *p_startup_cost,
                  Cost *p_run_cost)
{
  /*
   * If the GROUP BY clause isn't sort-able, the plan chosen by the remote
   * side is unlikely to generate properly-sorted output, so it would need
   * an explicit sort; adjust the given costs with cost_sort().  Likewise,
   * if the GROUP BY clause is sort-able but isn't a superset of the given
   * pathkeys, adjust the costs with that function.  Otherwise, adjust the
   * costs by applying the same heuristic as for the scan or join case.
   */
  if (!grouping_is_sortable(root->parse->groupClause) ||
    !pathkeys_contained_in(pathkeys, root->group_pathkeys))
  {
    Path    sort_path;  /* dummy for result of cost_sort */

    cost_sort(&sort_path,
          root,
          pathkeys,
          *p_startup_cost + *p_run_cost,
          retrieved_rows,
          width,
          0.0,
          work_mem,
          limit_tuples);

    *p_startup_cost = sort_path.startup_cost;
    *p_run_cost = sort_path.total_cost - sort_path.startup_cost;
  }
  else
  {
    /*
     * The default extra cost seems too large for foreign-grouping cases;
     * add 1/4th of that default.
     */
    double    sort_multiplier = 1.0 + (DEFAULT_FDW_SORT_MULTIPLIER
                       - 1.0) * 0.25;

    *p_startup_cost *= sort_multiplier;
    *p_run_cost *= sort_multiplier;
  }
}

/*
 * get_useful_ecs_for_relation
 *    Determine which EquivalenceClasses might be involved in useful
 *    orderings of this relation.
 *
 * This function is in some respects a mirror image of the core function
 * pathkeys_useful_for_merging: for a regular table, we know what indexes
 * we have and want to test whether any of them are useful.  For a foreign
 * table, we don't know what indexes are present on the remote side but
 * want to speculate about which ones we'd like to use if they existed.
 *
 * This function returns a list of potentially-useful equivalence classes,
 * but it does not guarantee that an EquivalenceMember exists which contains
 * Vars only from the given relation.  For example, given ft1 JOIN t1 ON
 * ft1.x + t1.x = 0, this function will say that the equivalence class
 * containing ft1.x + t1.x is potentially useful.  Supposing ft1 is remote and
 * t1 is local (or on a different server), it will turn out that no useful
 * ORDER BY clause can be generated.  It's not our job to figure that out
 * here; we're only interested in identifying relevant ECs.
 */
static List *
get_useful_ecs_for_relation(PlannerInfo *root, RelOptInfo *rel)
{
  List     *useful_eclass_list = NIL;
  ListCell   *lc;
  Relids    relids;

  /*
   * First, consider whether any active EC is potentially useful for a merge
   * join against this relation.
   */
  if (rel->has_eclass_joins)
  {
    foreach(lc, root->eq_classes)
    {
      EquivalenceClass *cur_ec = (EquivalenceClass *) lfirst(lc);

      if (eclass_useful_for_merging(root, cur_ec, rel))
        useful_eclass_list = lappend(useful_eclass_list, cur_ec);
    }
  }

  /*
   * Next, consider whether there are any non-EC derivable join clauses that
   * are merge-joinable.  If the joininfo list is empty, we can exit
   * quickly.
   */
  if (rel->joininfo == NIL)
    return useful_eclass_list;

  /* If this is a child rel, we must use the topmost parent rel to search. */
  if (IS_OTHER_REL(rel))
  {
    Assert(!bms_is_empty(rel->top_parent_relids));
    relids = rel->top_parent_relids;
  }
  else
    relids = rel->relids;

  /* Check each join clause in turn. */
  foreach(lc, rel->joininfo)
  {
    RestrictInfo *restrictinfo = (RestrictInfo *) lfirst(lc);

    /* Consider only mergejoinable clauses */
    if (restrictinfo->mergeopfamilies == NIL)
      continue;

    /* Make sure we've got canonical ECs. */
    update_mergeclause_eclasses(root, restrictinfo);

    /*
     * restrictinfo->mergeopfamilies != NIL is sufficient to guarantee
     * that left_ec and right_ec will be initialized, per comments in
     * distribute_qual_to_rels.
     *
     * We want to identify which side of this merge-joinable clause
     * contains columns from the relation produced by this RelOptInfo. We
     * test for overlap, not containment, because there could be extra
     * relations on either side.  For example, suppose we've got something
     * like ((A JOIN B ON A.x = B.x) JOIN C ON A.y = C.y) LEFT JOIN D ON
     * A.y = D.y.  The input rel might be the joinrel between A and B, and
     * we'll consider the join clause A.y = D.y. relids contains a
     * relation not involved in the join class (B) and the equivalence
     * class for the left-hand side of the clause contains a relation not
     * involved in the input rel (C).  Despite the fact that we have only
     * overlap and not containment in either direction, A.y is potentially
     * useful as a sort column.
     *
     * Note that it's even possible that relids overlaps neither side of
     * the join clause.  For example, consider A LEFT JOIN B ON A.x = B.x
     * AND A.x = 1.  The clause A.x = 1 will appear in B's joininfo list,
     * but overlaps neither side of B.  In that case, we just skip this
     * join clause, since it doesn't suggest a useful sort order for this
     * relation.
     */
    if (bms_overlap(relids, restrictinfo->right_ec->ec_relids))
      useful_eclass_list = list_append_unique_ptr(useful_eclass_list,
                            restrictinfo->right_ec);
    else if (bms_overlap(relids, restrictinfo->left_ec->ec_relids))
      useful_eclass_list = list_append_unique_ptr(useful_eclass_list,
                            restrictinfo->left_ec);
  }

  return useful_eclass_list;
}
/*
 * get_useful_pathkeys_for_relation
 *    Determine which orderings of a relation might be useful.
 *
 * Getting data in sorted order can be useful either because the requested
 * order matches the final output ordering for the overall query we're
 * planning, or because it enables an efficient merge join.  Here, we try
 * to figure out which pathkeys to consider.
 */
static List *
get_useful_pathkeys_for_relation(PlannerInfo *root, RelOptInfo *rel)
{
  List     *useful_pathkeys_list = NIL;
  List     *useful_eclass_list;
  PgFdwRelationInfo *fpinfo = (PgFdwRelationInfo *) rel->fdw_private;
  EquivalenceClass *query_ec = NULL;
  ListCell   *lc;

  /*
   * Pushing the query_pathkeys to the remote server is always worth
   * considering, because it might let us avoid a local sort.
   */
  fpinfo->qp_is_pushdown_safe = false;
  if (root->query_pathkeys)
  {
    bool    query_pathkeys_ok = true;

    foreach(lc, root->query_pathkeys)
    {
      PathKey    *pathkey = (PathKey *) lfirst(lc);
      EquivalenceClass *pathkey_ec = pathkey->pk_eclass;
      Expr     *em_expr;

      /*
       * The planner and executor don't have any clever strategy for
       * taking data sorted by a prefix of the query's pathkeys and
       * getting it to be sorted by all of those pathkeys. We'll just
       * end up resorting the entire data set.  So, unless we can push
       * down all of the query pathkeys, forget it.
       *
       * is_foreign_expr would detect volatile expressions as well, but
       * checking ec_has_volatile here saves some cycles.
       */
      if (pathkey_ec->ec_has_volatile ||
        !(em_expr = find_em_expr_for_rel(pathkey_ec, rel)) ||
        !is_foreign_expr(root, rel, em_expr))
      {
        query_pathkeys_ok = false;
        break;
      }
    }

    if (query_pathkeys_ok)
    {
      useful_pathkeys_list = list_make1(list_copy(root->query_pathkeys));
      fpinfo->qp_is_pushdown_safe = true;
    }
  }

  /*
   * Even if we're not using remote estimates, having the remote side do the
   * sort generally won't be any worse than doing it locally, and it might
   * be much better if the remote side can generate data in the right order
   * without needing a sort at all.  However, what we're going to do next is
   * try to generate pathkeys that seem promising for possible merge joins,
   * and that's more speculative.  A wrong choice might hurt quite a bit, so
   * bail out if we can't use remote estimates.
   */
  if (!fpinfo->use_remote_estimate)
    return useful_pathkeys_list;

  /* Get the list of interesting EquivalenceClasses. */
  useful_eclass_list = get_useful_ecs_for_relation(root, rel);

  /* Extract unique EC for query, if any, so we don't consider it again. */
  if (list_length(root->query_pathkeys) == 1)
  {
    PathKey    *query_pathkey = (PathKey*)linitial(root->query_pathkeys);

    query_ec = query_pathkey->pk_eclass;
  }

  /*
   * As a heuristic, the only pathkeys we consider here are those of length
   * one.  It's surely possible to consider more, but since each one we
   * choose to consider will generate a round-trip to the remote side, we
   * need to be a bit cautious here.  It would sure be nice to have a local
   * cache of information about remote index definitions...
   */
  foreach(lc, useful_eclass_list)
  {
    EquivalenceClass *cur_ec = (EquivalenceClass*)lfirst(lc);
    Expr     *em_expr;
    PathKey    *pathkey;

    /* If redundant with what we did above, skip it. */
    if (cur_ec == query_ec)
      continue;

    /* If no pushable expression for this rel, skip it. */
    em_expr = find_em_expr_for_rel(cur_ec, rel);
    if (em_expr == NULL || !is_foreign_expr(root, rel, em_expr))
      continue;

    /* Looks like we can generate a pathkey, so let's do it. */
    pathkey = make_canonical_pathkey(root, cur_ec,
                     linitial_oid(cur_ec->ec_opfamilies),
                     BTLessStrategyNumber,
                     false);
    useful_pathkeys_list = lappend(useful_pathkeys_list,
                     list_make1(pathkey));
  }

  return useful_pathkeys_list;
}

static void
add_paths_with_pathkeys_for_rel(PlannerInfo *root, RelOptInfo *rel,
                Path *epq_path)
{
  List     *useful_pathkeys_list = NIL; /* List of all pathkeys */
  ListCell   *lc;

  useful_pathkeys_list = get_useful_pathkeys_for_relation(root, rel);

  /* Create one path for each set of pathkeys we found above. */
  foreach(lc, useful_pathkeys_list)
  {
    double    rows;
    int     width;
    Cost    startup_cost;
    Cost    total_cost;
    List     *useful_pathkeys = (List*)lfirst(lc);
    Path     *sorted_epq_path;

    estimate_path_cost_size(root, rel, NIL, useful_pathkeys, NULL,
                &rows, &width, &startup_cost, &total_cost);

    /*
     * The EPQ path must be at least as well sorted as the path itself, in
     * case it gets used as input to a mergejoin.
     */
    sorted_epq_path = epq_path;
    if (sorted_epq_path != NULL &&
      !pathkeys_contained_in(useful_pathkeys,
                   sorted_epq_path->pathkeys))
      sorted_epq_path = (Path *)
        create_sort_path(root,
                 rel,
                 sorted_epq_path,
                 useful_pathkeys,
                 -1.0);

    if (IS_SIMPLE_REL(rel))
      add_path(rel, (Path *)
           create_foreignscan_path(root, rel,
                       NULL,
                       rows,
                       startup_cost,
                       total_cost,
                       useful_pathkeys,
                       rel->lateral_relids,
                       sorted_epq_path,
                       NIL));
    else
      add_path(rel, (Path *)
           create_foreign_join_path(root, rel,
                        NULL,
                        rows,
                        startup_cost,
                        total_cost,
                        useful_pathkeys,
                        rel->lateral_relids,
                        sorted_epq_path,
                        NIL));
  }
}


/*
 * Create cursor for node's query with current parameter values.
 */
void create_cursor(ForeignScanState *node)
{
  #if 0
  PgFdwScanState *fsstate = (PgFdwScanState *) node->fdw_state;
  ExprContext *econtext = node->ss.ps.ps_ExprContext;
  int     numParams = fsstate->numParams;
  const char **values = fsstate->param_values;
  PGconn     *conn = fsstate->conn;
  StringInfoData buf;
  PGresult   *res;

  /*
   * Construct array of query parameter values in text format.  We do the
   * conversions in the short-lived per-tuple context, so as not to cause a
   * memory leak over repeated scans.
   */
  if (numParams > 0)
  {
    MemoryContext oldcontext;

    oldcontext = MemoryContextSwitchTo(econtext->ecxt_per_tuple_memory);

    process_query_params(econtext,
               fsstate->param_flinfo,
               fsstate->param_exprs,
               values);

    MemoryContextSwitchTo(oldcontext);
  }

  /* Construct the DECLARE CURSOR command */
  initStringInfo(&buf);
  appendStringInfo(&buf, "DECLARE c%u CURSOR FOR\n%s",
           fsstate->cursor_number, fsstate->query);

  /*
   * Notice that we pass NULL for paramTypes, thus forcing the remote server
   * to infer types for all parameters.  Since we explicitly cast every
   * parameter (see deparse.c), the "inference" is trivial and will produce
   * the desired result.  This allows us to avoid assuming that the remote
   * server has the same OIDs we do for the parameters' types.
   */
  if (!PQsendQueryParams(conn, buf.data, numParams,
               NULL, values, NULL, NULL, 0))
    pgfdw_report_error(ERROR, NULL, conn, false, buf.data);

  /*
   * Get the result, and check for success.
   *
   * We don't use a PG_TRY block here, so be careful not to throw error
   * without releasing the PGresult.
   */
  res = pgfdw_get_result(conn, buf.data);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
    pgfdw_report_error(ERROR, res, conn, true, fsstate->query);
  PQclear(res);

  /* Mark the cursor as created, and show no tuples have been retrieved */
  fsstate->cursor_exists = true;
  fsstate->tuples = NULL;
  fsstate->num_tuples = 0;
  fsstate->next_tuple = 0;
  fsstate->fetch_ct_2 = 0;
  fsstate->eof_reached = false;

  /* Clean up */
  pfree(buf.data);
  #endif
}

/*
 * Fetch some more rows from the node's cursor.
 */
void fetch_more_data(ForeignScanState *node)
{

  #if 0
  PgFdwScanState *fsstate = (PgFdwScanState *) node->fdw_state;
  PGresult   *volatile res = NULL;
  MemoryContext oldcontext;

  /*
   * We'll store the tuples in the batch_cxt.  First, flush the previous
   * batch.
   */
  fsstate->tuples = NULL;
  MemoryContextReset(fsstate->batch_cxt);
  oldcontext = MemoryContextSwitchTo(fsstate->batch_cxt);

  /* PGresult must be released before leaving this function. */
  PG_TRY();
  {
    PGconn     *conn = fsstate->conn;
    char    sql[64];
    int     numrows;
    int     i;

    snprintf(sql, sizeof(sql), "FETCH %d FROM c%u",
         fsstate->fetch_size, fsstate->cursor_number);

    res = pgfdw_exec_query(conn, sql);
    /* On error, report the original query, not the FETCH. */
    if (PQresultStatus(res) != PGRES_TUPLES_OK)
      pgfdw_report_error(ERROR, res, conn, false, fsstate->query);

    /* Convert the data into HeapTuples */
    numrows = PQntuples(res);
    fsstate->tuples = (HeapTuple *) palloc0(numrows * sizeof(HeapTuple));
    fsstate->num_tuples = numrows;
    fsstate->next_tuple = 0;

    for (i = 0; i < numrows; i++)
    {
      Assert(IsA(node->ss.ps.plan, ForeignScan));

      fsstate->tuples[i] =
        make_tuple_from_result_row(res, i,
                       fsstate->rel,
                       fsstate->attinmeta,
                       fsstate->retrieved_attrs,
                       node,
                       fsstate->temp_cxt);
    }

    /* Update fetch_ct_2 */
    if (fsstate->fetch_ct_2 < 2)
      fsstate->fetch_ct_2++;

    /* Must be EOF if we didn't get as many tuples as we asked for. */
    fsstate->eof_reached = (numrows < fsstate->fetch_size);

    PQclear(res);
    res = NULL;
  }
  PG_CATCH();
  {
    if (res)
      PQclear(res);
    PG_RE_THROW();
  }
  PG_END_TRY();

  MemoryContextSwitchTo(oldcontext);
  #endif
}

/*
 * Utility routine to close a cursor.
 */
void close_cursor(PGconn *conn, unsigned int cursor_number)
{
  char    sql[64];
  PGresult   *res;

  snprintf(sql, sizeof(sql), "CLOSE c%u", cursor_number);

  /*
   * We don't use a PG_TRY block here, so be careful not to throw error
   * without releasing the PGresult.
   */
  // res = pgfdw_exec_query(conn, sql);
  // if (PQresultStatus(res) != PGRES_COMMAND_OK)
  //   pgfdw_report_error(ERROR, res, conn, true, sql);
  // PQclear(res);
}
