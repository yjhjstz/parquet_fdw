#include <sys/stat.h>

#include "postgres.h"
#include "fmgr.h"

#include "access/reloptions.h"
#include "catalog/pg_foreign_table.h"
#include "commands/defrem.h"
#include "commands/explain.h"
#include "foreign/fdwapi.h"
#include "optimizer/planmain.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/elog.h"


PG_MODULE_MAGIC;

void _PG_init(void);

/* FDW routines */
extern void parquetGetForeignRelSize(PlannerInfo *root,
                      RelOptInfo *baserel,
                      Oid foreigntableid);
extern void parquetGetForeignPaths(PlannerInfo *root,
                    RelOptInfo *baserel,
                    Oid foreigntableid);
extern ForeignScan *parquetGetForeignPlan(PlannerInfo *root,
                      RelOptInfo *baserel,
                      Oid foreigntableid,
                      ForeignPath *best_path,
                      List *tlist,
                      List *scan_clauses,
                      Plan *outer_plan);
extern TupleTableSlot *parquetIterateForeignScan(ForeignScanState *node);
extern void parquetBeginForeignScan(ForeignScanState *node, int eflags);
extern void parquetEndForeignScan(ForeignScanState *node);
extern void parquetReScanForeignScan(ForeignScanState *node);
extern int parquetAcquireSampleRowsFunc(Relation relation, int elevel,
                             HeapTuple *rows, int targrows,
                             double *totalrows,
                             double *totaldeadrows);
extern bool parquetAnalyzeForeignTable (Relation relation,
                            AcquireSampleRowsFunc *func,
                            BlockNumber *totalpages);
extern void parquetExplainForeignScan(ForeignScanState *node, ExplainState *es);
extern bool parquetIsForeignScanParallelSafe(PlannerInfo *root, RelOptInfo *rel,
                                             RangeTblEntry *rte);
extern Size parquetEstimateDSMForeignScan(ForeignScanState *node,
                                          ParallelContext *pcxt);
extern void parquetInitializeDSMForeignScan(ForeignScanState *node,
                                            ParallelContext *pcxt,
                                            void *coordinate);
extern void parquetReInitializeDSMForeignScan(ForeignScanState *node,
                                              ParallelContext *pcxt,
                                              void *coordinate);
extern void parquetInitializeWorkerForeignScan(ForeignScanState *node,
                                               shm_toc *toc,
                                               void *coordinate);
extern void parquetShutdownForeignScan(ForeignScanState *node);
extern List *parquetImportForeignSchema(ImportForeignSchemaStmt *stmt, Oid serverOid);

/* for insert  */
extern List *parquetPlanForeignModify(PlannerInfo *root,
        ModifyTable *plan,
        Index resultRelation,
        int subplan_index);
extern void parquetBeginForeignModify(ModifyTableState *mtstate,
        ResultRelInfo *resultRelInfo,
        List *fdw_private,
        int subplan_index,
        int eflags);
extern void parquetBeginForeignInsert(ModifyTableState *mtstate,
        ResultRelInfo *resultRelInfo);
extern TupleTableSlot *parquetExecForeignInsert(EState *estate,
        ResultRelInfo *resultRelInfo,
        TupleTableSlot *slot,
        TupleTableSlot *planSlot);
extern void parquetEndForeignInsert(EState *estate,
            ResultRelInfo *rrinfo);

extern void parquetExplainForeignModify(ModifyTableState *mtstate,
              ResultRelInfo *rinfo,
              List *fdw_private,
              int subplan_index,
              struct ExplainState *es);
extern void parquetEndForeignModify(EState *estate,
            ResultRelInfo *rrinfo);

extern void
parquetGetForeignUpperPaths(PlannerInfo *root, UpperRelationKind stage,
               RelOptInfo *input_rel, RelOptInfo *output_rel,
               void *extra);
extern void
parquetGetForeignJoinPaths(PlannerInfo *root,
              RelOptInfo *joinrel,
              RelOptInfo *outerrel,
              RelOptInfo *innerrel,
              JoinType jointype,
              JoinPathExtraData *extra);
/* GUC variable */
extern bool parquet_fdw_use_threads;

void
_PG_init(void)
{
	DefineCustomBoolVariable("parquet_fdw.use_threads",
							"Enables use_thread option",
							NULL,
							&parquet_fdw_use_threads,
							true,
							PGC_USERSET,
							0,
							NULL,
							NULL,
							NULL);

}

PG_FUNCTION_INFO_V1(parquet_fdw_handler);
Datum
parquet_fdw_handler(PG_FUNCTION_ARGS)
{
    FdwRoutine *fdwroutine = makeNode(FdwRoutine);

    fdwroutine->GetForeignRelSize = parquetGetForeignRelSize;
    fdwroutine->GetForeignPaths = parquetGetForeignPaths;
    fdwroutine->GetForeignPlan = parquetGetForeignPlan;
    fdwroutine->BeginForeignScan = parquetBeginForeignScan;
    fdwroutine->IterateForeignScan = parquetIterateForeignScan;
    fdwroutine->ReScanForeignScan = parquetReScanForeignScan;
    fdwroutine->EndForeignScan = parquetEndForeignScan;
    fdwroutine->AnalyzeForeignTable = parquetAnalyzeForeignTable;
    fdwroutine->ExplainForeignScan = parquetExplainForeignScan;
    fdwroutine->IsForeignScanParallelSafe = parquetIsForeignScanParallelSafe;
    fdwroutine->EstimateDSMForeignScan = parquetEstimateDSMForeignScan;
    fdwroutine->InitializeDSMForeignScan = parquetInitializeDSMForeignScan;
    fdwroutine->ReInitializeDSMForeignScan = parquetReInitializeDSMForeignScan;
    fdwroutine->InitializeWorkerForeignScan = parquetInitializeWorkerForeignScan;
    fdwroutine->ShutdownForeignScan = parquetShutdownForeignScan;
    fdwroutine->ImportForeignSchema = parquetImportForeignSchema;

    /* INSERT/DELETE support */
    fdwroutine->PlanForeignModify      = parquetPlanForeignModify;
    fdwroutine->BeginForeignModify     = parquetBeginForeignModify;
    fdwroutine->BeginForeignInsert     = parquetBeginForeignInsert;
    fdwroutine->ExecForeignInsert      = parquetExecForeignInsert;
    fdwroutine->EndForeignModify       = parquetEndForeignModify;
    fdwroutine->EndForeignInsert       = parquetEndForeignInsert;
    fdwroutine->ExplainForeignModify   = parquetExplainForeignModify;

    /* Support functions for join push-down */
    fdwroutine->GetForeignJoinPaths = parquetGetForeignJoinPaths;
    /* Support functions for upper relation push-down */
    fdwroutine->GetForeignUpperPaths = parquetGetForeignUpperPaths;

    PG_RETURN_POINTER(fdwroutine);
}

PG_FUNCTION_INFO_V1(parquet_fdw_validator);
Datum
parquet_fdw_validator(PG_FUNCTION_ARGS)
{
    List       *options_list = untransformRelOptions(PG_GETARG_DATUM(0));
    Oid         catalog = PG_GETARG_OID(1);
    ListCell   *lc;
    bool        filename_provided = false;

    /* Only check table options */
    if (catalog != ForeignTableRelationId)
        PG_RETURN_VOID();

    foreach(lc, options_list)
    {
        DefElem    *def = (DefElem *) lfirst(lc);

        if (strcmp(def->defname, "filename") == 0)
        {
            struct stat stat_buf;

            if (stat(defGetString(def), &stat_buf) != 0)
            {
                char     *filename = defGetString(def);
                //elog(INFO, "touch file %s", filename);
                File flip = PathNameOpenFile(filename, O_RDWR | O_CREAT | O_EXCL | PG_BINARY);
                if (flip > 0) {
                    FileClose(flip);
                } else {
                    int e = errno;

                    ereport(ERROR,
                          (errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
                           errmsg("parquet_fdw_validator: %s", strerror(e))));
                }
                
            }
            filename_provided = true;
        }
        else if (strcmp(def->defname, "sorted") == 0)
            ;  /* do nothing */
        else if (strcmp(def->defname, "batch_size") == 0)
            /* Check that int value is valid */
            strtol(defGetString(def), NULL, 10);
        else if (strcmp(def->defname, "use_mmap") == 0)
        {
            /* Check that bool value is valid */
            bool    use_mmap;

            if (!parse_bool(defGetString(def), &use_mmap))
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                         errmsg("invalid value for boolean option \"%s\": %s",
                                def->defname, defGetString(def))));
        }
        else if (strcmp(def->defname, "use_threads") == 0)
        {
            /* Check that bool value is valid */
            bool    use_threads;

            if (!parse_bool(defGetString(def), &use_threads))
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                         errmsg("invalid value for boolean option \"%s\": %s",
                                def->defname, defGetString(def))));
        }
        else
        {
            ereport(ERROR,
                    (errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
                     errmsg("parquet_fdw: invalid option \"%s\"",
                            def->defname)));
        }
    }

    if (!filename_provided)
        elog(ERROR, "parquet_fdw: filename is required");

    PG_RETURN_VOID();
}

