static char *VersionID = VERSIONID;
/* "Version 7.3  trec_eval Dec 15, 2004"; */

/* Copyright (c) 2004, 2003, 1991, 1990, 1984 - Chris Buckley. */

/********************   PROCEDURE DESCRIPTION   ************************
 *0 Take TREC results text file, TREC qrels file, and evaluate
 *1 local.convert.obj.trec_eval
 *2 trec_eval [-q] [-a] [-t] [-o] [-v] [-n num] trec_rel_file trec_top_file

 *7 Read text tuples from trec_top_file of the form
 *7     030  Q0  ZF08-175-870  0   4238   prise1
 *7     qid iter   docno      rank  sim   run_id
 *7 giving TREC document numbers (a string) retrieved by query qid 
 *7 (an integer) with similarity sim (a float).  The other fields are ignored.
 *7 Input is asssumed to be sorted numerically by qid.
 *7 Sim is assumed to be higher for the docs to be retrieved first.
 *7 Relevance for each docno to qid is determined from text_qrels_file, which
 *7 consists of text tuples of the form
 *7    qid  iter  docno  rel
 *7 giving TREC document numbers (a string) and their relevance to query qid
 *7 (a non-negative integer less than 128, or -1 to indicate unjudged). 
 *7 Tuples are asssumed to be sorted numerically by qid.
 *7 The text tuples with relevence judgements are converted to TR_VEC form
 *7 and then submitted to the evaluation routines.
 *7
 *7 -q: In addition to summary evaluation, give evaluation for each query
 *7 -a: Print all evaluation measures calculated, instead of just the
 *7      official measures for TREC 2.
 *7 -o: Print everything out in old, non-relational format
 *7 -v: Print version number and exit
 *7 -h: Print full help message and exit
 *7 -t: Treat similarity as time that document retrieved.  Compute
 *7      several time-based measures after ranking docs by time retrieved
 *7      (first doc (lowest sim) retrieved ranked highest). 
 *7      Only done if -a selected.
 *7 -J: Calculate all measures only over judged documents that appear 
 *7     in qrels. (DO NOT USE)
 *7 -n<num>: following integer is the number of queries to average over.
 *7 -ua<num>: Value to use for 'a' coefficient of utility computation.
 *7 -ub<num>: Value to use for 'b' coefficient of utility computation.
 *7 -uc<num>: Value to use for 'c' coefficient of utility computation.
 *7 -ud<num>: Value to use for 'd' coefficient of utility computation.
 *7 -N<num>: Number of docs in collection
 *7 -M<num>:Max number of results to evaluate per topic

 *8 Procedure is to read all the docs retrieved for a query, and all the
 *8 relevant docs for that query,
 *8 sort and rank the retrieved docs by sim/docno, 
 *8 and look up docno in the relevant docs to determine relevance.
 *8 The qid,did,rank,sim,rel fields of of TR_VEC are filled in; 
 *8 action,iter fields are set to 0.
 *8 Queries for which there is no relevance information are ignored completely.

***********************************************************************/

#include "common.h"
#include "sysfunc.h"
#include "smart_error.h"
#include "tr_vec.h"
#include "trec_eval.h"
#include "buf.h"

void print_error();
void old_print_trec_eval_list();
void print_rel_trec_eval_list();

int trec_eval_help(EVAL_PARAM_INFO *epi);
int accumulate_results (TREC_EVAL *query_eval, TREC_EVAL *accum_eval);
int get_top (char *trec_top_file, ALL_TREC_TOP *all_trec_top);
int get_qrels (char *text_qrels_file, ALL_TREC_QRELS *all_trec_qrels);
int form_trvec (EVAL_PARAM_INFO *ep, TREC_TOP *trec_top,
		TREC_QRELS *trec_qrels, TR_VEC *tr_vec, long *num_rel);
int trvec_trec_eval (EVAL_PARAM_INFO *epi, TR_VEC *tr_vec,
		     TREC_EVAL *eval, long num_rel, long num_nonrel);

static char *usage = "Usage: trec_eval [-h] [-q] [-a] [-o] [-v] trec_rel_file trec_top_file\n\
   -h: Give full help information, including other options\n\
   -q: In addition to summary evaluation, give evaluation for each query\n\
   -a: Print all evaluation measures, instead of just official measures\n\
   -o: Print requested measures in old non-relational format\n";


int
main (argc, argv)
int argc;
char *argv[];
{
    char *trec_rel_file, *trec_top_file;
    ALL_TREC_TOP all_trec_top;
    ALL_TREC_QRELS all_trec_qrels;
    TREC_EVAL accum_eval, query_eval;
    TR_VEC tr_vec;
    long num_rel;
    long num_eval_q;
    long i,j;
    EVAL_PARAM_INFO epi;

    /* Initialize static info before getting program optional args */
    epi.query_flag = epi.all_flag = epi.time_flag = epi.average_complete_flag = 0;
    epi.judged_docs_only_flag = 0;
    epi.relation_flag = 1;
    epi.utility_a = UTILITY_A; epi.utility_b = UTILITY_B;
    epi.utility_c = UTILITY_C; epi.utility_d = UTILITY_D;
    epi.num_docs_in_coll = 0;
    epi.relevance_level = 1;
    epi.max_num_docs_per_topic = MAXLONG;

    /* Should use getopts, but some people may not have it. */
    /* This keeps growing over the years.  Should redo */
    while (argc > 1 && argv[1][0] == '-') {
        if (argv[1][1] == 'q')
            epi.query_flag++;
	else if (argv[1][1] == 'v') {
	    fprintf (stderr, "trec_eval version %s\n", VersionID);
	    exit (0);
	}
	else if (argv[1][1] == 'h') {
	    (void) trec_eval_help(&epi);
	    exit (0);
	}
        else if (argv[1][1] == 'a')
            epi.all_flag++;
        else if (argv[1][1] == 'o')
            epi.relation_flag = 0;
        else if (argv[1][1] == 'c') {
	    epi.average_complete_flag++;
	}
        else if (argv[1][1] == 'l') {
	    epi.relevance_level = atol (&argv[1][2]);
	}
        else if (argv[1][1] == 'J') {
	    epi.judged_docs_only_flag++;
	}
        else if (argv[1][1] == 'N')
            epi.num_docs_in_coll = atol (&argv[1][2]);
        else if (argv[1][1] == 'M')
            epi.max_num_docs_per_topic = atol (&argv[1][2]);
        else if (argv[1][1] == 'U') {
	    if (argv[1][2] == 'a')
		epi.utility_a = atof (&argv[1][3]);
	    else if (argv[1][2] == 'b')
		epi.utility_b = atof (&argv[1][3]);
	    else if (argv[1][2] == 'c')
		epi.utility_c = atof (&argv[1][3]);
	    else if (argv[1][2] == 'd')
		epi.utility_d = atof (&argv[1][3]);
	    else {
		(void) fputs (usage,stderr);
		exit (1);
	    }
	}
        else if (argv[1][1] == 'T')
            epi.time_flag++;
        else {
            (void) fputs (usage,stderr);
            exit (1);
        }
        argc--; argv++;
    }

    if (argc != 3) {
        (void) fputs (usage,stderr);
        exit (1);
    }

    trec_rel_file = argv[1];
    trec_top_file = argv[2];

    /* Get qrels and top results information for all queries from the
       input text files */
    if (UNDEF == get_qrels (trec_rel_file, &all_trec_qrels) ||
	UNDEF == get_top (trec_top_file, &all_trec_top)) {
        print_error ("trec_eval: input error", "Quit");
        exit (2);
    }

    /* For each topic which has both qrels and top results information,
       calculate, possibly print (if query_flag), and accumulate
       evaluation measures. */
    num_eval_q = 0;
    (void) memset ((void *) &accum_eval, 0, sizeof (TREC_EVAL));
    accum_eval.qid = "All";

    for (i = 0; i < all_trec_top.num_q_tr; i++) {
	/* Find rel info for this query (skip if no rel info) */
	for (j = 0; j < all_trec_qrels.num_q_qrels; j++) {
	    if (0 == strcmp (all_trec_top.trec_top[i].qid,
			     all_trec_qrels.trec_qrels[j].qid))
		break;
	}
	if (j >= all_trec_qrels.num_q_qrels)
	    continue;

	/* Form results/rel into SMART TR_VEC form */
	if (UNDEF == form_trvec (&epi,
				 &all_trec_top.trec_top[i],
				 &all_trec_qrels.trec_qrels[j],
				 &tr_vec,
				 &num_rel)) {
	    print_error ("trec_eval: form_tr_vec error", "Quit");
	    exit (3);
	}

	/* Evaluate results/rel for this query */
	if (UNDEF == trvec_trec_eval (&epi,
				      &tr_vec,
				      &query_eval,
				      num_rel,
				      all_trec_qrels.trec_qrels[j].num_text_qrels - num_rel)) {
	    print_error ("trec_eval: evaluation error", "Quit");
	    exit (4);
	}

	/* Print results for this query, if desired */
	if (epi.query_flag) {
	    if (epi.relation_flag)
		print_rel_trec_eval_list (1, &epi, &query_eval, (SM_BUF *) NULL);
	    else
		old_print_trec_eval_list (&epi, &query_eval, 1, (SM_BUF *) NULL);
        }

	/* Accumulate results for later averaging */
	if (UNDEF == accumulate_results (&query_eval, &accum_eval)) {
	    print_error ("trec_eval: accumulation error", "Quit");
	    exit (5);
	}

	num_eval_q++;
    }

    /******** REMOVE THIS ONCE WARNING FLAG ADDED */
    /* Warn if numq_flag_num < num_eval_q */
    if (num_eval_q == 0) {
	    set_error (SM_INCON_ERR,
		       "No queries with both results and relevance info",
		       "trec_eval");
	    return (UNDEF);
	print_error ("trec_eval", "Quit");
	exit (6);
    }

    if (epi.average_complete_flag) {
	/* Want to average over possibly missing queries.  Pass in actual
	 *  number of queries in num_orig_queries */
	accum_eval.num_orig_queries = accum_eval.num_queries;
	accum_eval.num_queries = all_trec_qrels.num_q_qrels;
    }

    /* Print final evaluation results */
    if (epi.relation_flag)
	print_rel_trec_eval_list (0, &epi, &accum_eval, (SM_BUF *) NULL);
    else
	old_print_trec_eval_list (&epi, &accum_eval, 1, (SM_BUF *) NULL);

    exit (0);
}
