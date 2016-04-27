/* Copyright (c) 2003, 1991, 1990, 1984 - Chris Buckley. */

#include "common.h"
#include "trec_eval.h"

static char *help_message = 
"trec_eval [-h] [-q] [-a] [-o] [-c] [-l<num>  [-N<num>] [-M<num>] [-Ua<num>] [-Ub<num>] [-Uc<num>] [-Ud<num>] [-T] trec_rel_file trec_top_file \n\
 \n\
Calculate and print various evaluation measures, evaluating the results  \n\
in trec_top_file against the relevance judgements in trec_rel_file. \n\
 \n\
There are a fair number of options, of which only the lower case options are \n\
normally ever used.   \n\
 -h: Print full help message and exit \n\
 -q: In addition to summary evaluation, give evaluation for each query \n\
 -a: Print all evaluation measures calculated, instead of just the \n\
     main official measures for TREC. \n\
 -o: Print everything out in old, nonrelational format (default is relational) \n\
 -c: Average over the complete set of queries in the relevance judgements  \n\
      instead of the queries in the intersection of relevance judgements \n\
      and results.  Missing queries will contribute a value of 0 to all \n\
      evaluation measures (which may or may not be reasonable for a  \n\
      particular evaluation measure, but is reasonable for standard TREC \n\
      measures.) \n\
 -l<num>: Num indicates the minimum relevance judgement value needed for \n\
      a document to be called relevant. (All measures used by TREC eval are \n\
      based on binary relevance).  Used if trec_rel_file contains relevance \n\
      judged on a multi-relevance scale.  Default is 1. \n\
 -N<num>: Number of docs in collection \n\
 -M<num>: Max number of docs per topic to use in evaluation (discard rest). \n\
 -Ua<num>: Value to use for 'a' coefficient of utility computation. \n\
                        relevant  nonrelevant \n\
      retrieved            a          b \n\
      nonretrieved         c          d \n\
 -Ub<num>: Value to use for 'b' coefficient of utility computation. \n\
 -Uc<num>: Value to use for 'c' coefficient of utility computation. \n\
 -Ud<num>: Value to use for 'd' coefficient of utility computation. \n\
 -J: Calculate all values only over the judged (either relevant or  \n\
     nonrelevant) documents.  All unjudged documents are removed from the \n\
     retrieved set before any calculations (possibly leaving an empty set). \n\
     DO NOT USE, unless you really know what you're doing - very easy to get \n\
     reasonable looking, but invalid, numbers.  \n\
 -T: Treat similarity as time that document retrieved.  Compute \n\
      several time-based measures after ranking docs by time retrieved \n\
      (first doc (lowest sim) retrieved ranked highest).  \n\
      Only done if -a selected. \n\
 \n\
 \n\
Read text tuples from trec_top_file of the form \n\
     030  Q0  ZF08-175-870  0   4238   prise1 \n\
     qid iter   docno      rank  sim   run_id \n\
giving TREC document numbers (a string) retrieved by query qid  \n\
(a string) with similarity sim (a float).  The other fields are ignored, \n\
with the exception that the run_id field of the last line is kept and \n\
output.  In particular, note that the rank field is ignored here; \n\
internally ranks are assigned by sorting by the sim field with ties  \n\
broken deterministicly (using docno). \n\
Sim is assumed to be higher for the docs to be retrieved first. \n\
File may contain no NULL characters. \n\
Lines may contain fields after the run_id; they are ignored. \n\
 \n\
Relevance for each docno to qid is determined from text_qrels_file, which \n\
consists of text tuples of the form \n\
   qid  iter  docno  rel \n\
giving TREC document numbers (docno, a string) and their relevance (rel,  \n\
a non-negative integer less than 128, or -1 (unjudged)) \n\
to query qid (a string).  iter string field is ignored.   \n\
Fields are separated by whitespace, string fields can contain no whitespace. \n\
File may contain no NULL characters. \n\
 \n\
The text tuples with relevance judgements are converted to TR_VEC form \n\
and then submitted to the SMART evaluation routines. \n\
The did,rank,sim fields of TR_VEC are filled in from trec_top_file; \n\
action,iter fields are set to 0. \n\
The rel field is set to -1 if the document was not in the pool (not in \n\
text_qrels_file) or -2 if the document was in the pool but unjudged (some \n\
measures (infAP) allow the pool to be sampled instead of judged fully).  \n\
Otherwise it is set to the value in text_qrels_file. \n\
Most measures, but not all, will treat -1 or -2 the same as 0, \n\
namely nonrelevant.  Note that relevance_level is used to \n\
determine if the document is relevant during score calculations. \n\
Queries for which there is no relevance information are ignored. \n\
Warning: queries for which there are relevant docs but no retrieved docs \n\
are also ignored by default.  This allows systems to evaluate over subsets  \n\
of the relevant docs, but means if a system improperly retrieves no docs,  \n\
it will not be detected.  Use the -c flag to avoid this behavior. \n\
 \n\
EXPLANATION OF OFFICIAL VALUES PRINTED OF OLD NON-RELATIONAL FORMAT. \n\
Relational Format prints the same values, but all lines are of the form \n\
    measure_name   query   value \n\
 \n\
1. Total number of documents over all queries \n\
        Retrieved: \n\
        Relevant: \n\
        Rel_ret:     (relevant and retrieved) \n\
   These should be self-explanatory.  All values are totals over all \n\
   queries being evaluated. \n\
2. Interpolated Recall - Precision Averages: \n\
        at 0.00 \n\
        at 0.10 \n\
        ... \n\
        at 1.00 \n\
   See any standard IR text (especially by Salton) for more details of  \n\
   recall-precision evaluation.  Measures precision (percent of retrieved \n\
   docs that are relevant) at various recall levels (after a certain \n\
   percentage of all the relevant docs for that query have been retrieved). \n\
   'Interpolated' means that, for example, precision at recall \n\
   0.10 (ie, after 10% of rel docs for a query have been retrieved) is \n\
   taken to be MAXIMUM of precision at all recall points >= 0.10. \n\
   Values are averaged over all queries (for each of the 11 recall levels). \n\
   These values are used for Recall-Precision graphs. \n\
3. Average precision (non-interpolated) over all rel docs \n\
   The precision is calculated after each relevant doc is retrieved. \n\
   If a relevant doc is not retrieved, its precision is 0.0. \n\
   All precision values are then averaged together to get a single number \n\
   for the performance of a query.  Conceptually this is the area \n\
   underneath the recall-precision graph for the query. \n\
   The values are then averaged over all queries. \n\
4. Precision: \n\
       at 5    docs \n\
       at 10   docs \n\
       ... \n\
       at 1000 docs    \n\
   The precision (percent of retrieved docs that are relevant) after X \n\
   documents (whether relevant or nonrelevant) have been retrieved. \n\
   Values averaged over all queries.  If X docs were not retrieved \n\
   for a query, then all missing docs are assumed to be non-relevant. \n\
5. R-Precision (precision after R (= num_rel for a query) docs retrieved): \n\
   Measures precision (or recall, they're the same) after R docs \n\
   have been retrieved, where R is the total number of relevant docs \n\
   for a query.  Thus if a query has 40 relevant docs, then precision \n\
   is measured after 40 docs, while if it has 600 relevant docs, precision \n\
   is measured after 600 docs.  This avoids some of the averaging \n\
   problems of the 'precision at X docs' values in (4) above. \n\
   If R is greater than the number of docs retrieved for a query, then \n\
   the nonretrieved docs are all assumed to be nonrelevant. \n\
";


extern SINGLE_MEASURE sing_meas[];
extern PARAMETERIZED_MEASURE param_meas[];
extern MICRO_MEASURE micro_meas[];
extern int num_param_meas, num_sing_meas, num_micro_meas;

int
trec_eval_help(epi)
EVAL_PARAM_INFO *epi;
{
    long i, j;
    /* Note this trusts the format_strings  in measures.c will not overflow */
    char temp_buf1[200];
    char temp_buf2[200];

    printf ("%s\n", help_message);

    printf ("Major measures (again) with their relational names:\n");
    for (i = 0; i < num_sing_meas; i++) {
	if (sing_meas[i].print_short_flag)
	    printf ("%-15s\t%s\n", sing_meas[i].name, sing_meas[i].long_name);
    }
    for (i = 0; i < num_param_meas; i++) {
	if (param_meas[i].print_short_flag) {
	    for (j = 0; j < param_meas[i].num_values; j++) {
		sprintf (temp_buf1, param_meas[i].format_string,
			 param_meas[i].get_param_str (epi, j));
		sprintf (temp_buf2, param_meas[i].long_format_string,
			 param_meas[i].get_param_str (epi, j));
		printf ("%-15s\t%s%s\n", temp_buf1,
			param_meas[i].long_name, temp_buf2);
	    }
	}
    }
    for (i = 0; i < num_micro_meas; i++) {
	if (micro_meas[i].print_short_flag)
	    printf ("%-15s\t%s\n", micro_meas[i].name, micro_meas[i].long_name);
    }

    printf ("\n\nMinor measures with their relational names:\n");
    for (i = 0; i < num_sing_meas; i++) {
	if (sing_meas[i].print_short_flag)
	    continue;
	if (sing_meas[i].print_time_flag && (! epi->time_flag))
	    continue;
	if (! sing_meas[i].print_short_flag)
	    printf ("%-15s\t%s\n", sing_meas[i].name, sing_meas[i].long_name);
    }
    for (i = 0; i < num_param_meas; i++) {
	if (param_meas[i].print_short_flag)
	    continue;
	if (param_meas[i].print_time_flag && (! epi->time_flag))
	    continue;
	for (j = 0; j < param_meas[i].num_values; j++) {
	    sprintf (temp_buf1, param_meas[i].format_string,
		     param_meas[i].get_param_str (epi, j));
	    sprintf (temp_buf2, param_meas[i].long_format_string,
		     param_meas[i].get_param_str (epi, j));
	    printf ("%-15s\t%s%s\n", temp_buf1,
		    param_meas[i].long_name, temp_buf2);
	}
    }
    for (i = 0; i < num_micro_meas; i++) {
	if (! micro_meas[i].print_short_flag)
	    printf ("%-15s\t%s\n", micro_meas[i].name, micro_meas[i].long_name);
    }

    return (1);
}

