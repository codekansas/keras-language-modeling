trec_eval is the standard tool used by the TREC community for
evaluating an ad hoc retrieval run, given the results file and a
standard set of judged results.  

------------------------------------------------------------------------------
Installation: Should be as easy as typing "make" in the source directory,
if gcc is available.  Otherwise, comment out the gcc lines (lines 5-6) and
uncomment out the cc lines (lines 9-10)
If you wish the trec_eval binary to be placed in a standard location, alter
the first line of Makefile appropriately.

------------------------------------------------------------------------------
Testing: sample input and output files are included in the directory test.
"make quicktest" will perform some sample simple evaluations and compare
the results.

------------------------------------------------------------------------------
Usage:  Most options can be ignored.  The only one most folks will need
is the "-q" flag, to indicate whether to output results for individual 
queries as well as the averages over all queries.  Official TREC usage
might be something like 
	trec_eval -q -c -M1000 official_qrels submitted_results 
to ensure correct evaluation if submitted_results doesn't have results
for all queries, or returns more than 1000 documents per query.


------------------------------------------------------------------------------
Change Log  (only recent)
------------------------------------------------------------------------------
Version 8.1, Added infAP, minor bug fixes
7/24/06 Improved infAP comments (implementation verified by Yilmaz).
        trec_eval_help.c: allow longer measure explanations.
6/27/06 get_opt.c Fixed error message
6/22/06 Added measure infAP (Aslam et al) to allow judging only sample 
        of pools.  -1 for rel in qrels file interpreted as pool doc not judged.
6/22/06 trvec_teval.c: fixed bugs in calculation of bpref if multiple
	relevance levels were used and a non-default relevance level
	was given. (Eg. A doc with rel level of 2 was counted as unjudged
	rather than judged nonrel if a relevance level of 3 was needed
	to consider relevant.)
4/5/06  Changed comments in README, trec_eval.c, trec_eval_help.c files 
        which incorrectly claimed queries with no relevant docs are 
        ignored (this was true with very old versions of trec_eval).  Now
        reads that queries with no relevance information are ignored.
        Giorgio Di Nunzio and Nicola Ferro,
------------------------------------------------------------------------------
Version 8.0, full bpref bug fix, see file bpref_bug.  I decided to up the 
        version number since bpref results are incompatible with previous 
        results (though the changes are small).
------------------------------------------------------------------------------
------------------------------------------------------------------------------

Files:
Makefile     Compile and test trec_eval
README       This file
test         Collection of sample input and output for trec_eval
trec_eval.c  Main procedure
get_qrels.c  Called by main to read the standard judged documents (qrels)
get_top.c    Called by main to read the results file to be evaluated
form_trvec.c Called by main to put the results and qrels for an individual
	     query in the proper format to be evaluated.
trvec_teval.c Called by main to evaluate an individual query
print_meas.c Called by main to print an evaluated query, and to accumulate
	     the results for later averaging over the queries.
measures.c   Description of the measures used by printing.
trec_eval_help.c Descriptions of trec_eval, the output, and the measures.

trec_eval.h  Basic evaluation structures.
bpref_bug:   Description of bug in bpref that existed in trec_eval versions 6
             through 7.3.

The rest of the files are small utility portions from SMART.
tr_vec.h
smart_error.h
sysfunc.h
buf.h
common.h
buf_util.c
error_msgs.c

------------------------------------------------------------------------------

The rest of this file consists of information printed by "trec_eval -h":
(If you REALLY want a complete list of measures calculated, you can add the
time based measures and run "trec_eval -T -h".)

trec_eval [-h] [-q] [-a] [-o] [-c] [-l<num>  [-N<num>] [-M<num>] [-Ua<num>] [-Ub<num>] [-Uc<num>] [-Ud<num>] [-T] trec_rel_file trec_top_file 
 
Calculate and print various evaluation measures, evaluating the results  
in trec_top_file against the relevance judgements in trec_rel_file. 
 
There are a fair number of options, of which only the lower case options are 
normally ever used.   
 -h: Print full help message and exit 
 -q: In addition to summary evaluation, give evaluation for each query 
 -a: Print all evaluation measures calculated, instead of just the 
     main official measures for TREC. 
 -o: Print everything out in old, nonrelational format (default is relational) 
 -c: Average over the complete set of queries in the relevance judgements  
      instead of the queries in the intersection of relevance judgements 
      and results.  Missing queries will contribute a value of 0 to all 
      evaluation measures (which may or may not be reasonable for a  
      particular evaluation measure, but is reasonable for standard TREC 
      measures.) 
 -l<num>: Num indicates the minimum relevance judgement value needed for 
      a document to be called relevant. (All measures used by TREC eval are 
      based on binary relevance).  Used if trec_rel_file contains relevance 
      judged on a multi-relevance scale.  Default is 1. 
 -N<num>: Number of docs in collection 
 -M<num>: Max number of docs per topic to use in evaluation (discard rest). 
 -Ua<num>: Value to use for 'a' coefficient of utility computation. 
                        relevant  nonrelevant 
      retrieved            a          b 
      nonretrieved         c          d 
 -Ub<num>: Value to use for 'b' coefficient of utility computation. 
 -Uc<num>: Value to use for 'c' coefficient of utility computation. 
 -Ud<num>: Value to use for 'd' coefficient of utility computation. 
 -J: Calculate all values only over the judged (either relevant or  
     nonrelevant) documents.  All unjudged documents are removed from the 
     retrieved set before any calculations (possibly leaving an empty set). 
     DO NOT USE, unless you really know what you're doing - very easy to get 
     reasonable looking, but invalid, numbers.  
 -T: Treat similarity as time that document retrieved.  Compute 
      several time-based measures after ranking docs by time retrieved 
      (first doc (lowest sim) retrieved ranked highest).  
      Only done if -a selected. 
 
 
Read text tuples from trec_top_file of the form 
     030  Q0  ZF08-175-870  0   4238   prise1 
     qid iter   docno      rank  sim   run_id 
giving TREC document numbers (a string) retrieved by query qid  
(a string) with similarity sim (a float).  The other fields are ignored, 
with the exception that the run_id field of the last line is kept and 
output.  In particular, note that the rank field is ignored here; 
internally ranks are assigned by sorting by the sim field with ties  
broken deterministicly (using docno). 
Sim is assumed to be higher for the docs to be retrieved first. 
File may contain no NULL characters. 
Lines may contain fields after the run_id; they are ignored. 
 
Relevance for each docno to qid is determined from text_qrels_file, which 
consists of text tuples of the form 
   qid  iter  docno  rel 
giving TREC document numbers (docno, a string) and their relevance (rel,  
a non-negative integer less than 128, or -1 (unjudged)) 
to query qid (a string).  iter string field is ignored.   
Fields are separated by whitespace, string fields can contain no whitespace. 
File may contain no NULL characters. 
 
The text tuples with relevance judgements are converted to TR_VEC form 
and then submitted to the SMART evaluation routines. 
The did,rank,sim fields of TR_VEC are filled in from trec_top_file; 
action,iter fields are set to 0. 
The rel field is set to -1 if the document was not in the pool (not in 
text_qrels_file) or -2 if the document was in the pool but unjudged (some 
measures (infAP) allow the pool to be sampled instead of judged fully).  
Otherwise it is set to the value in text_qrels_file. 
Most measures, but not all, will treat -1 or -2 the same as 0, 
namely nonrelevant.  Note that relevance_level is used to 
determine if the document is relevant during score calculations. 
Queries for which there is no relevance information are ignored. 
Warning: queries for which there are relevant docs but no retrieved docs 
are also ignored by default.  This allows systems to evaluate over subsets  
of the relevant docs, but means if a system improperly retrieves no docs,  
it will not be detected.  Use the -c flag to avoid this behavior. 
 
EXPLANATION OF OFFICIAL VALUES PRINTED OF OLD NON-RELATIONAL FORMAT. 
Relational Format prints the same values, but all lines are of the form 
    measure_name   query   value 
 
1. Total number of documents over all queries 
        Retrieved: 
        Relevant: 
        Rel_ret:     (relevant and retrieved) 
   These should be self-explanatory.  All values are totals over all 
   queries being evaluated. 
2. Interpolated Recall - Precision Averages: 
        at 0.00 
        at 0.10 
        ... 
        at 1.00 
   See any standard IR text (especially by Salton) for more details of  
   recall-precision evaluation.  Measures precision (percent of retrieved 
   docs that are relevant) at various recall levels (after a certain 
   percentage of all the relevant docs for that query have been retrieved). 
   'Interpolated' means that, for example, precision at recall 
   0.10 (ie, after 10% of rel docs for a query have been retrieved) is 
   taken to be MAXIMUM of precision at all recall points >= 0.10. 
   Values are averaged over all queries (for each of the 11 recall levels). 
   These values are used for Recall-Precision graphs. 
3. Average precision (non-interpolated) over all rel docs 
   The precision is calculated after each relevant doc is retrieved. 
   If a relevant doc is not retrieved, its precision is 0.0. 
   All precision values are then averaged together to get a single number 
   for the performance of a query.  Conceptually this is the area 
   underneath the recall-precision graph for the query. 
   The values are then averaged over all queries. 
4. Precision: 
       at 5    docs 
       at 10   docs 
       ... 
       at 1000 docs    
   The precision (percent of retrieved docs that are relevant) after X 
   documents (whether relevant or nonrelevant) have been retrieved. 
   Values averaged over all queries.  If X docs were not retrieved 
   for a query, then all missing docs are assumed to be non-relevant. 
5. R-Precision (precision after R (= num_rel for a query) docs retrieved): 
   Measures precision (or recall, they're the same) after R docs 
   have been retrieved, where R is the total number of relevant docs 
   for a query.  Thus if a query has 40 relevant docs, then precision 
   is measured after 40 docs, while if it has 600 relevant docs, precision 
   is measured after 600 docs.  This avoids some of the averaging 
   problems of the 'precision at X docs' values in (4) above. 
   If R is greater than the number of docs retrieved for a query, then 
   the nonretrieved docs are all assumed to be nonrelevant. 

Major measures (again) with their relational names:
num_ret        	Total number of documents retrieved over all queries
num_rel        	Total number of relevant documents over all queries
num_rel_ret    	Total number of relevant documents retrieved over all queries
map            	Mean Average Precision (MAP)
gm_ap          	Average Precision. Geometric Mean, q_score=log(MAX(map,.00001))
R-prec         	R-Precision (Precision after R (= num-rel for topic) documents retrieved)
bpref          	Binary Preference, top R judged nonrel
recip_rank     	Reciprical rank of top relevant document
ircl_prn.0.00  	Interpolated Recall - Precision Averages at 0.00 recall
ircl_prn.0.10  	Interpolated Recall - Precision Averages at 0.10 recall
ircl_prn.0.20  	Interpolated Recall - Precision Averages at 0.20 recall
ircl_prn.0.30  	Interpolated Recall - Precision Averages at 0.30 recall
ircl_prn.0.40  	Interpolated Recall - Precision Averages at 0.40 recall
ircl_prn.0.50  	Interpolated Recall - Precision Averages at 0.50 recall
ircl_prn.0.60  	Interpolated Recall - Precision Averages at 0.60 recall
ircl_prn.0.70  	Interpolated Recall - Precision Averages at 0.70 recall
ircl_prn.0.80  	Interpolated Recall - Precision Averages at 0.80 recall
ircl_prn.0.90  	Interpolated Recall - Precision Averages at 0.90 recall
ircl_prn.1.00  	Interpolated Recall - Precision Averages at 1.00 recall
P5             	Precision after 5 docs retrieved
P10            	Precision after 10 docs retrieved
P15            	Precision after 15 docs retrieved
P20            	Precision after 20 docs retrieved
P30            	Precision after 30 docs retrieved
P100           	Precision after 100 docs retrieved
P200           	Precision after 200 docs retrieved
P500           	Precision after 500 docs retrieved
P1000          	Precision after 1000 docs retrieved


Minor measures with their relational names:
num_nonrel_judged_ret	Total number of judged non-relevant documents retrieved over all queries
exact_prec     	Exact Precision over retrieved set
exact_recall   	Exact Recall over retrieved set
11-pt_avg      	Average over all 11 points of recall-precision graph
3-pt_avg       	Average over 3 points of recall-precision graph
avg_doc_prec   	Rel doc precision averaged over all relevant docs (NOT over topics)
exact_relative_prec	Exact relative precision
avg_relative_prec	Average relative precision
exact_unranked_avg_prec	Exact Unranked Average Precision
exact_relative_unranked_avg_prec	Exact Relative Unranked Average Precision
map_at_R       	Average Precision over first R docs retrieved
int_map        	Interpolated Mean Average Precision
exact_int_R_rcl_prec	Exact R-based-interpolated-Precision
int_map_at_R   	Average Interpolated Precision for first R docs retrieved
bpref_allnonrel	Binary Preference, all judged nonrel
bpref_retnonrel	Binary Preference, all retrieved judged nonrel
bpref_topnonrel	Binary Preference, top 100 judged nonrel
bpref_top5Rnonrel	Binary Preference, top 5R judged nonrel
bpref_top10Rnonrel	Binary Preference, top 10R judged nonrel
bpref_top10pRnonrel	Binary Preference, top 10 + R judged nonrel
bpref_top25pRnonrel	Binary Preference, top 25 + R judged nonrel
bpref_top50pRnonrel	Binary Preference, top 50 + R judged nonrel
bpref_top25p2Rnonrel	Binary Preference, top 25 + 2*R judged nonrel
bpref_retall   	Binary Preference, Only retrieved judged rel and nonrel
bpref_5        	Binary Preference, top 5 rel, top 5 nonrel
bpref_10       	Binary Preference, top 10 rel, top 10 nonrel
bpref_num_all  	Binary Preference, Number not retrieved before (all judged)
bpref_num_ret  	Binary Preference, Number retrieved after
bpref_num_correct	Binary Preference, Number correct preferences
bpref_num_possible	Binary Preference, Number possible correct_preferences
old_bpref      	Buggy Version 7.3. Binary Preference, top R judged nonrel
old_bpref_top10pRnonrel	Buggy Version 7.3. Binary Preference,top 10+R judged nonrel
infAP          	Inferred AP. Calculate AP using only a judged random sample of the pool, averaging in unpooled documents as nonrel.
gm_bpref       	Binary Preference, top R judged nonrel, Geometric Mean, q_score=log(MAX(bpref,.00001))
rank_first_rel 	Rank of top relevant document (0 if none)
recall5        	Recall after 5 docs retrieved
recall10       	Recall after 10 docs retrieved
recall15       	Recall after 15 docs retrieved
recall20       	Recall after 20 docs retrieved
recall30       	Recall after 30 docs retrieved
recall100      	Recall after 100 docs retrieved
recall200      	Recall after 200 docs retrieved
recall500      	Recall after 500 docs retrieved
recall1000     	Recall after 1000 docs retrieved
0.20R-prec     	R-based precision- precision after 0.20 * R docs retrieved
0.40R-prec     	R-based precision- precision after 0.40 * R docs retrieved
0.60R-prec     	R-based precision- precision after 0.60 * R docs retrieved
0.80R-prec     	R-based precision- precision after 0.80 * R docs retrieved
1.00R-prec     	R-based precision- precision after 1.00 * R docs retrieved
1.20R-prec     	R-based precision- precision after 1.20 * R docs retrieved
1.40R-prec     	R-based precision- precision after 1.40 * R docs retrieved
1.60R-prec     	R-based precision- precision after 1.60 * R docs retrieved
1.80R-prec     	R-based precision- precision after 1.80 * R docs retrieved
2.00R-prec     	R-based precision- precision after 2.00 * R docs retrieved
relative_prec5 	Relative precision after 5 docs retrieved
relative_prec10	Relative precision after 10 docs retrieved
relative_prec15	Relative precision after 15 docs retrieved
relative_prec20	Relative precision after 20 docs retrieved
relative_prec30	Relative precision after 30 docs retrieved
relative_prec100	Relative precision after 100 docs retrieved
relative_prec200	Relative precision after 200 docs retrieved
relative_prec500	Relative precision after 500 docs retrieved
relative_prec1000	Relative precision after 1000 docs retrieved
unranked_avg_prec5	Unranked Average Precision after 5 docs retrieved
unranked_avg_prec10	Unranked Average Precision after 10 docs retrieved
unranked_avg_prec15	Unranked Average Precision after 15 docs retrieved
unranked_avg_prec20	Unranked Average Precision after 20 docs retrieved
unranked_avg_prec30	Unranked Average Precision after 30 docs retrieved
unranked_avg_prec100	Unranked Average Precision after 100 docs retrieved
unranked_avg_prec200	Unranked Average Precision after 200 docs retrieved
unranked_avg_prec500	Unranked Average Precision after 500 docs retrieved
unranked_avg_prec1000	Unranked Average Precision after 1000 docs retrieved
relative_unranked_avg_prec5	Relative Unranked Average Precision after 5 docs retrieved
relative_unranked_avg_prec10	Relative Unranked Average Precision after 10 docs retrieved
relative_unranked_avg_prec15	Relative Unranked Average Precision after 15 docs retrieved
relative_unranked_avg_prec20	Relative Unranked Average Precision after 20 docs retrieved
relative_unranked_avg_prec30	Relative Unranked Average Precision after 30 docs retrieved
relative_unranked_avg_prec100	Relative Unranked Average Precision after 100 docs retrieved
relative_unranked_avg_prec200	Relative Unranked Average Precision after 200 docs retrieved
relative_unranked_avg_prec500	Relative Unranked Average Precision after 500 docs retrieved
relative_unranked_avg_prec1000	Relative Unranked Average Precision after 1000 docs retrieved
utility_1.0_-1.0_0.0_0.0	Utility (a,b,c,d) Coefficients 1.0_-1.0_0.0_0.0 
rcl_at_142_nonrel	Recall averaged at X nonrel docs    X= 142 
fallout_recall_0	Fallout - Recall Averages- recall after 0 nonrel docs retrieved
fallout_recall_14	Fallout - Recall Averages- recall after 14 nonrel docs retrieved
fallout_recall_28	Fallout - Recall Averages- recall after 28 nonrel docs retrieved
fallout_recall_42	Fallout - Recall Averages- recall after 42 nonrel docs retrieved
fallout_recall_56	Fallout - Recall Averages- recall after 56 nonrel docs retrieved
fallout_recall_71	Fallout - Recall Averages- recall after 71 nonrel docs retrieved
fallout_recall_85	Fallout - Recall Averages- recall after 85 nonrel docs retrieved
fallout_recall_99	Fallout - Recall Averages- recall after 99 nonrel docs retrieved
fallout_recall_113	Fallout - Recall Averages- recall after 113 nonrel docs retrieved
fallout_recall_127	Fallout - Recall Averages- recall after 127 nonrel docs retrieved
fallout_recall_142	Fallout - Recall Averages- recall after 142 nonrel docs retrieved
int_0.20R-prec 	Interpolated R-based precision, after 0.20 * R docs retrieved
int_0.40R-prec 	Interpolated R-based precision, after 0.40 * R docs retrieved
int_0.60R-prec 	Interpolated R-based precision, after 0.60 * R docs retrieved
int_0.80R-prec 	Interpolated R-based precision, after 0.80 * R docs retrieved
int_1.00R-prec 	Interpolated R-based precision, after 1.00 * R docs retrieved
int_1.20R-prec 	Interpolated R-based precision, after 1.20 * R docs retrieved
int_1.40R-prec 	Interpolated R-based precision, after 1.40 * R docs retrieved
int_1.60R-prec 	Interpolated R-based precision, after 1.60 * R docs retrieved
int_1.80R-prec 	Interpolated R-based precision, after 1.80 * R docs retrieved
int_2.00R-prec 	Interpolated R-based precision, after 2.00 * R docs retrieved
micro_prec     	Total relevant retrieved documents / Total retrieved documents
micro_recall   	Total relevant retrieved documents / Total relevant documents
micro_bpref    	Total correct preferences / Total possible preferences
