BIN = /home/smart/bin
H   = .

VERSIONID = 8.1

# gcc
CC       = gcc
CFLAGS   = -g -I$H -O3 -Wall -DVERSIONID=\"$(VERSIONID)\"
CFLAGS   = -g -I$H  -Wall -DVERSIONID=\"$(VERSIONID)\"

# cc
###CC       = cc
###CFLAGS   = -I$H -g -DVERSIONID=\"$(VERSIONID)\"

# Other macros used in some or all makefiles
INSTALL = /bin/mv


OBJS = trec_eval.o get_qrels.o get_top.o form_trvec.o measures.o print_meas.o\
	trvec_teval.o buf_util.o error_msgs.o \
       trec_eval_help.o


SRCS = trec_eval.c get_qrels.c get_top.c form_trvec.c measures.c print_meas.c\
       trvec_teval.c buf_util.c error_msgs.c \
       trec_eval_help.c

SRCH = common.h trec_eval.h smart_error.h sysfunc.h tr_vec.h buf.h

SRCOTHER = README Makefile test bpref_bug Changelog

trec_eval: $(SRCS) Makefile $(SRCH)
	$(CC) $(CFLAGS)  -o trec_eval $(SRCS) -lm

install: $(BIN)/trec_eval

quicktest: trec_eval
	./trec_eval test/qrels.test test/results.test | diff - test/out.test
	./trec_eval -a test/qrels.test test/results.test | diff - test/out.test.a
	./trec_eval -a -q test/qrels.test test/results.test | diff - test/out.test.aq
	./trec_eval -a -q -c test/qrels.test test/results.trunc | diff - test/out.test.aqc
	./trec_eval -a -q -c -M100 test/qrels.test test/results.trunc | diff - test/out.test.aqcM
	./trec_eval -a -q -l2 test/qrels.rel_level test/results.test | diff - test/out.test.aql
	/bin/echo "Test succeeeded"

longtest: trec_eval
	/bin/rm -rf test.long; mkdir test.long
	./trec_eval test/qrels.test test/results.test > test.long/out.test
	./trec_eval -a test/qrels.test test/results.test > test.long/out.test.a
	./trec_eval -a -q test/qrels.test test/results.test > test.long/out.test.aq
	./trec_eval -a -q -c test/qrels.test test/results.trunc > test.long/out.test.aqc
	./trec_eval -a -q -c -M100 test/qrels.test test/results.trunc > test.long/out.test.aqcM
	./trec_eval -a -q -l2 test/qrels.rel_level test/results.test > test.long/out.test.aql
	diff test.long test

$(BIN)/trec_eval: trec_eval
	if [ -f $@ ]; then $(INSTALL) $@ $@.old; fi;
	$(INSTALL) trec_eval $@

##4##########################################################################
##5##########################################################################
#  All code below this line (except for automatically created dependencies)
#  is independent of this particular makefile, and should not be changed!
#############################################################################

#########################################################################
# Odds and ends                                                         #
#########################################################################
clean semiclean:
	/bin/rm -f *.o *.BAK *~ trec_eval trec_eval.*.tar out.trec_eval Makefile.bak

tar:
	-/bin/rm -rf ./trec_eval.$(VERSIONID)
	mkdir trec_eval.$(VERSIONID)
	cp -rp $(SRCOTHER) $(SRCS) $(SRCH) trec_eval.$(VERSIONID)
	tar cf - ./trec_eval.$(VERSIONID) > trec_eval.$(VERSIONID).tar

lint:
	lint $(SRCS)

#########################################################################
# Determining program dependencies                                      #
#########################################################################
depend:
	grep '^#[ ]*include' *.c \
		| sed -e 's?:[^"]*"\([^"]*\)".*?: \$H/\1?' \
			-e '/</d' \
			-e '/functions.h/d' \
		        -e 's/\.c/.o/' \
		        -e 's/\.y/.o/' \
		        -e 's/\.l/.o/' \
		> makedep
	echo '/^# DO NOT DELETE THIS LINE/+2,$$d' >eddep
	echo '$$r makedep' >>eddep
	echo 'w' >>eddep
	cp Makefile Makefile.bak
	ed - Makefile < eddep
	/bin/rm eddep makedep
	echo '# DEPENDENCIES MUST END AT END OF FILE' >> Makefile
	echo '# IF YOU PUT STUFF HERE IT WILL GO AWAY' >> Makefile
	echo '# see make depend above' >> Makefile

# DO NOT DELETE THIS LINE -- make depend uses it

buf_util.o: ./common.h
buf_util.o: ./sysfunc.h
buf_util.o: ./buf.h
error_msgs.o: ./smart_error.h
error_msgs.o: ./sysfunc.h
form_trvec.o: ./common.h
form_trvec.o: ./sysfunc.h
form_trvec.o: ./smart_error.h
form_trvec.o: ./tr_vec.h
form_trvec.o: ./trec_eval.h
form_trvec.o: ./buf.h
get_qrels.o: ./common.h
get_qrels.o: ./sysfunc.h
get_qrels.o: ./smart_error.h
get_qrels.o: ./trec_eval.h
get_top.o: ./common.h
get_top.o: ./sysfunc.h
get_top.o: ./smart_error.h
get_top.o: ./trec_eval.h
measures.o: ./common.h
measures.o: ./sysfunc.h
measures.o: ./buf.h
measures.o: ./trec_eval.h
print_meas.o: ./common.h
print_meas.o: ./sysfunc.h
print_meas.o: ./buf.h
print_meas.o: ./trec_eval.h
trec_eval.o: ./common.h
trec_eval.o: ./sysfunc.h
trec_eval.o: ./smart_error.h
trec_eval.o: ./tr_vec.h
trec_eval.o: ./trec_eval.h
trec_eval.o: ./buf.h
trec_eval_help.o: ./common.h
trvec_teval.o: ./common.h
trvec_teval.o: ./sysfunc.h
trvec_teval.o: ./smart_error.h
trvec_teval.o: ./tr_vec.h
trvec_teval.o: ./trec_eval.h
# DEPENDENCIES MUST END AT END OF FILE
# IF YOU PUT STUFF HERE IT WILL GO AWAY
# see make depend above
