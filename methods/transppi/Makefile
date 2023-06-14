PY = python3

FOLD ?= 0
PPI_NAME ?= Pans

EPOCHS ?= 1

OUT_PREPROCESSING=out/preprocessing
COORD=$(OUT_PREPROCESSING)/coord.hdf5
PROTTRANS=$(OUT_PREPROCESSING)/prottrans_normed.hdf5

OUT_OP = out
OUT = $(OUT_OP)/$(PPI_NAME)/$(FOLD)
OUT_CHECKPOINT=$(OUT)/checkpoint
OUT_PREDICTION=$(OUT)/predictions
OUT_ANALYZING=$(OUT)/analyzing

PPI=$(OUT_PREPROCESSING)/ppi/$(PPI_NAME)
PPI_train=$(PPI)/train_$(FOLD).json
PPI_test=$(PPI)/test_$(FOLD).json


TRAIN = $(PY) train.py --ppi $(PPI_train) --coord $(COORD) --prottrans $(PROTTRANS) --output $(OUT_CHECKPOINT) --epochs $(EPOCHS)
LAST_CHECKPOINT = $(lastword $(sort $(wildcard $(OUT_CHECKPOINT)/checkpoint_*.pt)))

.PHONY: train resume analyze analyze-train analyze-test

ifeq ($(LAST_CHECKPOINT), )
all: train analyze
else
all: resume analyze
endif

# all:	
# 	echo $(CHECKPOINTS)
# 	echo $(PREDICTION_train)

# Use train and resume to generate a series of checkpoints.
CHECKPOINTS = $(wildcard $(OUT_CHECKPOINT)/checkpoint_*.pt)
train: train.py model_config.json train_config.json $(PPI_train)
	mkdir -p $(OUT_CHECKPOINT)
	$(TRAIN)

resume: train.py model_config.json train_config.json $(PPI_train) $(LAST_CHECKPOINT)
	$(TRAIN) --checkpoint $(LAST_CHECKPOINT)

# $(OUT_CHECKPOINT)/checkpoint_%.pt : train.py model_config.json train_config.json $(PPI_train)
# 	mkdir -p $(OUT_CHECKPOINT)
# 	$(TRAIN) --checkpoint $(LAST_CHECKPOINT)

# Use checkpoint to generate prediction of the model on the given ppi datatset.
analyze: analyze-test analyze-train

PREDICTION_template = $(OUT_PREDICTION)/$(1)/$(notdir $(basename $(2))).pred

PREDICTION_train = $(foreach ckpt,$(CHECKPOINTS),$(call PREDICTION_template,train,$(ckpt)))
PREDICTION_test = $(foreach ckpt,$(CHECKPOINTS),$(call PREDICTION_template,test,$(ckpt)))
PREDICTION = $(PREDICTION_train) $(PREDICTION_test)

TEST = $(PY) test.py --coord $(COORD) --prottrans $(PROTTRANS)
define PREDICTION_rule_template=
$(call PREDICTION_template,$(1),$(2)) : test.py $(PPI_$(1)) $(2)
	$(TEST) --ppi $$(word 2,$$^) --checkpoint $$(word 3,$$^) --output $$@
endef

$(foreach ppi,test train,$(foreach ckpt,$(CHECKPOINTS),$(eval $(call PREDICTION_rule_template,$(ppi),$(ckpt)))))


# Generate analysis draws according to PREDICTIONS
define analyze_template = 
analyze-$(1) : $$(PREDICTION_$(1))
	echo $$^
	mkdir -p $(OUT_ANALYZING)/$(1)
	make -C analyzing IN=$(abspath $(OUT_PREDICTION)/$(1)) OUT=$(abspath $(OUT_ANALYZING)/$(1))
endef
$(foreach ppi,test train,$(eval $(call analyze_template,$(ppi))))
# .SECONDEXPANSION:
# analyze-test analyze-train : analyze-% : $$(PREDICTION_%)
# 	echo $(PREDICTION_test)
# 	echo $(PREDICTION)
# 	echo $^
# 	mkdir -p $(OUT_ANALYZING)/$*
# 	make -C analyzing IN=$(abspath $(OUT_PREDICTION)/$*) OUT=$(abspath $(OUT_ANALYZING)/$*)

clean:
	rm -rI $(OUT)
