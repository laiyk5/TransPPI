PY=python3
IN ?= ../out/Pans/0/predictions/test
OUT ?= out2

SHELL = /bin/bash

PREDICTIONS := $(wildcard $(IN)/*.pred)

.PHONY: all clean

all:

clean:
	rm -rf $(OUT)/*

DRAW_NAME = dist_err \
						dist_pred \
						prc \
						roc 

DRAW_template = $(OUT)/$(basename $(notdir $(1)))/$(2).png
DRAW = $(foreach pred,$(PREDICTIONS),$(foreach draw,$(DRAW_NAME),$(call DRAW_template,$(pred),$(draw))))
define DRAW_rule_template = 
$(DRAW_template) : draw_$(2).py $(1)
	mkdir -p $$(@D)
	$(PY) $$< $$(word 2,$$^) $$@
endef
$(foreach pred,$(PREDICTIONS),$(foreach draw,$(DRAW_NAME),$(eval $(call DRAW_rule_template,$(pred),$(draw)))))


split_ = $(subst _, ,$(1))
METRIC_SOURCE = auprc \
								auroc \
								err \
								f1_pre_re
METRIC_NAME = $(call split_,$(METRIC_SOURCE))
METRIC_template = $(foreach metric,$(call split_,$(2)),$(OUT)/$(basename $(notdir $(1)))/$(metric).txt)
METRIC = $(foreach pred,$(PREDICTIONS),$(foreach source,$(METRIC_SOURCE),$(call METRIC_template,$(pred),$(source))))
define METRIC_rule_template =
$(call METRIC_template,$(1),$(2)) &: metric_$(2).py $(1)
	mkdir -p $$(@D)
	i=1; \
	for metric in $(call split_,$(2)); do \
		echo $$$$i; \
		sed -n -e $$$$i'p' <($(PY) $$< $$(word 2,$$^)) > $(call METRIC_template,$(1),$$$$metric); \
		i=$$$$(echo $$$$i+1 | bc); \
	done; 
endef
$(foreach pred,$(PREDICTIONS),$(foreach source,$(METRIC_SOURCE),$(eval $(call METRIC_rule_template,$(pred),$(source)))))

TREND = $(addprefix $(OUT)/,$(addsuffix .png,$(METRIC_NAME)))
define TREND_template =
$(OUT)/$(1).png : draw_trend.py $(wildcard $(OUT)/*/$(1).txt)
	mkdir -p $$(@D)
	$(PY) $$< <(cat $(OUT)/*/$(1).txt) $$@
endef
$(foreach metric,$(METRIC_NAME),$(eval $(call TREND_template,$(metric))))

all: $(METRIC) $(DRAW) $(TREND)
metric: $(METRIC)
draw: $(DRAW)
trend: $(TREND)
