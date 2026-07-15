CC = gcc
CFLAGS = -O3 -Wall -std=c11 -fopenmp -g
INCLUDES = -Iinclude
TARGET = cascade_filter
SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
# Exclude dsp_engine from cascade_filter target
MAIN_SRCS = $(filter-out $(SRC_DIR)/dsp_engine.c, $(SRCS))
MAIN_OBJS = $(MAIN_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

DSP_SRCS = $(filter-out $(SRC_DIR)/main.c, $(SRCS))
DSP_OBJS = $(DSP_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: cascade_filter dsp_engine

cascade_filter: $(MAIN_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

dsp_engine: $(DSP_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) cascade_filter dsp_engine

.PHONY: all clean
