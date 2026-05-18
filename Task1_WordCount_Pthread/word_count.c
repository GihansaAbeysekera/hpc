#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <ctype.h>

#define MAX_WORD_LEN 100
#define CHUNK_SIZE 100

typedef struct {
    char word[MAX_WORD_LEN];
    int count;
} WordCount;

char **words = NULL;
int totalWords = 0;

WordCount *globalCounts = NULL;
int globalSize = 0;
int globalCapacity = 1000;

int nextIndex = 0;

pthread_mutex_t indexMutex;
pthread_mutex_t globalMutex;

void cleanWord(char *word) {
    int i, j = 0;
    char temp[MAX_WORD_LEN];

    for (i = 0; word[i] != '\0'; i++) {
        if (isalpha(word[i]) || isdigit(word[i])) {
            temp[j++] = tolower(word[i]);
        }
    }

    temp[j] = '\0';
    strcpy(word, temp);
}

void addToLocal(WordCount **localCounts, int *localSize, int *localCapacity, char *word) {
    for (int i = 0; i < *localSize; i++) {
        if (strcmp((*localCounts)[i].word, word) == 0) {
            (*localCounts)[i].count++;
            return;
        }
    }

    if (*localSize >= *localCapacity) {
        *localCapacity *= 2;
        *localCounts = realloc(*localCounts, (*localCapacity) * sizeof(WordCount));
    }

    strcpy((*localCounts)[*localSize].word, word);
    (*localCounts)[*localSize].count = 1;
    (*localSize)++;
}

void addToGlobal(char *word, int count) {
    for (int i = 0; i < globalSize; i++) {
        if (strcmp(globalCounts[i].word, word) == 0) {
            globalCounts[i].count += count;
            return;
        }
    }

    if (globalSize >= globalCapacity) {
        globalCapacity *= 2;
        globalCounts = realloc(globalCounts, globalCapacity * sizeof(WordCount));
    }

    strcpy(globalCounts[globalSize].word, word);
    globalCounts[globalSize].count = count;
    globalSize++;
}

void *processWords(void *arg) {
    WordCount *localCounts = malloc(1000 * sizeof(WordCount));
    int localSize = 0;
    int localCapacity = 1000;

    while (1) {
        pthread_mutex_lock(&indexMutex);

        int start = nextIndex;
        int end = start + CHUNK_SIZE;
        nextIndex = end;

        pthread_mutex_unlock(&indexMutex);

        if (start >= totalWords) {
            break;
        }

        if (end > totalWords) {
            end = totalWords;
        }

        for (int i = start; i < end; i++) {
            addToLocal(&localCounts, &localSize, &localCapacity, words[i]);
        }
    }

    pthread_mutex_lock(&globalMutex);

    for (int i = 0; i < localSize; i++) {
        addToGlobal(localCounts[i].word, localCounts[i].count);
    }

    pthread_mutex_unlock(&globalMutex);

    free(localCounts);
    return NULL;
}

int compareWords(const void *a, const void *b) {
    WordCount *w1 = (WordCount *)a;
    WordCount *w2 = (WordCount *)b;
    return strcmp(w1->word, w2->word);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <filename> <number_of_threads>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int threadCount = atoi(argv[2]);

    if (threadCount <= 0) {
        printf("Error: Number of threads must be greater than 0.\n");
        return 1;
    }

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error: Cannot open input file.\n");
        return 1;
    }

    int wordCapacity = 1000;
    words = malloc(wordCapacity * sizeof(char *));

    char buffer[MAX_WORD_LEN];

    while (fscanf(file, "%99s", buffer) == 1) {
        cleanWord(buffer);

        if (strlen(buffer) == 0) {
            continue;
        }

        if (totalWords >= wordCapacity) {
            wordCapacity *= 2;
            words = realloc(words, wordCapacity * sizeof(char *));
        }

        words[totalWords] = malloc(MAX_WORD_LEN * sizeof(char));
        strcpy(words[totalWords], buffer);
        totalWords++;
    }

    fclose(file);

    globalCounts = malloc(globalCapacity * sizeof(WordCount));

    pthread_t *threads = malloc(threadCount * sizeof(pthread_t));

    pthread_mutex_init(&indexMutex, NULL);
    pthread_mutex_init(&globalMutex, NULL);

    for (int i = 0; i < threadCount; i++) {
        pthread_create(&threads[i], NULL, processWords, NULL);
    }

    for (int i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
    }

    qsort(globalCounts, globalSize, sizeof(WordCount), compareWords);

    FILE *output = fopen("result.txt", "w");

    if (output == NULL) {
        printf("Error: Cannot create result.txt\n");
        return 1;
    }

    fprintf(output, "Word Occurrence Results\n");
    fprintf(output, "=======================\n\n");

    for (int i = 0; i < globalSize; i++) {
        fprintf(output, "%s : %d\n", globalCounts[i].word, globalCounts[i].count);
    }

    fclose(output);

    for (int i = 0; i < totalWords; i++) {
        free(words[i]);
    }

    free(words);
    free(globalCounts);
    free(threads);

    pthread_mutex_destroy(&indexMutex);
    pthread_mutex_destroy(&globalMutex);

    printf("Word counting completed. Results saved in result.txt\n");

    return 0;
}
