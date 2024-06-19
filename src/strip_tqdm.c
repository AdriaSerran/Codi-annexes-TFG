#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int main(int argc, char *argv[]){
    FILE    *fpTqdm = stdin;
    char    linea[100000], *strip;
	char	*blank = "\x1b""2dK", *inicio = "\r";

	extern int	optind;
    int     opcion;

    int     help = 0;

	while ((opcion = getopt(argc, argv, "h")) != -1) {
		switch (opcion) {
			case 'h' :	
			case '?' :	help = 1;
                        break;
        }
    }

    argv += optind;
    argc -= optind;

    if (help) {
        fprintf(stderr, "Empleo: %s [-h] [fichero_tqdm]...\n", argv[0]);
        return -1;
    }

    if (*argv != NULL && (fpTqdm = fopen(*argv, "rt")) == NULL) {
        fprintf(stderr, "Error al abrir el fichero %s (%s)\n", *argv, strerror(errno));
        return -1;
    }

    do {
        char *fgets(char *s, int size, FILE *stream);
        while (fgets(linea, sizeof(linea), fpTqdm) != NULL) {
            while ((strip = strstr(linea, blank)) != NULL) {
                memmove(linea, strip + strlen(blank), strlen(strip + strlen(blank)) + 1);
            }
            while ((strip = strstr(linea, inicio)) != NULL) {
                memmove(linea, strip + strlen(inicio), strlen(strip + strlen(inicio)) + 1);
            }
            printf("%s", linea);
        }

        if (*argv && *++argv != NULL && (fpTqdm = fopen(*argv, "rt")) == NULL) {
            fprintf(stderr, "Error al abrir el fichero %s (%s)\n", *argv, strerror(errno));
            return -1;
        }
    } while (*argv);

    return 0;
}
