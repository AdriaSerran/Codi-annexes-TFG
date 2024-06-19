#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

char    *stripaTQDM(char *linea);

int main(int argc, char *argv[]) {
    FILE    *fpOut;
    char    linea[1000000];
    char    *mode = "wt";

    int   c, pos;

	extern int	optind;
    int     opcion;

	while ((opcion = getopt(argc, argv, "ha")) != -1) {
		switch (opcion) {
			case 'a' :	mode = strdup("at");
                        break;
			case 'h' :	
			case '?' :	fprintf(stderr, "Empleo: %s [-h] [-a] FILE\n", argv[0]);
                        return -1;
                        break;
        }
    }

    argv += optind;
    argc -= optind;

    if ((fpOut = fopen(*argv, mode)) == NULL) {
        fprintf(stderr, "Error al abrir el fichero %s (%s)\n", *argv, strerror(errno));
        return -1;
    }

    pos = 0;
    do {
        while ((c = getchar()) != EOF) {
            putchar(c);
            fflush(stdout);

            linea[pos++] = c;
            if (c == '\n') break;
        }

        linea[pos] = '\0';
        pos = 0;

        fprintf(fpOut, "%s", stripaTQDM(linea));
        fflush(fpOut);
    } while (c != EOF);

    fclose(fpOut);

    return 0;
}

char    *stripaTQDM(char *linea) {
	char	*blank = "\x1b""2dK", *inicio = "\r";
    char    *strip;

    while ((strip = strstr(linea, blank)) != NULL) {
        memmove(linea, strip + strlen(blank), strlen(strip + strlen(blank)) + 1);
    }
    while ((strip = strstr(linea, inicio)) != NULL) {
        memmove(linea, strip + strlen(inicio), strlen(strip + strlen(inicio)) + 1);
    }

    return linea;
}
