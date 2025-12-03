# Revisor de Carpeta – Física Uniandes

Aplicación en Streamlit que revisa automáticamente la “carpeta” de un estudiante de Física: carga el historial de cursos, valida requisitos específicos y genera reportes descargables.

## Requisitos
- Python 3.9+
- Dependencias: `streamlit`, `pandas`, `numpy`, `openpyxl`

Instala con:
```bash
pip install -r requirements.txt  # si tienes este archivo
# o directo
pip install streamlit pandas numpy openpyxl
```

## Cómo ejecutar
```bash
streamlit run revisor_carpeta_app.py
```
La app se abre en el navegador. En la barra lateral sube:
- **Excel/CSV del estudiante**: columnas mínimas `Materia/examen`, `Créditos`, `Nota final`; opcional `Atributos sección` y `Nombre largo curso o examen`.
- **Tabla agrupada**: columnas `Sección` y `Materia/examen` (puede venir como lista o string separado por comas).
- Ajusta el JSON con mínimos de créditos por sección y los switches para CBUS y ECUR/EPSI.

## Lógica principal
1) **Normalización y mapeo**: limpia nombres de cursos, convierte notas (`A` y numéricas), y construye un mapa Curso → Sección desde la tabla agrupada.
2) **Mejor intento por curso**: ordena por aprobado y nota, y se queda con la mejor versión de cada `Materia/examen`.
3) **Créditos por sección**: suma créditos aprobados por sección y calcula faltantes vs. los mínimos definidos.
4) **Reglas específicas**:
   - **Educación General**: exige ESCR I/II (por nombre largo), `DERE-1300`, 2×ECUR y 1×EPSI (según `Atributos sección`).
   - **CBUS**: pide `CBCC-1177`, mínimo 1 de cada tipo (CBCA/CBPC/CBCO) y total de CBUS ≥6 si está activado.
   - **CLE**: suma créditos válidos desde DEPO de 1 crédito, cursos sin sección (no CBUS/DEPO), excedente de Adv. Science no CBUS, y EduGen de 3 créditos que no sean obligatorios ni con atributos especiales; compara contra la meta CLE.
   - **ECAP/RLEC/RDOM**: valida notas especiales (FISI para ECAP, 1 para RLEC/RDOM).
5) **Asignación de cursos**: marca qué cursos se usan para cada requisito/sección y cuáles quedan aprobados pero no contados. Añade columna `Sección original` para rastrear el origen.

## Salidas en la UI
- Métricas rápidas: CLE, Educación General, CBUS, ECAP/RLEC/RDOM.
- Tabs:
  - **Créditos por sección**: aprobados, mínimos y faltantes.
  - **Detalles evaluados**: columnas clave del estudiante con “Motivo uso” y “Sección original”.
  - **Checks específicos**: JSON con el estado de cada regla.
  - **Descargas**: CSV del resumen y XLSX con detalle, resumen y cursos no usados.
  - **Cursos no usados**: aprobados que no contaron para el cómputo.

## Notas de datos
- Las notas “A” se consideran aprobadas (equivalente a 5.0 para ordenar).
- Si faltan columnas requeridas se muestra un error en la app.
- El archivo `Estudiantes/Requisitos de grado.xlsx` contiene la tabla de mínimos usada por defecto. Añade tus propios mínimos editando el JSON en la barra lateral.
