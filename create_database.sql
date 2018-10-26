CREATE TABLE IF NOT EXISTS public.Database (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TYPE public.hand AS ENUM ('L', 'R');
CREATE TYPE public.gender AS ENUM ('M', 'F');

CREATE TABLE IF NOT EXISTS public.Writer (
    id SERIAL PRIMARY KEY,
    databaseId INTEGER REFERENCES public.Database(id),
    name TEXT,
    age integer,
    gender gender,
    hand hand
);

CREATE TABLE IF NOT EXISTS public.Formula (
    id SERIAL PRIMARY KEY,
    writerId INTEGER REFERENCES public.Writer(id),
    formula TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.TraceGroup (
    id SERIAL PRIMARY KEY,
    formulaId INTEGER REFERENCES public.Formula(id),
    truth TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.Trace (
    id SERIAL PRIMARY KEY,
    traceGroupId INTEGER REFERENCES public.TraceGroup(id),
    trace TEXT
);

CREATE TABLE IF NOT EXISTS public.Format (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.FormulaFormat (
    formulaId INTEGER REFERENCES public.Formula(id),
    formatId INTEGER REFERENCES public.Format(id),
    "order" INTEGER NOT NULL
);