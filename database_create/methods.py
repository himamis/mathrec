import psycopg2

try:
    conn = psycopg2.connect("dbname=crohme")
except:
    print("Unable to connect to the database")
    exit()

cur = conn.cursor()


def _gender(ink):
    if ink.gender is None:
        return None
    l = ink.gender.lower()
    if l == 'm' or l == "male":
        return 'm'
    if l == 'f' or l == 'female':
        return 'f'

    return l


def _hand(ink):
    if ink.hand is None:
        return None
    l = ink.hand.lower()
    if l == 'l' or l == "left" or l == "left-handed":
        return 'l'
    if l == 'right' or l == 'r' or l == 'right-handed':
        return 'r'
    return l


def _age(ink):
    return ink.age

def maybe_add_database(inkml_dir):
    select_sql = "SELECT id FROM public.Database WHERE name=(%s);"
    cur.execute(select_sql, (inkml_dir,))
    id = cur.fetchone()
    if id is None:
        insert_sql = "INSERT INTO public.Database(name) VALUES(%s) RETURNING id;"
        cur.execute(insert_sql, (inkml_dir,))
        id = cur.fetchone()[0]
    else:
        id = id[0]
    return id


def maybe_add_writer(ink, database_id):
    sql = "INSERT INTO public.writer(name,databaseid,age,gender,hand) VALUES(%s,%s,%s,%s,%s) RETURNING id;"
    params = (ink.writer, database_id, _age(ink), _gender(ink), _hand(ink))
    if ink.writer is not None:
        select_sql = "SELECT id FROM public.Writer WHERE name=(%s) AND databaseid=(%s);"
        cur.execute(select_sql, (ink.writer, database_id))
        id = cur.fetchone()
        if id is None:
            cur.execute(sql, params)
            id = cur.fetchone()[0]
        else:
            id = id[0]
    else:
        cur.execute(sql, params)
        id = cur.fetchone()[0]
    return id


def add_formula(ink, writer_id):
    insert_sql = "INSERT INTO public.formula(formula,writerid) VALUES(%s,%s) RETURNING id;"
    cur.execute(insert_sql, (ink.truth, writer_id))
    id = cur.fetchone()[0]
    return id


def add_formula_formats(ink, formula_id):
    for index in range(len(ink.format)):
        format = ink.format[index]
        format_id = maybe_add_format(format)
        add_formula_format(formula_id, format_id, index)


def maybe_add_format(format):
    select_sql = "SELECT id FROM public.format WHERE name=(%s) AND type=(%s);"
    params = (format[0], format[1])
    cur.execute(select_sql, params)
    id = cur.fetchone()
    if id is None:
        insert_sql = "INSERT INTO public.format(name,type) VALUES(%s,%s) RETURNING id;"
        cur.execute(insert_sql, params)
        id = cur.fetchone()[0]
    else:
        id = id[0]
    return id


def add_formula_format(formula_id, format_id, index):
    sql = "INSERT INTO public.formulaformat(formulaid,formatid,\"order\") VALUES(%s,%s,%s);"
    cur.execute(sql, (formula_id, format_id, index+1))


def add_trace_groups(ink, formula_id):
    for trace_group in ink.trace_groups:
        add_trace_group(ink, trace_group, formula_id)


def add_trace_group(ink, trace_group, formula_id):
    sql = "INSERT INTO public.tracegroup(formulaid,truth) VALUES(%s,%s) RETURNING id;"
    cur.execute(sql, (formula_id, trace_group[0]))
    id = cur.fetchone()[0]
    add_traces(ink, trace_group, id)


def add_traces(ink, trace_group, trace_group_id):
    for trace in trace_group[1]:
        add_trace(ink, trace, trace_group_id)


def add_trace(ink, trace, trace_group_id):
    sql = "INSERT INTO public.trace(tracegroupid,trace) VALUES(%s,%s);"
    the_trace = ink.symbols.get(trace)
    if the_trace is not None:
        cur.execute(sql, (trace_group_id, ink.symbols[trace]))
    else:
        print("\nKey not found. Ignoring trace group..")


def add_inkml(ink, database_id):
    writer_id = maybe_add_writer(ink, database_id)
    formula_id = add_formula(ink, writer_id)
    add_formula_formats(ink, formula_id)
    add_trace_groups(ink, formula_id)


def clear_all_tables():
    for table in ['trace', 'tracegroup', 'formulaformat', 'format', 'formula', 'writer', 'database']:
        sql = "DELETE FROM " + table
        cur.execute(sql)
    commit()


def commit():
    conn.commit()


def rollback():
    conn.rollback()


def close():
    conn.close()
    cur.close()

def query(string):
    cur.execute(string)
    rows = cur.fetchall()
    return rows